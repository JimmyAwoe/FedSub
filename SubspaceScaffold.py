from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoConfig,
    LlamaConfig,
    LlamaForCausalLM,
    DataCollatorForLanguageModeling,
    get_cosine_schedule_with_warmup,
)
from datasets.distributed import split_dataset_by_node
import torch
import torch.distributed as dist
import os
from loguru import logger
from tqdm import tqdm
import argparse
from torch.utils.data import DataLoader
import wandb
import time
import torch.nn as nn
from utils import SubScafSGD, SubScafLinear, gene_random_matrix, log, set_seed, init_process_group
from pickle import dump
from torch.amp import GradScaler




def parse_args(args):
    parser = argparse.ArgumentParser()

    # Training 
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", default=16, type=int, help="batch size per round")
    parser.add_argument("--total_batch_size", default=32, type=int, help="batch size per step")
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--num_training_steps", type=int, default=10000)
    parser.add_argument("--grad_clip", type=float, default=0.0)
    parser.add_argument("--warmup", type=int, default=1000)
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--mixed_precision", default=None, type=str, choices=['bf16', 'fp16'])

    # subscaf
    parser.add_argument("--comp_dim", default=64, type=int, help="the compression dimension")
    parser.add_argument("--tau", type=int, help="inner loop steps")
    parser.add_argument("--gene_method", default='cd', type=str, 
                        help="set the method to generate compression matrix")

    # model
    parser.add_argument("--model_config", type=str, default="configs/llama_60m.json")

    # optimizer
    parser.add_argument("--optimizer", choices=['subscafsgd', 'subscafadam', 'scaf', 'sgd', 'adam'], default='subscafsgd',
                        type=str, help="assign the optimization algorithm")
    parser.add_argument("--momentum", default=0, type=float)
    parser.add_argument("--dampening", default=0, type=float)
    parser.add_argument("--nesterov", action="store_true")
    parser.add_argument("--per_layer_weight_update", action="store_true")

    # wandb
    parser.add_argument("--wandb_run_name", default='subscaf_sgd')

    # memory monitor
    parser.add_argument("--mem_monitor", action="store_true")
    args = parser.parse_args(args)

    return args


def mem():
    torch.cuda.empty_cache()
    log('memory allocated: ' + str((torch.cuda.memory_allocated() / (1024 ** 3))) + 'GB')
    log('memory reserved: ' + str(torch.cuda.memory_reserved() / (1024 ** 3)) + 'GB')
    log('max memory allocated: ' + str(torch.cuda.max_memory_allocated() / (1024 ** 3)) + 'GB')
    log('max memory reserved: ' + str(torch.cuda.max_memory_reserved() / (1024 ** 3)) + 'GB')

def main(args):
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    set_seed(args.seed)
    if rank == 0 and args.mem_monitor:
        torch.cuda.memory._record_memory_history(enabled='all')
    log("Process group initialize")
    log("*" * 40)
    log("Start training with arguments")
    for k, v in vars(args).items():
        log(f"{k:30} {v}")
    log("*" * 40)

    device = f"cuda:{local_rank}"

    # ensure grad_accumulation is integer
    assert args.total_batch_size % args.batch_size == 0, "grad accumulation must be integer"
    grad_accumulation = args.total_batch_size // args.batch_size

    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained("t5-base")

    # dataset
    ds = load_dataset("/data/datasets/c4_en", split="train", streaming=True)
    
    def tokenize_fun(data):
        output = tokenizer(data["text"],
                           truncation=True,
                           max_length=args.max_length,
                           padding="max_length",)
        return output

    dataset = ds.map(tokenize_fun, batched=True, remove_columns=["url", "text", "timestamp"])
    collate_fn = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    dataset = split_dataset_by_node(dataset, rank, world_size)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=collate_fn)

    # model
    model_config = AutoConfig.from_pretrained(args.model_config)
    model = LlamaForCausalLM(model_config).to(device)
    for param in model.parameters():
        dist.broadcast(param, src=0)

    # optimizer
    trainable_param = [p for p in model.parameters() if p.requires_grad == True]
    param_before_comp = sum(p.numel() for p in model.parameters()) / 1_000_000
    trainable_param_before_comp = sum(p.numel() for p in model.parameters() if p.requires_grad == True) / 1_000_000
    if 'subscaf' in args.optimizer.lower():
        target_modules_list = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        # define nonlocal variables for replace module
        num_subscaf_params = 0
        subscaf_params = []
        lbd = []
        comp_mat_rec = {} 
        def replace_module(model, target_modules_list):
            """replace Linear module in model into SubScafLinear module"""
            nonlocal num_subscaf_params, subscaf_params, lbd, comp_mat_rec
            for name, module in model.named_children():
                # only revise module with param_name has "mlp" or "attn"
                if isinstance(module, nn.Linear) and any(target_key in name for target_key in target_modules_list):
                    log(f"enable Subspace Scaffold for weights in module: {name}")

                    # create compression matrix only when new shape demand occur
                    if (args.comp_dim, module.in_features) not in comp_mat_rec.keys():
                        #comp_mat = gene_random_matrix(module.out_features, args.comp_dim, args.gene_method).to(device)
                        comp_mat = gene_random_matrix(args.comp_dim, module.in_features, args.gene_method).to(device)
                        dist.broadcast(comp_mat, src=0)
                        comp_mat_rec[(args.comp_dim, module.in_features)] = comp_mat
                    else:
                        comp_mat = comp_mat_rec[(args.comp_dim, module.in_features)]

                    # substitue all Linear module into SubScafLinear module
                    new_layer = SubScafLinear(args.comp_dim, comp_mat, module)
                    setattr(model, name, new_layer)

                    # record the subscaf module total parameters
                    num_subscaf_params += sum(p.numel() for p in new_layer.parameters())

                    # add parameter into trainable parameter
                    subscaf_params += [p for p in new_layer.parameters()]

                    # initialize lambda
                    #lbd.append(torch.zeros((args.comp_dim, module.in_features), device=device, requires_grad=False))
                    lbd.append(torch.zeros((module.out_features, args.comp_dim), device=device, requires_grad=False))

                else:
                    replace_module(module, target_modules_list)
        
        @torch.no_grad()
        def outer_update(model, lbd, comp_mat_rec, target_modules_list, opt):
            def subscaf_outer_update(model, lbd, comp_mat_rec, target_modules_list):
                """carry out one outer update for subspace scaffold algorithm"""
                nonlocal idx, new_comp_mat_rec
                for name, module in model.named_children():
                    if isinstance(module, nn.Linear) and any(target_key in name for target_key in target_modules_list):
                        # all_reduce b
                        avg_b = module.b.detach().clone()
                        dist.all_reduce(avg_b, op=dist.ReduceOp.AVG)

                        # generate new compression matrix
                        if (args.comp_dim, module.in_features) not in new_comp_mat_rec.keys():
                            #new_comp_mat = gene_random_matrix(module.out_features, args.comp_dim, args.gene_method).to(device)
                            new_comp_mat = gene_random_matrix(args.comp_dim, module.in_features, args.gene_method).to(device)
                            dist.broadcast(new_comp_mat, src=0)
                            new_comp_mat_rec[(args.comp_dim, module.in_features)] = new_comp_mat
                        else:
                            new_comp_mat = new_comp_mat_rec[(args.comp_dim, module.in_features)]

                        update_factor = comp_mat_rec[(args.comp_dim, module.in_features)] @ new_comp_mat.T

                        # update momentum_buffer
                        if args.momentum > 0:
                            if not args.per_layer_weight_update:
                                opt.update_m(module.b, - avg_b @ update_factor / (args.lr * args.tau))
                                #opt.update_m(module.b, update_factor = update_factor)
                            else:
                                opt[module.b].update_m(module.b, - avg_b @ update_factor / (args.lr * args.tau))
                                #opt[module.b].update_m(module.b, update_factor=update_factor)
                                
                        # update lbd for every modules
                        lbd[idx] = (lbd[idx] + module.b - avg_b) @ update_factor 
                        assert lbd[idx].shape == (module.out_features, args.comp_dim)

                        # update compression matrix, b and x
                        new_x = module.x + avg_b @ comp_mat_rec[(args.comp_dim, module.in_features)] 
                        module.update(comp_mat=new_comp_mat, x=new_x, b=True)

                        # update idx
                        idx += 1
                    else:
                        subscaf_outer_update(module, lbd, comp_mat_rec, target_modules_list)
            idx = 0
            new_comp_mat_rec = {}
            subscaf_outer_update(model, lbd, comp_mat_rec, target_modules_list)
            comp_mat_rec = new_comp_mat_rec

            # update lbd
            if not args.per_layer_weight_update:
                opt.update_lbd(lbd)
            else:
                for (p, l) in zip(subscaf_params, lbd):
                    opt[p].update_lbd(lbd=[l])
        #mem()
        replace_module(model, target_modules_list)
        #mem()
        id_subscaf_params = [id(p) for p in subscaf_params]
        # make parameters without "is_comp" to another group
        regular_params = [p for p in model.parameters() if id(p) not in id_subscaf_params]
        # make parameters with "is_comp" to a single group
        param_groups = [{'params': regular_params, 'is_comp': False}, 
                        {'params': subscaf_params, 'is_comp': True, 'lbd': lbd, 'tau': args.tau, 
                         'compression_dim': args.comp_dim}]

    log(f"\n{model}\n")
    log(f"Total params: {param_before_comp:.2f}M")
    if 'subscaf' in args.optimizer.lower():
        log(f"Trainable params before compression: {trainable_param_before_comp:.2f}M")
        log(f"Trainable params after compression: {sum(p.numel() for p in model.parameters()) / 1_000_000:.2f}M")
        log(f"Total params with Subspace Scaffold enabled: {num_subscaf_params / 1_000_000:.2f}M")
    else: 
        log(f"Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1_000_000:.2f}M")

    if args.optimizer == 'subscafsgd':
        if not args.per_layer_weight_update:
            optimizer = SubScafSGD(param_groups, 
                                lr=args.lr, 
                                tau=args.tau, 
                                compression_dim=args.comp_dim,
                                foreach=False,
                                momentum=args.momentum,
                                dampening=args.dampening,
                                )
            # we add 1 to num_training_steps for avoiding lr become zero when training, which would cause
            # lbd to be nan
            schedule = get_cosine_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=args.warmup,
                                                    num_training_steps=args.num_training_steps + 1)
        else:
            optimizer_dict = {p: SubScafSGD([{'params': p, 'is_comp': False}], 
                                            lr=args.lr, 
                                            tau=args.tau, 
                                            compression_dim=args.comp_dim,
                                            momentum=args.momentum,
                                            dampening=args.dampening,
                                            foreach=False) for p in regular_params}
            for (p, l) in zip(subscaf_params, lbd):
                optimizer_dict.update({p: SubScafSGD([{'params':p, 'is_comp': True, 'lbd': [l]}],
                                                    lr=args.lr,
                                                    tau=args.tau,
                                                    compression_dim=args.comp_dim,
                                                    foreach=False,
                                                    dampening=args.dampening,
                                                    momentum=args.momentum)})
            def optimizer_hook(p):
                if p.grad is None:
                    return
                dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)
                schedule_dict[p].step()
                optimizer_dict[p].step()
                optimizer_dict[p].zero_grad()

            schedule_dict = {}
            for p in model.parameters():
                if p.requires_grad:
                    # we add 1 to num_training_steps for avoiding lr become zero when training, which would cause
                    # lbd to be nan
                    # because in this condition, every backward would call optimizer_hook once, hence push the lr,
                    # so in the case of gradient accumulation, we should correspondily longer the warmup and training 
                    # step
                    schedule_dict[p] = get_cosine_schedule_with_warmup(optimizer_dict[p],
                                                                    num_warmup_steps=args.warmup * grad_accumulation,
                                                                    num_training_steps=args.num_training_steps * grad_accumulation + 1)
                    p.register_post_accumulate_grad_hook(optimizer_hook)
    #elif args.optimizer == 'subscafadam':
        #if not args.per_layer_weight_update:
            #optimizer = SubScafAdam(param_groups, 
                                #lr=args.lr, 
                                #foreach=False,
                                #fused=True,
                                #)
            ## we add 1 to num_training_steps for avoiding lr become zero when training, which would cause
            ## lbd to be nan
            #schedule = get_cosine_schedule_with_warmup(optimizer,
                                                    #num_warmup_steps=args.warmup,
                                                    #num_training_steps=args.num_training_steps + 1)
        #else:
            #optimizer_dict = {p: SubScafAdam([{'params': p, 'is_comp': False}], 
                                            #lr=args.lr, 
                                            #foreach=False,
                                            #fused=True) for p in regular_params}
            #for (p, l) in zip(subscaf_params, lbd):
                #optimizer_dict.update({p: SubScafAdam([{'params':p, 'is_comp': True, 'lbd': [l]}],
                                                    #lr=args.lr,
                                                    #foreach=False,
                                                    #fused=True)})
            #def optimizer_hook(p):
                #if p.grad is None:
                    #return
                #schedule_dict[p].step()
                #optimizer_dict[p].step()
                #optimizer_dict[p].zero_grad()
            #schedule_dict = {}
            #for p in model.parameters():
                #if p.requires_grad:
                    ## we add 1 to num_training_steps for avoiding lr become zero when training, which would cause
                    ## lbd to be nan
                    #schedule_dict[p] = get_cosine_schedule_with_warmup(optimizer_dict[p],
                                                                    #num_warmup_steps=args.warmup,
                                                                    #num_training_steps=args.num_training_steps + 1)
                    #p.register_post_accumulate_grad_hook(optimizer_hook)
    elif args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(trainable_param, 
                                    lr=args.lr, 
                                    momentum=args.momentum,
                                    nesterov=args.nesterov,
                                    dampening=args.dampening,
                                    foreach=False)
        # schedule
        schedule = get_cosine_schedule_with_warmup(optimizer,
                                                num_warmup_steps=args.warmup,
                                                num_training_steps=args.num_training_steps + 1)
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(trainable_param,
                                     lr=args.lr,
                                     )

        # schedule
        schedule = get_cosine_schedule_with_warmup(optimizer,
                                                num_warmup_steps=args.warmup,
                                                num_training_steps=args.num_training_steps + 1)

    n_total_params = sum(p.numel() for p in model.parameters())
    if rank == 0: 
        pbar = tqdm(total=args.num_training_steps, desc="update step", ncols=80)
        if args.use_wandb:
            run_config = dict(vars(args))
            run_config.update({
                "max_lr": run_config.pop("lr"),
                "total_params_M": n_total_params / 1_000_000,
                "model": model_config.to_dict(),
                "world_size": world_size,
                "devive": str(device),
            })
            args.wandb_run_name = f"{args.wandb_run_name}-lr{args.lr}"
            if args.optimizer == 'subscaf':
                args.wandb_run_name += f"-tau{args.tau}-CPDim{args.comp_dim}"
            wandb.init(project="SubScaf", name=args.wandb_run_name)
            wandb.config.update(run_config, allow_val_change=True)

    local_step = 0
    update_step = 0
    token_seen = 0
    token_seen_before = 0
    pad_idx = tokenizer.pad_token_id
    update_time = time.time()
    if args.mixed_precision:
        scaler = GradScaler()
        precision_map_dict = {'fp16': torch.float16, 'bf16': torch.bfloat16}
        precision = precision_map_dict[args.mixed_precision]

    for batch_idx, batch in enumerate(dataloader):
        local_step += 1
        if update_step > args.num_training_steps:
            log(f"attain assigned training step {args.num_training_steps}. Stop Training")
            print(f"Rank {rank} stopping training")
            break
        batch = {k: v.to(device) for k, v in batch.items()}
        labels = batch["input_ids"].clone()
        labels[labels == pad_idx] = -100
        token_seen += (batch["input_ids"] != pad_idx).sum().item() * world_size

        if args.mixed_precision:
            with torch.amp.autocast(device_type='cuda', dtype=precision):
                loss = model(**batch).loss
            scaler.scale(loss / grad_accumulation).backward()
        else:
            loss = model(**batch).loss
            scaled_loss = loss / grad_accumulation
            scaled_loss.backward()
        if rank == 0:
            pbar.set_postfix({"loss": loss.item()})



        if local_step % grad_accumulation != 0 or local_step == 0:
            continue

        # the below code is only executed during the update step

        # add grad cliping
        if args.grad_clip != 0.0:
            torch.nn.utils.clip_grad_norm_(trainable_param, args.grad_clip)

        if rank == 0:
            pbar.update(1)
        
        if not args.per_layer_weight_update or 'subscaf' not in args.optimizer.lower(): 
            for params in model.parameters():
                dist.all_reduce(params.grad, op=dist.ReduceOp.AVG)

        # because warmup will make the first step with lr 0, and it will cause lbd
        # to be nan. So we choose to update lr before step. And for consistency, sgd
        # also follow this setup
            schedule.step()
            optimizer.step()
            optimizer.zero_grad()

        if "subscaf" in args.optimizer.lower() and (local_step // grad_accumulation) % args.tau == 0:
            if args.per_layer_weight_update:
                outer_update(model, lbd, comp_mat_rec, target_modules_list, optimizer_dict)
            else:
                outer_update(model, lbd, comp_mat_rec, target_modules_list, optimizer)


        update_step += 1
        update_time = time.time() - update_time
        token_in_update = token_seen - token_seen_before
        token_seen_before = token_seen
        batch_in_update = grad_accumulation * world_size

        if rank == 0 and args.use_wandb:
            torch.cuda.empty_cache()
            record_dict = {
                "loss": loss.item(),
                "update_step": update_step,
                "throughput_tokens": token_in_update / update_time,
                "throughput_examples": args.total_batch_size * world_size / update_time,
                "throughput_batchs": batch_in_update,
                "cuda_max_memory(GB)": torch.cuda.max_memory_allocated() / (1024 ** 3),
            }
            if "subscaf" in args.optimizer and args.per_layer_weight_update:
                record_dict.update({"lr": optimizer_dict[next(model.parameters())].param_groups[0]["lr"]})
            else:
                record_dict.update({"lr": optimizer.param_groups[0]["lr"]})

            wandb.log(record_dict, step=update_step,)

        update_time = time.time()

    log("finish training")
    if rank == 0:
        if args.mem_monitor:
            # NOTE this will generate a giant file to record the memory consumption during training
            # so make sure the training step is few enough to make this file possible to store.
            s = torch.cuda.memory._snapshot()
            with open(f"snapshot.pickle", "wb") as f:
                dump(s, f)
        pbar.close()
        if args.use_wandb:
            wandb.finish()

if __name__ == "__main__":
    init_process_group()
    args = parse_args(None)
    # setting baseline optimizer list, if we choose one of optimizer in that,
    # then the optimizer procedure would be a little bit different
    baseline_optimizer = ['sgd', 'adam']
    main(args)







