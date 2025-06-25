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
from utils import (
    SubScafSGD, 
    SubScafLinear, 
    log, 
    init_process_group, 
    main_parse_args, 
    get_subscaf_optimizer,
    outer_update,
    replace_with_subscaf_layer,
    apply_activation_checkpointing,
)
from pickle import dump
from torch.amp import GradScaler




def parse_args(args, remaining_args):
    parser = argparse.ArgumentParser()
    parser.add_argument("--flash_attn", action="store_true", help="flash_attn is conflicted with mixed precision")
    parser.add_argument("--ckpt", action="store_true", help="checkpoint is conflicted with flash_attention")
    parser.add_argument("--change_cd", default=4000, type=int)
    new_args, _ = parser.parse_known_args(remaining_args)
    args = argparse.Namespace(**vars(args), **vars(new_args))
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
    if args.flash_attn:
       assert not args.ckpt, "flash-attention is conflicted with checkpoint"
       assert not args.mixed_precision, "flash-attention is conflicted with mixed precision"
       model_config._attn_implementation = "flash_attention_2"
    elif args.ckpt:
        assert not args.flash_attn, "flash-attention is conflicted with checkpoint"
        # we use eager attention for easy checkpoint implementation
        model_config._attn_implementation = "eager" 
    model = LlamaForCausalLM(config=model_config).to(device)
    if args.flash_attn:
        # flash-attn only support fp16 or bf16 precision
        model = model.to(torch.float16)
    if args.ckpt:
        model = apply_activation_checkpointing(model)
    for param in model.parameters():
        dist.broadcast(param, src=0)

    # mixed precision scaler
    if args.mixed_precision:
        assert not args.per_layer_weight_update, "mixed precision is conflicted with per layer weight update"
        if not (args.per_layer_weight_update and "subscaf" in args.optimizer):
            scaler = GradScaler()
        else:
            scaler_dict = {}
            for p in model.parameters():
                scaler_dict[p] = GradScaler()
        precision_map_dict = {'fp16': torch.float16, 'bf16': torch.bfloat16}
        precision = precision_map_dict[args.mixed_precision]
    else:
        scaler = None
        scaler_dict = None

    # replace module
    trainable_param = [p for p in model.parameters() if p.requires_grad == True]
    param_before_comp = sum(p.numel() for p in model.parameters()) / 1_000_000
    trainable_param_before_comp = sum(p.numel() for p in model.parameters() if p.requires_grad == True) / 1_000_000
    if 'subscaf' in args.optimizer.lower():
        target_modules_list = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        jump_modules_list = []
        # define nonlocal variables for replace module
        num_subscaf_params, subscaf_params, lbd, comp_mat_rec = replace_with_subscaf_layer(model, 
                                                                                            target_modules_list, 
                                                                                            device, 
                                                                                            args, 
                                                                                            jump_modules_list)
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

    # optimizer
    if args.optimizer == 'subscafsgd':
        if args.per_layer_weight_update:
            optimizer_dict = get_subscaf_optimizer(args, param_groups, regular_params, subscaf_params, lbd, model, scaler_dict)
        else:
            optimizer, schedule = get_subscaf_optimizer(args, param_groups, regular_params, subscaf_params, lbd, model)
    elif 'sgd' in args.optimizer:
        optimizer = torch.optim.SGD(trainable_param, 
                                    lr=args.lr, 
                                    momentum=args.momentum,
                                    nesterov=args.nesterov,
                                    weight_decay=args.weight_decay,
                                    dampening=args.dampening,
                                    foreach=False)
        # schedule
        if not args.constant_lr:
            schedule = get_cosine_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=args.warmup,
                                                    num_training_steps=args.num_training_steps + 1)
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(trainable_param,
                                     lr=args.lr,
                                     )

        # schedule
        if not args.constant_lr:
            schedule = get_cosine_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=args.warmup,
                                                    num_training_steps=args.num_training_steps + 1)

    n_total_params = sum(p.numel() for p in model.parameters())
    if rank == 0: 
        if args.use_tqdm:
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
            wandb.init(project="SubScaf", name=args.wandb_run_name)
            wandb.config.update(run_config, allow_val_change=True)

    local_step = 0
    update_step = 0
    token_seen = 0
    token_seen_before = 0
    ewm_loss = 0
    grad_dict = {}
    pad_idx = tokenizer.pad_token_id
    update_time = time.time()

    for _, batch in enumerate(dataloader):
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
        if rank == 0 and args.use_tqdm:
            pbar.set_postfix({"loss": loss.item()})



        if local_step % grad_accumulation != 0 or local_step == 0:
            continue

        # the below code is only executed during the update step

        # add grad cliping
        if args.grad_clip != 0.0:
            torch.nn.utils.clip_grad_norm_(trainable_param, args.grad_clip)

        if rank == 0 and args.use_tqdm:
            pbar.update(1)
        
        if args.optimizer.lower() == 'sgd': 
            for params in model.parameters():
                if params.grad is not None:
                    dist.all_reduce(params.grad, op=dist.ReduceOp.AVG)
            
            if not args.constant_lr:
                schedule.step()
            if args.mixed_precision:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()

        if 'fedavg' in args.optimizer.lower():
            if not args.constant_lr:
                schedule.step()
            if args.mixed_precision:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()

        # because warmup will make the first step with lr 0, and it will cause lbd
        # to be nan. So we choose to update lr before step. And for consistency, sgd
        # also follow this setup
        if not args.per_layer_weight_update and 'subscaf' in args.optimizer.lower():
            if not args.constant_lr:
                schedule.step()
            if args.mixed_precision:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            if args.gene_method == 'svd' and (update_step + 1) % args.update_cp_freq == 0:
                for p in model.parameters():
                    grad_dict[p] = p.grad
                    p.grad = None
            optimizer.zero_grad()

        update_step += 1
        update_time = time.time() - update_time

        if args.gene_method == 'svd' and args.change_cd <= update_step:
            args.gene_method = 'cd'
            # clear grad_dict
            grad_dict = {}

        if "fedavg" in args.optimizer.lower() and update_step % args.tau == 0:
            with torch.no_grad():
                for p in model.parameters():
                    dist.all_reduce(p, op=dist.ReduceOp.AVG)

        if "subscaf" in args.optimizer.lower() and update_step % args.tau == 0:
            if update_step % args.update_cp_freq != 0:
                gene_new_cp = False 
            else:
                gene_new_cp = True
            if args.per_layer_weight_update:
                outer_update(model, lbd, comp_mat_rec, target_modules_list, optimizer_dict, 
                                subscaf_params, args, device, jump_modules_list, gene_new_cp)
            else:
                outer_update(model, lbd, comp_mat_rec, target_modules_list, optimizer, 
                                subscaf_params, args, device, jump_modules_list, gene_new_cp, grad_dict)

        if rank == 0 and args.use_log:
            if update_step == 1:
                time_per_iter = update_time
            else:
                time_per_iter = 0.9 * time_per_iter + 0.1 * update_time

            remain_total_seconds = time_per_iter * (args.num_training_steps - update_step)
        token_in_update = token_seen - token_seen_before
        token_seen_before = token_seen
        batch_in_update = grad_accumulation * world_size
        throughput_examples = args.total_batch_size * world_size / update_time

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


        
        if args.use_log and rank == 0:
            if "subscaf" in args.optimizer and args.per_layer_weight_update:
                lr =  optimizer_dict[next(model.parameters())].param_groups[0]["lr"]
            else:
                lr = optimizer.param_groups[0]["lr"]

            if ewm_loss == 0:
                ewm_loss = loss.item()
            else:
                ewm_loss = 0.9 * ewm_loss + 0.1 * loss.item()

            #cuda_mem_usage = f"{torch.cuda.max_memory_allocated() / (1024 ** 3):.3f} GB"
            cuda_mem_usage = f"{torch.cuda.max_memory_allocated() / (1024 ** 3):.3f} GB"
            torch.cuda.reset_peak_memory_stats()
            log(f"step: {update_step}/{args.num_training_steps} Loss: {loss.item():.4f}\{ewm_loss:.3f} Lr: {lr * 1000:.3f}e-3 Mem: {cuda_mem_usage} Throughput_tokens: {throughput_examples:.4f}")
            #log(f"time:{update_time}, step: {update_step}, examples: {args.total_batch_size * world_size}")
            if update_step % 10 == 0:
                hours = int(remain_total_seconds // 3600)
                minutes = int((remain_total_seconds % 3600) // 60)
                seconds = int(remain_total_seconds % 60)
                log(f"ETA: {hours:02d}:{minutes:02d}:{seconds:02d} ")

        update_time = time.time()

    log("finish training")
    if rank == 0:
        if args.mem_monitor:
            # NOTE this will generate a giant file to record the memory consumption during training
            # so make sure the training step is few enough to make this file possible to store.
            s = torch.cuda.memory._snapshot()
            with open(f"snapshot.pickle", "wb") as f:
                dump(s, f)
        if args.use_tqdm:
            pbar.close()
        if args.use_wandb:
            wandb.finish()

if __name__ == "__main__":
    init_process_group()
    args, unknown = main_parse_args(None)
    args = parse_args(args, unknown)
    # setting baseline optimizer list, if we choose one of optimizer in that,
    # then the optimizer procedure would be a little bit different
    baseline_optimizer = ['sgd', 'adam']
    s_time = time.time()
    main(args)
    if args.use_log:
        total_time = time.time() - s_time
        hours = int(total_time // 3600)
        minutes = int((total_time % 3600) // 60)
        seconds = int(total_time % 60)
        log(f"Total Time: {hours:02d}:{minutes:02d}:{seconds:02d}")
    dist.destroy_process_group()







