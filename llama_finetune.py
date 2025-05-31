import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForLanguageModeling, get_cosine_schedule_with_warmup
from datasets import load_dataset
import os
from tqdm import tqdm
import wandb
import math
import argparse
import time
from utils import (
    log,
    init_process_group,
    replace_with_subscaf_linear,
    outer_update,
    get_subscaf_optimizer,
    main_parse_args,
)

def parse_args(args, remaining_args):
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--epoch", default=2, type=int)
    parser.add_argument("--use_tqdm", action="store_true")
    parser.add_argument("--use_log", action="store_true")
    parser.add_argument("--eval_freq", default=1000, type=int)

    new_args, _ = parser.parse_known_args(remaining_args)
    args = argparse.Namespace(**vars(args), **vars(new_args))

    return args

def evaluate(model,  eval_dataloader, epoch, device, rank, args):
    model.eval()
    eval_losses = []
    eval_accuracies = []
    log(f"start eval in {epoch} epoch")
    
    if rank == 0 and args.use_tqdm:
        pbar = tqdm(eval_dataloader, desc="Evaulation", ncols=100)
    
    with torch.no_grad():
        idx = 0
        for batch in eval_dataloader:
            batch = {k:v.to(device) for k, v in batch.items()}
            labels = batch['labels']

            output = model(**batch)
            logits = output.logits
            loss = output.loss

            if rank == 0:
                if loss is not None:
                    eval_losses.append(loss.item())
                    shift_logits = logits[..., :-1, :].contiguous()
                    shift_labels = labels[..., 1:].contiguous()
                    predictions = shift_logits.argmax(dim=-1)
                    correct = (predictions == shift_labels).float().mean()
                    eval_accuracies.append(correct.item())
                if args.use_tqdm:
                    pbar.set_postfix({"loss": loss.item()})
            if idx % 50 == 0 and args.use_log:
                log(f"finish {idx}\{len(eval_dataloader)}, please wait...")
            idx += 1
    
    metrics = {}
    if rank == 0 and eval_losses:
        metrics["loss"] = sum(eval_losses) / len(eval_losses)
        metrics["accuracy"] = sum(eval_accuracies) / len(eval_accuracies)
        metrics["perplexity"] = math.exp(metrics["loss"])
    if rank == 0:
        log(f"Epoch {epoch}, Eval Loss: {metrics['loss']:.4f}, "
                f"Eval Accuracy: {metrics['accuracy']:.4f}")
        if args.use_wandb:
            wandb.log({
                "eval/loss": metrics["loss"],
                "eval/accuracy": metrics["accuracy"],
                "eval/perplexity": metrics["perplexity"],
                "epoch": epoch
            })
        
    return metrics

def main(args):
    
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    device = f"cuda:{rank}"
    
    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained("/data/pretrained_models/llama-3.2-1b").to(device)
    tokenizer = AutoTokenizer.from_pretrained("t5-base")

    log("*" * 40)
    log("Start training with arguments")
    for k, v in vars(args).items():
        log(f"{k:30} {v}")
    log("*" * 40)
    
    # Load dataset
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
    
    # Tokenize function
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=False,  # Do not truncate, let group_texts handle lengths
            max_length=args.max_length,
            padding="max_length"
        )
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False
    )
    # Tokenize
    tokenized_datasets = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset["train"].column_names,
        desc="Running tokenizer on dataset",
    )

    train_dataset = tokenized_datasets['train']
    eval_dataset = tokenized_datasets["validation"]


    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=data_collator
    )
    
    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=data_collator
    )

    # revise model architecture
    trainable_param = [p for p in model.parameters() if p.requires_grad == True]
    param_before_comp = sum(p.numel() for p in model.parameters()) / 1_000_000
    trainable_param_before_comp = sum(p.numel() for p in model.parameters() if p.requires_grad == True) / 1_000_000
    if 'subscaf' in args.optimizer.lower():
        target_modules_list = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        jump_modules_list = ['5', '6', '7']
        num_subscaf_params, subscaf_params, lbd, comp_mat_rec = replace_with_subscaf_linear(model, 
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
    
    
    # Create optimizer
    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(trainable_param, 
                                    lr=args.lr, 
                                    momentum=args.momentum,
                                    nesterov=args.nesterov,
                                    weight_decay=args.weight_decay,
                                    dampening=args.dampening,
                                    foreach=False)
        # schedule
        schedule = get_cosine_schedule_with_warmup(optimizer,
                                                num_warmup_steps=args.warmup,
                                                num_training_steps=args.num_training_steps + 1)
    elif args.optimizer == 'subscafsgd':
        if args.per_layer_weight_update:
            optimizer_dict = get_subscaf_optimizer(args, param_groups, regular_params, subscaf_params, lbd, model)
        else:
            optimizer, schedule = get_subscaf_optimizer(args, param_groups, regular_params, subscaf_params, lbd, model)
    
    n_total_param = sum(p.numel() for p in model.parameters() if p.requires_grad is True)
    grad_accumulation = args.total_batch_size // args.batch_size
    log(f"\n{model}\n")
    log(f"Total params: {param_before_comp:.2f}M")
    if 'subscaf' in args.optimizer.lower():
        # XXX need to consider model.x
        all_param_after_replace = sum(p.numel() for p in model.parameters())
        log(f"Trainable params before compression: {trainable_param_before_comp:.2f}M")
        log(f"Trainable params after compression: {sum(p.numel() for p in model.parameters()) / 1_000_000:.2f}M")
        log(f"Trainable params with Subspace Scaffold Linear Layer: {num_subscaf_params / 1_000_000:.2f}M")
        log(f"All params: {all_param_after_replace / 1_000_000:.2f}M")
    else: 
        log(f"Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1_000_000:.2f}M")

    # Setup Wandb after distributed initialization
    if rank == 0 and args.use_wandb:
        run_config = dict(vars(args))
        run_config.update({
            "max_lr": run_config.pop("lr"),
            "world_size": world_size,
            "total_param": n_total_param,
            "device": device,
            "model": model.config.to_dict(),
        })
        wandb.init(
            project="SubScaf_Finetune",
            name=args.wandb_run_name
        )
        wandb.config.update(run_config, allow_val_change=True)

    for epoch in range(args.epoch):
        # Evaluation first
        log(f"start epoch: {epoch}")

        if epoch != 0:
            _ = evaluate(model, eval_dataloader, epoch, device, rank, args)
        
        # Then training
        model.train()
        if epoch == 0:
            update_time = time.time()

        if rank == 0 and args.use_tqdm:
            pbar = tqdm(train_dataloader, desc=f"Epoch {epoch}", ncols=100)
        training_length = len(train_dataloader)
    
        update_step = 0
        local_step = 0
        for batch in train_dataloader:
            local_step += 1
            batch = {k:v.to(device) for k, v in batch.items()}
            loss = model(**batch).loss
            scaled_loss = loss / grad_accumulation
            scaled_loss.backward()
            if rank == 0 and args.use_tqdm:
                pbar.set_postfix({"loss": loss.item()})

            if local_step % grad_accumulation != 0 or local_step == 0:
                continue

            # the below code is only executed during the update step
            if 'subscaf' not in args.optimizer.lower(): 
                for params in model.parameters():
                    dist.all_reduce(params.grad, op=dist.ReduceOp.AVG)
                schedule.step()
                optimizer.step()
                optimizer.zero_grad()
            elif not args.per_layer_weight_update and 'subscaf' in args.optimizer.lower():
                # because warmup will make the first step with lr 0, and it will cause lbd
                # to be nan. So we choose to update lr before step. And for consistency, sgd
                # also follow this setup
                schedule.step()
                optimizer.step()
                optimizer.zero_grad()

            if "subscaf" in args.optimizer.lower() and (local_step // grad_accumulation) % args.tau == 0:
                if (local_step // grad_accumulation) % args.update_cp_freq != 0:
                    gene_new_cp = False 
                else:
                    gene_new_cp = True
                if args.per_layer_weight_update:
                    outer_update(model, lbd, comp_mat_rec, target_modules_list, optimizer_dict, 
                                 subscaf_params, args, device, jump_modules_list, gene_new_cp)
                else:
                    outer_update(model, lbd, comp_mat_rec, target_modules_list, optimizer, 
                                 subscaf_params, args, device, jump_modules_list, gene_new_cp)


            update_step += 1
            update_time = time.time() - update_time

            if update_step == 1 and epoch == 0 :
                time_per_iter = update_time
            else:
                time_per_iter = 0.9 * time_per_iter + 0.1 * update_time
            
            if rank == 0:
                remain_total_seconds = time_per_iter * (training_length - update_step)
                #pbar.update(grad_accumulation)

            if rank == 0 and args.use_wandb:
                torch.cuda.empty_cache()
                record_dict = {
                    "loss": loss.item(),
                    "update_step": update_step,
                    "cuda_max_memory(GB)": torch.cuda.max_memory_allocated() / (1024 ** 3),
                }
                if "subscaf" in args.optimizer and args.per_layer_weight_update:
                    record_dict.update({"lr": optimizer_dict[next(model.parameters())].param_groups[0]["lr"]})
                else:
                    record_dict.update({"lr": optimizer.param_groups[0]["lr"]})

                wandb.log(record_dict, step=update_step,)

            if "subscaf" in args.optimizer and args.per_layer_weight_update:
                lr =  optimizer_dict[next(model.parameters())].param_groups[0]["lr"]
            else:
                lr = optimizer.param_groups[0]["lr"]

            if rank == 0 and args.use_log:
                #torch.cuda.empty_cache()
                cuda_mem_usage = f"{torch.cuda.max_memory_allocated() / (1024 ** 3):.3} GB"
                log(f"step: {update_step}/{training_length} Loss: {loss:.8f} Lr: {lr * 10000:.5f}e-4 Mem: {cuda_mem_usage}")
                if update_step % 10 == 0:
                    hours = int(remain_total_seconds // 3600)
                    minutes = int((remain_total_seconds % 3600) // 60)
                    seconds = int(remain_total_seconds % 60)
                    log(f"ETA: {hours:02d}:{minutes:02d}:{seconds:02d}")
            
            if update_step % args.eval_freq == 0:
                _ = evaluate(model, eval_dataloader, epoch, device, rank, args)
                model.train()
            update_time = time.time()
            
    
    # Final evaluation
    _ = evaluate(model, eval_dataloader, epoch, device, rank, args)

    dist.destroy_process_group()
    if args.use_tqdm and rank == 0:
        pbar.close()

if __name__ == "__main__":
    s_time = time.time()
    # Initialize distributed training first
    init_process_group()
    args, unknown_args = main_parse_args(None)
    args = parse_args(args, unknown_args)
    main(args)
    if args.use_log and dist.get_rank() == 0:
        total_time = time.time() - s_time
        hours = int(total_time // 3600)
        minutes = int((total_time % 3600) // 60)
        seconds = int(total_time % 60)
        log(f"Total Time: {hours:02d}:{minutes:02d}:{seconds:02d}")