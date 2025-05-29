import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForLanguageModeling
from datasets import load_dataset
import os
from tqdm import tqdm
import wandb
import math
import argparse
from utils import (
    gene_random_matrix,
    SubScafLinear,
    log,
    init_process_group,
    replace_with_subscaf_linear,
    outer_update,
    set_seed,
    SubScafSGD
)




def train_step(model, optimizer, batch, labels):
    # Forward pass
    output = model(**batch)
        
    # Compute loss (only on the last stage)
    loss = None
    logits = None
    logits = output[0] if isinstance(output, tuple) else output
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous() 
    loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1)
    )
    # eval state
    if model.training == False:
        return loss, logits
    
    # Backward pass
    if loss is not None:
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    return loss


def evaluate(model, optimizer, eval_dataloader, epoch, device, rank, args):
    eval_losses = []
    eval_accuracies = []
    
    pbar = tqdm(eval_dataloader, desc="Evaulation", ncols=160)
    
    with torch.no_grad():
        for batch in pbar:
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            
            loss, logits = train_step(model, optimizer, input_ids, labels)

            if rank == 0:
                if loss is not None:
                    eval_losses.append(loss.item())
                    shift_logits = logits[..., :-1, :].contiguous()
                    shift_labels = labels[..., 1:].contiguous()
                    predictions = shift_logits.argmax(dim=-1)
                    correct = (predictions == shift_labels).float().mean()
                    eval_accuracies.append(correct.item())
                    # Update progress bar with current loss
                    pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    metrics = {}
    if rank == 0 and eval_losses:
        metrics["loss"] = sum(eval_losses) / len(eval_losses)
        metrics["accuracy"] = sum(eval_accuracies) / len(eval_accuracies)
        metrics["perplexity"] = math.exp(metrics["loss"])
    if dist.get_rank() == dist.get_world_size() - 1:
        print(f"Epoch {epoch}, Eval Loss: {metrics['loss']:.4f}, "
                f"Eval Accuracy: {metrics['accuracy']:.4f}")
        if config.get('use_wandb', False):
            wandb.log({
                "eval/loss": metrics["loss"],
                "eval/accuracy": metrics["accuracy"],
                "eval/perplexity": metrics["perplexity"],
                "epoch": epoch
            })
        
    return metrics

def train_epoch(stage_module, schedule, optimizer, train_dataloader, epoch, config):
    stage_module.train()
    
    if hasattr(train_dataloader.dataset, 'update_epoch'):
        train_dataloader.dataset.update_epoch(epoch)
    
    # Progress bar only on main process
    if dist.get_rank() == dist.get_world_size() - 1:
        pbar = tqdm(train_dataloader, desc=f"Epoch {epoch}")
    else:
        pbar = train_dataloader
        
    for batch in pbar:
        input_ids = batch["input_ids"].to(f"cuda:{dist.get_rank()}").contiguous()
        labels = batch["labels"].to(f"cuda:{dist.get_rank()}").contiguous()
        indices = batch.get("indices", None)
        if indices is not None:
            indices = indices.to(f"cuda:{dist.get_rank()}").contiguous()

        loss = train_step(schedule, optimizer, input_ids, labels, indices, stage_module)
        
        if dist.get_rank() == dist.get_world_size() - 1:
            if loss is not None:
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})


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
    parser.add_argument("--nesterov", action="store_true")
    parser.add_argument("--per_layer_weight_update", action="store_true")

    # wandb
    parser.add_argument("--wandb_run_name", default='subscaf_sgd')

    # memory monitor
    parser.add_argument("--mem_monitor", action="store_true")
    args = parser.parse_args(args)

    return args

def main(args):
    
    set_seed(args.seed)
    rank = os.environ.get("RANK", 0)
    local_rank = os.environ.get("LOCAL_RANK", 0)
    world_size = os.environ.get("WORLD_SIZE", 1)
    device = f"cuda:{rank}"
    
    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained("/data/pretrained_models/Llama-3.2-1b")
    tokenizer = AutoTokenizer.from_pretrained("t5-base")
    
    # Load dataset
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
    
    # Tokenize function
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=False,  # Do not truncate, let group_texts handle lengths
            max_lenth=args.max_length,
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
    
    
    # Create optimizer
    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, nesterov=args.nesterov)
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    elif args.optimizer == 'subscafsgd':
        optimizer_dict = {p: SubScafSGD([{'params': p, 'is_comp': False}], 
                                        lr=args.lr, 
                                        tau=args.tau, 
                                        compression_dim=args.comp_dim,
                                        nesterov=args.nesterov,
                                        momentum=args.momentum,
                                        weight_decay=args.weight_decay,
                                        dampening=args.dampening,
                                        foreach=False) for p in regular_params}
        for (p, l) in zip(subscaf_params, lbd):
            optimizer_dict.update({p: SubScafSGD([{'params':p, 'is_comp': True, 'lbd': [l]}],
                                                lr=args.lr,
                                                tau=args.tau,
                                                compression_dim=args.comp_dim,
                                                foreach=False,
                                                weight_decay=args.weight_decay,
                                                nesterov=args.nesterov,
                                                dampening=args.dampening,
                                                momentum=args.momentum)})
        def optimizer_hook(p):
            if p.grad is None:
                return
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
    
    n_total_param = sum(p.numel() for p in model.parameters() if p.require_grads is True)

    # Setup Wandb after distributed initialization
    if rank == 0:
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

        evaluate(stage_module, schedule, optimizer, eval_dataloader,epoch,config)

        stage_module.eval()
        eval_losses = []
        eval_accuracies = []
    
        # Add progress bar only on the last rank
        if dist.get_rank() == dist.get_world_size() - 1:
            pbar = tqdm(eval_dataloader, desc="Evaluating")
        else:
            pbar = eval_dataloader
    
        with torch.no_grad():
            for batch in pbar:
                input_ids = batch["input_ids"].to(f"cuda:{dist.get_rank()}")
                labels = batch["labels"].to(f"cuda:{dist.get_rank()}")
            
                loss, logits = train_step(schedule, optimizer, input_ids, labels, stage_module=stage_module)

                if dist.get_rank() == dist.get_world_size() - 1:
                    if loss is not None:
                        eval_losses.append(loss.item())
                        shift_logits = logits[..., :-1, :].contiguous()
                        shift_labels = labels[..., 1:].contiguous()
                        predictions = shift_logits.argmax(dim=-1)
                        correct = (predictions == shift_labels).float().mean()
                        eval_accuracies.append(correct.item())
                        # Update progress bar with current loss
                        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
        metrics = {}
        if dist.get_rank() == dist.get_world_size() - 1 and eval_losses:
            metrics["loss"] = sum(eval_losses) / len(eval_losses)
            metrics["accuracy"] = sum(eval_accuracies) / len(eval_accuracies)
            metrics["perplexity"] = math.exp(metrics["loss"])
        if dist.get_rank() == dist.get_world_size() - 1:
            print(f"Epoch {epoch}, Eval Loss: {metrics['loss']:.4f}, "
                    f"Eval Accuracy: {metrics['accuracy']:.4f}")
            if config.get('use_wandb', False):
                wandb.log({
                    "eval/loss": metrics["loss"],
                    "eval/accuracy": metrics["accuracy"],
                    "eval/perplexity": metrics["perplexity"],
                    "epoch": epoch
                })
        
        # Then training
        train_epoch(stage_module, schedule, optimizer, train_dataloader, epoch, config)
    
    # Final evaluation
    metrics = evaluate(stage_module, schedule, optimizer, eval_dataloader)
    if dist.get_rank() == dist.get_world_size() - 1:
        print(f"Final Eval Loss: {metrics['loss']:.4f}, "
              f"Final Eval Accuracy: {metrics['accuracy']:.4f}")
        if config.get('use_wandb', False):
            wandb.log({
                "eval/final_loss": metrics["loss"],
                "eval/final_accuracy": metrics["accuracy"],
                "eval/final_perplexity": metrics["perplexity"]
            })
            wandb.finish()

if __name__ == "__main__":
    # Initialize distributed training first
    init_process_group()
    args = parse_args(None)
    main(args)