from datasets import load_dataset 
from transformers import (
    AutoTokenizer,
    AutoConfig,
    LlamaForCausalLM,
    DataCollatorForLanguageModeling,
    get_cosine_schedule_with_warmup,
)

import torch
import torch.distributed as dist
import os
from loguru import logger
import tqdm
import argparse
from torch.utils.data import DataLoader
import wandb
import time




def parse_args(args):
    parser = argparse.ArgumentParser()

    # Training 
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", default=16, help="batch size per round")
    parser.add_argument("--total_batch_size", default=32, help="batch size per step")
    parser.add_argument("--max_length", default=1024)
    parser.add_argument("--num_training_steps", default=10000)
    parser.add_argument("--grad_clip", default=0.0)
    parser.add_argument("--num_warmup", default=1000)

    # model
    parser.add_argument("--model_config", default="configs/llama_60m.json")




    args = parser.parse_args(args)
    return args


def log(info):
    logger.info(f"[{int(os.environ.get("RANK"))}]: " + info)

def ddp_setup():
    dist.init_process_group(backend="nccl")



def main(args):
    rank = int(os.environ.get("RANK"))
    local_rank = int(os.environ.get("LOCAL_RANK"))
    world_size = int(os.environ.get("WORLD_SIZE"))

    if rank == 0:
        log("Process group initialize")
        wandb.init(project="SubScaf")
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
    ds = load_dataset("/data/datasets/c4/en", split="train", streaming=True)
    
    def tokenize_fun(data):
        output = tokenizer(data["text"],
                           truncation=True,
                           max_length=args.max_length,
                           padding="max_length",)
        return output

    dataset = ds.map(tokenize_fun, batched=True, remove_columns=["url", "text", "timestamp"])
    dataset = split_
    collate_fn = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=collate_fn)

    # model
    model_config = AutoConfig.from_pretrained(args.model_config)
    model = LlamaForCausalLM(model_config).to(device)

    # optimizer
    trainable_param = [p for p in model.parameters if p.requires_grad == True]
    optimizer = torch.optim.AdamW(trainable_param, lr=args.lr)

    # schedule
    schedule = get_cosine_schedule_with_warmup(optimizer,
                                               num_warmup_steps=args.num_warmup,
                                               num_training_steps=args.num_training_steps)

    run_config = dict(vars(args))
    run_config.update({
        "max_lr": run_config.pop("lr"),
        "model": model_config.to_dict(),
        "world_size": world_size,
        "devive": str(device),
    })

    if rank == 0:
        wandb.config.update(run_config, allow_val_change=True)
        pbar = tqdm(args.num_training_steps, desc="update step", ncols=80)

    local_step = 0
    update_step = 0
    token_seen = 0
    token_seen_before = 0
    pad_idx = tokenizer.pad_token_id
    update_time = time.time()

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

        loss = model(**batch, labels=labels).loss
        scaled_loss = loss / grad_accumulation
        scaled_loss.backward()



        if local_step % grad_accumulation != 0:
            continue

        # the below code is only executed during the update step

        # add grad cliping
        if args.grad_clip != 0.0:
            torch.nn.utils.clip_grad_norm_(trainable_param, args.grad_clip)

        if rank == 0:
            pbar.update(1)

        optimizer.step()
        schedule.step()
        optimizer.zero_grad()

        update_step += 1
        update_time = time.time() - update_time
        token_in_update = token_seen - token_seen_before
        token_seen_before = token_seen
        batch_in_update = grad_accumulation * world_size

        if rank == 0:
            wandb.log({
                "loss": loss.item(),
                "lr": optimizer.param_groups[0]["lr"],
                "update_step": update_step,
                "throughput_tokens": token_in_update / update_time,
                "throughput_examples": args.total_batch_size * world_size / update_time,
                "throughput_batchs": batch_in_update,
            },
            step=update_step,)

        update_time = time.time()

    log("finish training")
    if rank == 0:
        pbar.close()


        

    
    
    



if __name__ == "__main__":
    ddp_setup()
    args = parse_args(None)
    main(args)







