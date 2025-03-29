from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch
from torch.utils.data import Dataset, DataLoader
import torch.distributed as dist
import argparse
import os
import wandb
import tqdm
from loguru import logger
from copy import deepcopy
from utils import *


class MyDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        label = self.labels[idx]
        return sample, label


def standarize(input):
    mean = torch.mean(input, dim=0)
    var = torch.std(input, dim=0)
    return (input - mean) / var

def log(info):
    logger.info(f"[{int(os.environ.get('RANK'))}] " + info) 


def eval(test_x, test_y, loss_fun, model):
    output = model(test_x)
    loss = loss_fun(output, test_y)
    return loss


def parse_args(args):
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--rank", type=int, default=3, help="compression rank")
    parser.add_argument("--dim", type=int, default=8, help="data dimension")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--tau", type=int, default=5, help="the number of inner loop")
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--epoch", type=int, default=1)
    parser.add_argument("--record", type=bool, default=True)

    args = parser.parse_args(args)
    return args


def SubScaf(args):

    rank = int(os.environ.get("RANK"))
    local_rank = int(os.environ.get("LOCAL_RANK"))
    world_size = int(os.environ.get("WORLD_SIZE"))
    device = f"cuda:{rank}"

    if rank == 0:
        wandb.init(project="SubScaf_simple")
        log("start loading arguments")
        log("*" * 40)
        for k, v in vars(args).items():
            log(f"{k:30} {v}")
        log("*" * 40)

    run_config = vars(args)
    run_config.update({
        "world_size": world_size,
        "device": device,
    })
    # load dataset 
    housing = fetch_california_housing()
    X = standarize(torch.tensor(housing.data, dtype=torch.float32)).to(device)
    y = torch.tensor(housing.target, dtype=torch.float32).to(device)

    # split train set and test set 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # encapsulate data into dataloader
    dataset = MyDataset(X_train, y_train)
    # dataset = split_dataset_by_node(dataset, rank, world_size)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # load loss fun
    mse_loss = nn.MSELoss()

    # +1 for bias
    init_x = torch.randn(args.dim + 1).to(device)
    lbd = torch.zeros(args.rank).to(device)
    init_comp_mat = generate_compression_mat_random(args.dim + 1, args.rank).to(device)
    dist.broadcast(init_x, src=0)
    dist.broadcast(lbd, src=0)
    dist.broadcast(init_comp_mat, src=0)
    model = SubScafLinearClassifier(args.rank, init_x, init_comp_mat).to(device)
    optimizer = SubScaffoldSGD(model.parameters(), args.lr, init_comp_mat, lbd, args.tau)
    
    if rank == 0:
        log("start training")
    for e in range(args.epoch):
        if rank == 0:
            log(f"exercute {e} epoch")

        loss_eval = eval(X_test, y_test, mse_loss, model)
        if rank == 0:
            log(f"eval loss in {e} epoch: {loss_eval}")
        if args.record:
            with torch.no_grad():
                dist.all_reduce(loss_eval, op=dist.ReduceOp.AVG)        
            if rank == 0:
                wandb.log({"eval_loss": loss_eval.item()})

        for idx, (x, y) in enumerate(dataloader):
            output = model(x)
            loss = mse_loss(output, y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            if args.record:
                with torch.no_grad():
                    dist.all_reduce(loss, op=dist.ReduceOp.AVG)
                if rank == 0:
                    wandb.log({"loss": loss.item()})
            if idx % args.tau != 0 or idx == 0:
                continue
            # below code only exercute while outer propagate
            with torch.no_grad():
                # get average of b
                avg_b = deepcopy(model.b)
                dist.all_reduce(avg_b, op=dist.ReduceOp.AVG)
                # update x
                model.weight += model.p @ avg_b
                # update lbd
                for group in optimizer.param_groups:
                    group['lbd'] += model.b - avg_b
                # update b and p
                model.refresh()

    if rank == 0:
        log("finish training")
        log("start eval")

    loss_eval = eval(X_test, y_test, mse_loss, model)
    dist.all_reduce(loss_eval, op=dist.ReduceOp.AVG)
    if rank == 0:
        log(f"eval loss is {loss_eval}")
        wandb.log({"eval_loss": loss_eval.item()})

if __name__ == "__main__":
    ddp_setup()
    args = parse_args(None)
    SubScaf(args)







