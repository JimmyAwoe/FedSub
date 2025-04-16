from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
import torch.optim as optim
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

def ddp_setup():
    dist.init_process_group(backend="nccl")

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



def baseline(args):
    rank = int(os.environ.get("RANK"))
    local_rank = int(os.environ.get("LOCAL_RANK"))
    world_size = int(os.environ.get("WORLD_SIZE"))
    device = f"cuda:{rank}"
    # baseline
    class basemodel(nn.Module):
        def __init__(self, in_feature, out_feature):
            super().__init__()
            self.linear = nn.Linear(in_feature, out_feature)
        
        def forward(self, x):
            if len(x) == 1:
                bias = torch.ones(1).to(x.device)
                return self.linear(torch.cat((x, bias), dim=0)).squeeze(1)
            else:
                bias = torch.ones(x.shape[0], 1).to(x.device)
                return self.linear(torch.cat((x,bias), dim=1)).squeeze(1)

    model = basemodel(args.dim+1, 1).to(device)
    # adamW
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    # sgd
    #optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)

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
    if rank == 0:
        log(f"baseline start training")

    if rank == 0:
        wandb.init(project="SubScaf_simple")
        log("start loading arguments")

    run_config = vars(args)
    run_config.update({
        "world_size": world_size,
        "device": device,
    })

    for e in range(args.epoch):
        loss_eval = eval(X_test, y_test, mse_loss, model)
        if args.record:
            with torch.no_grad():
                dist.all_reduce(loss_eval, op=dist.ReduceOp.AVG)        
            if rank == 0:
                wandb.log({"eval_loss": loss_eval.item()})
        for idx, (x, y) in enumerate(dataloader):
            output = model(x)
            loss = mse_loss(output, y)
            loss.backward()
            if args.record:
                with torch.no_grad():
                    dist.all_reduce(loss, op=dist.ReduceOp.AVG)
                if rank == 0:
                    wandb.log({"loss": loss.item()})
            with torch.no_grad():
                for p in model.parameters():
                    dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)
            optimizer.step()
            optimizer.zero_grad()

    if rank == 0:
        log(f"baseline start evaulation")

    loss_eval = eval(X_test, y_test, mse_loss, model)
    dist.all_reduce(loss_eval, op=dist.ReduceOp.AVG)
    if rank == 0:
        log(f"baseline eval loss is {loss_eval}")
        wandb.log({"eval_loss": loss_eval})


    
if __name__ == "__main__":
    ddp_setup()
    args = parse_args(None)
    baseline(args)