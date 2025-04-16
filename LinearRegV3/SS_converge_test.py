import numpy as np
from utils import *
import matplotlib.pyplot as plt
import argparse
import os
import torch.nn as nn
import torch.optim as optim
import torch
from copy import deepcopy
import pandas as pd

def generate_linear_data(sample_num, dim, weight_for_noise):
    A = torch.randn(sample_num, dim + 1)
    noise = torch.randn(sample_num, 1)
    # the last column is bias term
    A[:, -1] = 1
    true_weight = torch.randn(dim + 1, 1)
    y = A @ true_weight + weight_for_noise * noise
    return A, y.squeeze(), true_weight.squeeze()

def least_square_mehtod(A, y):
    return (torch.inverse(A.T @ A) @ (A.T @ y)).squeeze()

def parse_arg(args):
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample_num", default=10_000, type=int)
    parser.add_argument("--dim", help="the dimension of data", default=10, type=int)
    parser.add_argument("--rank", help="the compression rank", default=4, type=int)
    parser.add_argument("--lr", default=0.01, type=float)
    parser.add_argument("--tau", default=10, type=int)
    parser.add_argument("--iter_num", default=1000, type=int)
    parser.add_argument("--plot", default=False, type=bool)
    parser.add_argument("--noise", default=0, type=float, help="determine the magnitude of noise in data generation")

    args = parser.parse_args(args)

    return args


norm_fun = lambda x, y: torch.norm(x - y, p=2) / torch.sqrt(torch.tensor(len(x)))

def plot(data, title, label, output_name, start_point=0, end_point=None):
    plt.figure(figsize=(6,4))
    plt.title(title)
    plt.grid(True)
    plt.xlabel('step')
    plt.ylabel('loss')
    if isinstance(data, list):
        length = len(data)
    elif isinstance(data, dict):
        # randomly choose ont to measure length
        length = len(list(data.values())[0])
    # set plot range
    if end_point == None and start_point == 0:
        x = np.arange(length)
    elif end_point != None and start_point == 0:
        x = np.arange(end_point)
    elif end_point == None and start_point != 0:
        x = np.arange(start_point, length)
    elif end_point != None and start_point != 0:
        x = np.arange(start_point, end_point)

    if isinstance(data, list):
        plt.plot(x, data, linewidth=2, label=label)
        color = plt.gca().lines[-1].get_color()
        ewm = pd.Series(data).ewm(alpha=0.3).mean()
        #plt.plot(x, ewm, color=color, linewidth=1, label=label)    
        plt.axhline(y=ewm.iat[-1], color=color, linestyle='--')
        plt.yscale('log')
    elif isinstance(data, dict):
        for name, value in data.items():
            plt.plot(x, value, linewidth=2, label=name)
            color = plt.gca().lines[-1].get_color()
            ewm = pd.Series(value).ewm(alpha=0.3).mean()
            #plt.plot(x, ewm, color=color, linewidth=1, label=label)    
            plt.axhline(y=ewm.iat[-1], color=color, linestyle='--')
            plt.yscale('log')
    plt.legend()
    plt.savefig(f"./plot/{output_name}.png")
    plt.show()
    plt.close()


def baseline(model, opt, data, label, solution, loss_fun, args):
    """Test the baseline in given model and optimizer"""
    rank = int(os.environ.get("RANK"))
    loss_baseline = []
    distance_baseline = []
    for i in range(args.iter_num):
        if rank == 0 and i % 10000 == 0 and i > 0:
            print(f"start iter{i}")
            print(f"loss: {loss}")
            print(f"distance: {distance}")
        output = model(data)
        loss = loss_fun(output, label)
        loss.backward()
        with torch.no_grad():
            dist.all_reduce(loss, op=dist.ReduceOp.AVG)
            loss_baseline.append(loss.item())
            for p in model.parameters():
                dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)
        # for the reason that every node has the same weight, so don't need to carry
        # out all_reduce before computer distane.
        distance = norm_fun(list(model.parameters())[0].squeeze(), solution) / norm_fun(solution, torch.zeros_like(solution))
        distance_baseline.append(distance.item())
        opt.step()
        opt.zero_grad()
    return loss_baseline, distance_baseline

def SubScaf(model, opt, data, label, solution, loss_fun, args):
    rank = int(os.environ.get("RANK"))
    loss_rec = []
    distance_rec = []
    for i in range(args.iter_num):
        if rank == 0 and i % 10000 == 0 and i > 0:
            print(f"start iter{i}")
            print(f"loss: {loss}")
            print(f"distance: {distance}")
        output = model(data)
        loss = loss_fun(output, label)
        loss.backward()
        dist.all_reduce(loss, op=dist.ReduceOp.AVG)
        loss_rec.append(torch.sqrt(loss).item())
        opt.step()
        opt.zero_grad()
        # record the distance between solution and weight 
        with torch.no_grad():
            avg_b = deepcopy(model.b)
            dist.all_reduce(avg_b, op=dist.ReduceOp.AVG)
            # update x
            x = model.weight + model.p @ avg_b
            distance = norm_fun(x, solution) / norm_fun(solution, torch.zeros_like(solution))
            dist.all_reduce(distance, op=dist.ReduceOp.AVG)
            distance_rec.append(distance.item())
        if i % args.tau != 0 or i == 0:
            continue
        with torch.no_grad():
            # get average of b
            avg_b = deepcopy(model.b)
            dist.all_reduce(avg_b, op=dist.ReduceOp.AVG)
            # update x
            model.weight += model.p @ avg_b
            # update lbd
            for group in opt.param_groups:
                group['lbd'] += model.b - avg_b
            # update b and p
            model.refresh(model.weight.shape[0], model.rank)
    return loss_rec, distance_rec


def SubScafReOptState(model, opt, data, label, solution, loss_fun, args):
    rank = int(os.environ.get("RANK"))
    loss_rec = []
    distance_rec = []
    for i in range(args.iter_num):
        if rank == 0 and i % 10000 == 0 and i > 0:
            print(f"start iter{i}")
            print(f"loss: {loss}")
            print(f"distance: {distance}")
        output = model(data)
        loss = loss_fun(output, label)
        loss.backward()
        dist.all_reduce(loss, op=dist.ReduceOp.AVG)
        loss_rec.append(torch.sqrt(loss).item())
        opt.step()
        opt.zero_grad()
        # record the distance between solution and weight 
        with torch.no_grad():
            avg_b = deepcopy(model.b)
            dist.all_reduce(avg_b, op=dist.ReduceOp.AVG)
            # update x
            x = model.weight + model.p @ avg_b
            distance = norm_fun(x, solution) / norm_fun(solution, torch.zeros_like(solution))
            dist.all_reduce(distance, op=dist.ReduceOp.AVG)
            distance_rec.append(distance.item())
        if i % args.tau != 0 or i == 0:
            continue
        with torch.no_grad():
            # get average of b
            avg_b = deepcopy(model.b)
            dist.all_reduce(avg_b, op=dist.ReduceOp.AVG)
            # update x
            model.weight += model.p @ avg_b
            # update lbd
            for group in opt.param_groups:
                group['lbd'] += model.b - avg_b
                # restart optimizer state
                for p in group['p']:
                    opt.state[p] = {}
            # update b and p
            model.refresh(model.weight.shape[0], model.rank)
    return loss_rec, distance_rec

def SubScafGaLore(model, opt, data, label, solution, loss_fun, args):
    rank = int(os.environ.get("RANK"))
    loss_rec = []
    distance_rec = []
    for i in range(args.iter_num):
        if rank == 0 and i % 10000 == 0 and i > 0:
            print(f"start iter{i}")
            print(f"loss: {loss}")
            print(f"distance: {distance}")
        output = model(data)
        loss = loss_fun(output, label)
        loss.backward()
        dist.all_reduce(loss, op=dist.ReduceOp.AVG)
        loss_rec.append(torch.sqrt(loss).item())
        opt.step()
        opt.zero_grad()
        # record the distance between solution and weight 
        with torch.no_grad():
            avg_b = deepcopy(model.b)
            dist.all_reduce(avg_b, op=dist.ReduceOp.AVG)
            # update x
            x = model.aggregate(model.weight, model.p, avg_b)
            distance = norm_fun(x, solution) / norm_fun(solution, torch.zeros_like(solution))
            dist.all_reduce(distance, op=dist.ReduceOp.AVG)
            distance_rec.append(distance.item())
        if i % args.tau != 0 or i == 0:
            continue
        with torch.no_grad():
            # get average of b
            avg_b = deepcopy(model.b)
            dist.all_reduce(avg_b, op=dist.ReduceOp.AVG)
            # update weight
            model.weight = model.aggregate(model.weight, model.p, model.b)
            # update lbd
            for group in opt.param_groups:
                group['lbd'] += model.b - avg_b
            # update b and p
            model.refresh(data, label, loss_fun)
            # clear the difference made by refresh
            opt.zero_grad()
    return loss_rec, distance_rec

def LowRankFedAvg(model, opt, data, label, solution, loss_fun, args):
    rank = int(os.environ.get("RANK"))
    loss_rec = []
    distance_rec = []
    for i in range(args.iter_num):
        if rank == 0 and i % 10000 == 0 and i > 0:
            print(f"start iter{i}")
            print(f"loss: {loss}")
            print(f"distance: {distance}")
        output = model(data)
        loss = loss_fun(output, label)
        loss.backward()
        dist.all_reduce(loss, op=dist.ReduceOp.AVG)
        loss_rec.append(torch.sqrt(loss).item())
        opt.step()
        opt.zero_grad()
        # record the distance between solution and weight 
        with torch.no_grad():
            avg_b = deepcopy(model.b)
            dist.all_reduce(avg_b, op=dist.ReduceOp.AVG)
            # update x
            x = model.weight + model.p @ avg_b
            distance = norm_fun(x, solution) / norm_fun(solution, torch.zeros_like(solution))
            dist.all_reduce(distance, op=dist.ReduceOp.AVG)
            distance_rec.append(distance.item())
        if i % args.tau != 0 or i == 0:
            continue
        with torch.no_grad():
            # get average of b
            avg_b = deepcopy(model.b)
            dist.all_reduce(avg_b, op=dist.ReduceOp.AVG)
            # update x
            model.weight += model.p @ avg_b
            # update b and p
            model.refresh(model.weight.shape[0], model.rank)
    return loss_rec, distance_rec


def main(args):
    rank = int(os.environ.get("RANK"))
    local_rank = int(os.environ.get("LOCAL_RANK"))
    world_size = int(os.environ.get("WORLD_SIZE"))
    device = f"cuda:{rank}"

    # set loss fun
    mse_loss = nn.MSELoss()
    # generate data and send to GPU
    sample_x, sample_y, solution = generate_linear_data(args.sample_num, args.dim, args.noise)
    solution1 = least_square_mehtod(sample_x, sample_y)
    if rank == 0:
        print("finish generate data")
        print(f"the solution has loss {mse_loss(sample_x @ solution, sample_y)}")
    sample_x = sample_x.to(device)
    sample_y = sample_y.to(device)
    solution = solution.to(device)
    dist.broadcast(sample_x, src=0)
    dist.broadcast(sample_y, src=0)
    dist.broadcast(solution, src=0)

    def SubScaf_SGD():
        # SubScaf sgd
        rank = int(os.environ.get("RANK"))
        if rank == 0:
            print("start SubScaf SGD")
        init_x = torch.randn(args.dim + 1).to(device)
        lbd = torch.zeros(args.rank).to(device)
        init_comp_mat = generate_compression_mat_random(args.dim + 1, args.rank).to(device)
        dist.broadcast(init_x, src=0)
        dist.broadcast(lbd, src=0)
        dist.broadcast(init_comp_mat, src=0)
        model = SubScafLinearClassifier(args.rank, init_x, init_comp_mat).to(device)
        optimizer = SubScaffoldSGD(model.parameters(), args.lr, init_comp_mat, lbd, args.tau)

        loss_rec, distance_rec = SubScaf(model, 
                                        optimizer, 
                                        sample_x[:, :-1], 
                                        sample_y, 
                                        solution, 
                                        mse_loss, 
                                        args)
        return loss_rec, distance_rec

    def SubScaf_SGD_disunity_init():
        # SubScaf sgd
        rank = int(os.environ.get("RANK"))
        if rank == 0:
            print("start SubScaf SGD")
        init_x = torch.randn(args.dim + 1).to(device)
        lbd = torch.zeros(args.rank).to(device)
        init_comp_mat = generate_compression_mat_random(args.dim + 1, args.rank).to(device)
        dist.broadcast(init_comp_mat, src=0)
        model = SubScafLinearClassifier(args.rank, init_x, init_comp_mat).to(device)
        optimizer = SubScaffoldSGD(model.parameters(), args.lr, init_comp_mat, lbd, args.tau)

        loss_rec, distance_rec = SubScaf(model, 
                                        optimizer, 
                                        sample_x[:, :-1], 
                                        sample_y, 
                                        solution, 
                                        mse_loss, 
                                        args)
        return loss_rec, distance_rec

    def SubScaf_SGD_AvgCPMat():
        # SubScaf sgd
        rank = int(os.environ.get("RANK"))
        if rank == 0:
            print("start SubScaf SGD")
        init_x = torch.randn(args.dim + 1).to(device)
        lbd = torch.zeros(args.rank).to(device)
        init_comp_mat = generate_compression_mat_random(args.dim + 1, args.rank).to(device)
        dist.broadcast(init_x, src=0)
        dist.all_reduce(init_comp_mat, op=dist.ReduceOp.AVG)
        model = SubScafAvgCPMatLinearClassifier(args.rank, init_x, init_comp_mat).to(device)
        optimizer = SubScaffoldSGD(model.parameters(), args.lr, init_comp_mat, lbd, args.tau)

        loss_rec, distance_rec = SubScaf(model, 
                                        optimizer, 
                                        sample_x[:, :-1], 
                                        sample_y, 
                                        solution, 
                                        mse_loss, 
                                        args)
        return loss_rec, distance_rec

    def SubScaf_Adam():
        # SubScaf adam 
        if rank == 0:
            print("start SubScaf Adam")
        init_x = torch.randn(args.dim + 1).to(device)
        lbd = torch.zeros(args.rank).to(device)
        init_comp_mat = generate_compression_mat_random(args.dim + 1, args.rank).to(device)
        dist.broadcast(init_x, src=0)
        dist.broadcast(lbd, src=0)
        dist.broadcast(init_comp_mat, src=0)
        model = SubScafLinearClassifier(args.rank, init_x, init_comp_mat).to(device)
        optimizer = SubScaffoldAdam(model.parameters(), args.lr, init_comp_mat, lbd, args.tau)

        loss_rec, distance_rec = SubScaf(model, 
                                        optimizer, 
                                        sample_x[:, :-1], 
                                        sample_y, 
                                        solution, 
                                        mse_loss, 
                                        args)
        return loss_rec, distance_rec

    def SubScaf_Adam_ReOpt():
        # SubScaf adam restart optimizer state
        
        if rank == 0:
            print("start SubScaf Adam restart opt state")
        init_x = torch.randn(args.dim + 1).to(device)
        lbd = torch.zeros(args.rank).to(device)
        init_comp_mat = generate_compression_mat_random(args.dim + 1, args.rank).to(device)
        dist.broadcast(init_x, src=0)
        dist.broadcast(lbd, src=0)
        dist.broadcast(init_comp_mat, src=0)
        model = SubScafLinearClassifier(args.rank, init_x, init_comp_mat).to(device)
        optimizer = SubScaffoldAdam(model.parameters(), args.lr, init_comp_mat, lbd, args.tau)

        loss_rec, distance_rec = SubScafReOptState(
                                        model, 
                                        optimizer, 
                                        sample_x[:, :-1], 
                                        sample_y, 
                                        solution, 
                                        mse_loss, 
                                        args)
        return loss_rec, distance_rec

    def Vanilla_Adam():
        # adam
        if rank == 0:
            print("start baseline adam")
        basemodel_adam = basemodel(args.dim+1, 1).to(device)
        # synchronize initial parameter
        with torch.no_grad():
            for p in basemodel_adam.parameters():
                dist.broadcast(p, src=0)
        baseopt_adam = optim.Adam(basemodel_adam.parameters(), lr=args.lr)
        loss_rec, distance_rec= baseline(basemodel_adam,
                                                    baseopt_adam, 
                                                    sample_x[:, :-1], 
                                                    sample_y, 
                                                    solution, 
                                                    mse_loss, 
                                                    args)
        return loss_rec, distance_rec

    def Vanilla_SGD():
        # sgd
        if rank == 0:
            print("start baseline sgd")
        basemodel_sgd = basemodel(args.dim+1, 1).to(device)
        # synchronize initial parameter
        with torch.no_grad():
            for p in basemodel_sgd.parameters():
                dist.broadcast(p, src=0)
        baseopt_sgd = optim.SGD(basemodel_sgd.parameters(), lr=args.lr)
        loss_rec, distance_rec = baseline(basemodel_sgd,
                                                    baseopt_sgd, 
                                                    sample_x[:, :-1], 
                                                    sample_y, 
                                                    solution, 
                                                    mse_loss, 
                                                    args)
        return loss_rec, distance_rec
                        
    
    def SubScafGaLore():
        # scaffold+Galore
        if rank == 0:
            print("start scaffold+GaLore")
        init_x = torch.randn(args.dim + 1).to(device)
        lbd = torch.zeros(args.rank).to(device)
        init_comp_mat = generate_compression_mat_random(args.dim + 1, args.rank).to(device)
        dist.broadcast(init_x, src=0)
        dist.broadcast(lbd, src=0)
        dist.broadcast(init_comp_mat, src=0)
        model = SubScafGaloreLinearClassifier(args.rank, init_x, init_comp_mat, generate_compression_mat_svd).to(device)
        optimizer = SubScaffoldSGD(model.parameters(), args.lr, init_comp_mat, lbd, args.tau)

        loss_rec, distance_rec= SubScafGaLore(
                                        model, 
                                        optimizer, 
                                        sample_x[:, :-1], 
                                        sample_y, 
                                        solution, 
                                        mse_loss, 
                                        args)
        return loss_rec, distance_rec

    def LowRank_FedAvg():
        # fedavg+random low-rank projection
        if rank == 0:
            print("start low rank fedavg")
        init_x = torch.randn(args.dim + 1).to(device)
        init_comp_mat = generate_compression_mat_random(args.dim + 1, args.rank).to(device)
        dist.broadcast(init_x, src=0)
        dist.broadcast(init_comp_mat, src=0)
        model = SubScafLinearClassifier(args.rank, init_x, init_comp_mat).to(device)
        optimizer = optim.SGD(model.parameters(), lr=args.lr)

        loss_rec, distance_rec= LowRankFedAvg(model, 
                                        optimizer, 
                                        sample_x[:, :-1], 
                                        sample_y, 
                                        solution, 
                                        mse_loss, 
                                        args)
        return loss_rec, distance_rec
    
    # below is the base comparation setting
    loss_rec_sgd, distance_rec_sgd = SubScaf_SGD()
    loss_rec_adam, distance_rec_adam = SubScaf_Adam()
    baseloss_adam, basedistance_adam = Vanilla_Adam()
    baseloss_sgd, basedistance_sgd = Vanilla_SGD()
    loss_rec_fedavg, distance_rec_fedavg = LowRank_FedAvg()


    # compare plot
    loss_dict = {'SubScaf_sgd': loss_rec_sgd, 
                'SubScaf_adam': loss_rec_adam, 
                'AdamW': baseloss_adam, 
                'SGD': baseloss_sgd,
                'FedAvg': loss_rec_fedavg}

    distance_dict = {'SubScaf_sgd': distance_rec_sgd, 
                    'SubScaf_adam': distance_rec_adam, 
                    'AdamW': basedistance_adam, 
                    'SGD': basedistance_sgd,
                    'FedAvg': distance_rec_fedavg}
    
    if rank == 0 and args.plot:
        
        print("start plot")
        plot(loss_dict, 'compare_loss', None, 'compare_loss')
        plot(distance_dict, 'compare_distance', None, 'compare_distance')

    if False:
        # below we want to compare different compression rank
        args.rank = 5
        loss_5, distance_5 = SubScaf_SGD()
        args.rank = 10
        loss_10, distance_10 = SubScaf_SGD()
        args.rank = 15
        loss_15, distance_15 = SubScaf_SGD()
        args.rank = 20 
        loss_20, distance_20 = SubScaf_SGD()
        args.rank = 30 
        loss_30, distance_30 = SubScaf_SGD()
        args.rank = 40 
        loss_40, distance_40 = SubScaf_SGD()
        args.rank = 10


        # compare plot
        loss_dict = {
                    'dim=5': loss_5, 
                    'dim=10': loss_10, 
                    'dim=15': loss_15, 
                    'dim=20': loss_20, 
                    'dim=30': loss_30, 
                    'dim=40': loss_40, 
                    }

        distance_dict = {
                    'dim=5': distance_5, 
                    'dim=10': distance_10, 
                    'dim=15': distance_15, 
                    'dim=20': distance_20, 
                    'dim=30': distance_30, 
                    'dim=40': distance_40, 
        } 


        if rank == 0 and args.plot:
        
            print("start plot")
            plot(loss_dict, 'compare_loss_with_diff_dim', None, 'compare_loss_with_diff_dim')
            plot(distance_dict, 'compare_distance_with_diff_dim', None, 'compare_distance_with_diff_dim')

        # compare the opt state restart difference for SubScaf adam
        loss, distance = SubScaf_Adam()
        loss_restart, distance_restart = SubScaf_Adam_ReOpt()

        loss_dict = {
            "Don't Restart Opt State": loss,
            "Restart Opt State": loss_restart,
        }
        distance_dict = {
            "Don't Restart Opt State": distance,
            "Restart Opt State": distance_restart,
        }

        if rank == 0 and args.plot:
        
            print("start plot")
            plot(loss_dict, 'Loss_SubScaf_Adam_ReOpt', None, 'Loss_SubScaf_Adam_ReOpt')
            plot(distance_dict, 'Distance_SubScaf_Adam_ReOpt', None, 'Distance_SubScaf_Adam_ReOpt')

        # compare subscaf sgd with unity or disunity initial param
        loss_unity, distance_unity = SubScaf_SGD()
        loss_disunity, distance_disunity = SubScaf_SGD_disunity_init()
        loss_dict = {
            'Unit_Init_Param': loss_unity,
            'DisUnit_init_Param': loss_disunity,
        }
        distance_dict = {
            'Unit_Init_Param': distance_unity,
            'DisUnit_init_Param': distance_disunity,
        }
        if rank == 0 and args.plot:
        
            print("start plot")
            plot(loss_dict, 'Loss_SubScaf_SGD_Init', None, 'Loss_SubScaf_SGD_Init')
            plot(distance_dict, 'Distance_SubScaf_SGD_Init', None, 'Distance_SubScaf_SGD_Init')
    

        # below we want to compare different inner loop num i.e. tau 
        args.tau = 5
        loss_5, distance_5 = SubScaf_SGD()
        args.tau = 10
        loss_10, distance_10 = SubScaf_SGD()
        args.tau = 50
        loss_50, distance_50 = SubScaf_SGD()
        args.tau = 100
        loss_100 , distance_100 = SubScaf_SGD()
        args.tau = 500
        loss_500, distance_500 = SubScaf_SGD()
        args.tau = 1000
        loss_1000, distance_1000 = SubScaf_SGD()
        args.tau = 10


        # compare plot
        loss_dict = {
                    'tau=5': loss_5, 
                    'tau=10': loss_10, 
                    'tau=50': loss_50, 
                    'tau=100': loss_100, 
                    'tau=500': loss_500, 
                    'tau=1000': loss_1000, 
                    }

        distance_dict = {
                    'tau=5': distance_5, 
                    'tau=10': distance_10, 
                    'tau=50': distance_50, 
                    'tau=100': distance_100, 
                    'tau=500': distance_500, 
                    'tau=1000': distance_1000, 
        } 


        if rank == 0 and args.plot:
        
            print("start plot")
            plot(loss_dict, 'compare_loss_with_diff_inner_num', None, 'compare_loss_with_diff_inner_num')
            plot(distance_dict, 'compare_distance_with_diff_inner_num', None, 'compare_distance_with_diff_inner_num')

        # if average randomly generated compression matrix between workers
        loss_broadcast, distance_broadcast = SubScaf_SGD()
        loss_all_reduce, distance_all_reduce = SubScaf_SGD_AvgCPMat()
        loss_dict = {
            'Use Worker 0': loss_broadcast,
            'Avg Between Worker': loss_all_reduce,
        }
        distance_dict = {
            'Use Worker 0': distance_broadcast,
            'Avg Between Worker': distance_all_reduce,
        }
        if rank == 0 and args.plot:
            print("start plot")
            plot(loss_dict, 'Loss_SubScaf_SGD_AvgCPMat', None, 'Loss_SubScaf_SGD_AvgCPMat')
            plot(distance_dict, 'Distance_SubScaf_SGD_AvgCPMat', None, 'Distance_SubScaf_SGD_AvgCPMat')




if __name__ == "__main__":
    ddp_setup()
    args = parse_arg(None)
    
    main(args)



