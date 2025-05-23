import numpy as np
import matplotlib.pyplot as plt
import math
import pandas as pd

class cosine_alpha():
    def __init__(self, max_lr, max_step):
        self.max_lr = max_lr
        self.max_step = max_step
    
    def __call__(self, step):
        scale_step = (self.max_step - step) / self.max_step
        lr = self.max_lr * math.cos(math.pi / 2 * (1 - scale_step))
        return lr

# data generation utility
def genLS(total_sample_size, d):
    a = np.random.randn(total_sample_size, d)
    ans = np.random.randn(d, 1)
    b = a @ ans + 0.1 * np.random.randn(total_sample_size, 1)
    return a, b

def genHeLS(total_sample_size, d, worker_num):
    assert total_sample_size % worker_num == 0, "please make sure total_sample can be divided for each worker equally"
    #mean = np.random.uniform(low=-1, high=1, size=worker_num)
    mean = np.zeros(shape=worker_num)
    scale = np.sqrt(np.random.uniform(low=0.5, high=1.5, size=worker_num))
    a_list = []
    for i in range(worker_num):
        a_list.append(np.random.normal(loc=mean[i], scale=scale[i], size=(total_sample_size // worker_num, d)))
    a = np.concatenate(a_list, axis=0)
    ans = np.random.randn(d, 1)
    b = a @ ans + 0.1 * np.random.randn(total_sample_size, 1)
    return a, b


# Solution utility
def solLS(A, b, args):
    x_sol = np.linalg.inv(A.T@A)@(A.T@b)
    if args.print_sol:
        print(A.T@(A@x_sol-b) / len(A))
    return x_sol

# Gradient utility
def ls_full_grad_dist(X, y, W):
    n, m = X.shape
    Q = W.shape[0]
    N_agent = n//Q    
    G = np.zeros((Q, m))
    for k in range(Q):
        wk = W[k,:].reshape(m, 1)        
        Xk = X[k*N_agent:(k+1)*N_agent, :]
        yk = y[k*N_agent:(k+1)*N_agent].reshape(N_agent, 1)
        grad = Xk.T@(Xk@wk-yk)
        G[k,:] = grad.T
    return G

def gene_random_matrix(in_dim, out_dim):
    return np.random.randn(in_dim, out_dim) / np.sqrt(out_dim)

def gene_ide_matrix(in_dim, out_dim):
    assert in_dim == out_dim, "two dimensions must be equal"
    return np.eye(in_dim)


def Coordinate_descend_genep(d, r, n=1):
    sum_p = np.zeros((d, r)) 
    for _ in range(n):
        ide = np.eye(d)
        col_num = np.arange(d)
        select_col = np.random.choice(col_num, r, replace=False)
        sign = np.random.choice([-1, 1], r)
        P = np.sqrt(d / r) * ide[:, select_col] * sign
        #P = ide[:, select_col] * sign
        sum_p += P
    P = sum_p / n
    return P


def Spherical_smoothing_genep(d, r, n=1):
    sum_p = np.zeros((d, r)) 
    for _ in range(n):
        z = np.random.randn(d, d)
        Q, R = np.linalg.qr(z)
        D = np.diag(np.sign(np.diag(R)))
        Q = Q @ D
        R = D @ R
        assert np.allclose(Q @ R, z, atol=1e-7), "the QR decomposion is not accuracy"
        P = np.sqrt(d / r) * Q[:, :r]
        #P = Q[:, :r]
        sum_p += P
    P = sum_p / n
    return P

# Gradient utility with compression
def ls_full_grad_dist_compression(X, y, W, P):
    n, m = X.shape
    Q = W.shape[0]
    N_agent = n//Q    
    G = np.zeros((Q, P.shape[1]))
    for k in range(Q):
        wk = W[k,:].reshape(m, 1)        
        Xk = X[k*N_agent:(k+1)*N_agent, :]
        yk = y[k*N_agent:(k+1)*N_agent].reshape(N_agent, 1)
        grad = P.T @ Xk.T@(Xk@wk-yk) 
        G[k,:] = grad.T.squeeze()
    return G

def multiply_with_row(a, b):
    N = b.shape[0]
    result = np.zeros((N, a.shape[0]))
    for i in range(N):
        row = b[i, :]
        product = np.dot(a, row)
        result[i, :] = product
    return result

def Scaf(X, y, N, A, alpha, alpha_bar, noise, inner_iter_num=10, maxite=500, epochs=10):
    """
    X: 输入数据
    y: label
    N: worker数量
    alpha: 学习率
    alpha_bar: 外部学习率
    noise: 梯度的噪声
    inner_iter_num: 内部迭代次数
    maxite: 外部迭代次数
    epochs: 进行多少次重复来平均msd
    """
    R = np.ones((N,N))/N
    _, M = X.shape
    msd_mean = np.zeros((maxite * inner_iter_num, 1))
    for e in range(epochs):
        if (e+1)%1 == 0:
            print('Centralized SGD epoch index:', e + 1)
        W = np.zeros((N, M))
        #W = np.random.randn(N, M)
        W_bar = np.zeros((N, M))
        msd = np.zeros((maxite * inner_iter_num, 1))
        c = np.zeros((N, M))
        for ite in range(maxite):
            W = W_bar.copy()
            G_inner = np.zeros((N, M))
            for in_ite in range(inner_iter_num):
                G = ls_full_grad_dist(X, y, W) + noise * np.random.randn(N, M)
                W -= alpha * (G / N + c)
                G_inner += G 
                msd[ite * inner_iter_num + in_ite] = np.sum((W-Ws)**2)/N / (np.sum((Ws)**2)/N)
            update = (R.dot(W) - W_bar)
            W_bar += alpha_bar * update
            c = 1 / (alpha * inner_iter_num) * (- update) - 1 / inner_iter_num * G_inner / N 
        msd_mean += msd
    msd_mean = msd_mean/epochs
    return msd_mean


def parser_args(args):
    import argparse
    parser = argparse.ArgumentParser(description="Linear Regression Experiment for SubScaf and Scaf")
    parser.add_argument("--lr", default=0.01, type=float)
    parser.add_argument("--inner_iter_num", default=10, type=int)
    parser.add_argument("--out_iter_num", default=1000, type=int)
    parser.add_argument("--grad_noise", default=0, type=float)
    parser.add_argument("--worker_num", default=20, type=int)
    parser.add_argument("--cp_gene_method", default='rd', type=str)
    parser.add_argument("--epochs", default=5, type=int)
    parser.add_argument("--dim", default=10, type=int)
    parser.add_argument("--total_sample", default=10000, type=int)
    parser.add_argument("--print_sol", action="store_true")
    parser.add_argument("--grad_clip", default=None, type=float)
    parser.add_argument("--lbd_clip", default=None, type=float)

    # data
    parser.add_argument("--hete", action="store_true")

    # subscaf
    parser.add_argument("--cp_rank", default=5, type=int)
    parser.add_argument("--grad_down_lr", action="store_true")
    parser.add_argument("--grad_divide", default=None, type=float)
    parser.add_argument("--dual_re_proj", action="store_true")

    args = parser.parse_args(args)
    return args

def gene_data(args):
    np.random.seed(2022)
    if args.hete:
        data, label = genHeLS(args.total_sample, args.dim, args.worker_num)
    else:
        data, label = genLS(args.total_sample, args.dim)
    w_sol = solLS(data, label, args)
    Ws = np.ones((args.worker_num, 1))@w_sol.T
    return Ws, data, label
    
#def _clip(value, bound):
    
    
def subscaf(data, label, worker_num, lr, grad_noise, cp_gene_method, cp_rank, inner_iter_num, out_iter_num, epochs, args):
    comm_matrix = np.ones((worker_num, worker_num)) / worker_num
    _, dim = data.shape
    total_iter_num = out_iter_num * inner_iter_num
    msd_mean = np.zeros((total_iter_num, 1))
    if args.grad_down_lr:
        lr_record = lr 
    for e in range(epochs):
        print('Centralized SGD epoch index:', e + 1)
        if args.grad_down_lr:
            lr = lr_record
            sign = 1
        Weight = np.zeros((worker_num, dim))
        msd = np.zeros((total_iter_num, 1))
        lbd = np.zeros((worker_num, cp_rank))
        for ite in range(out_iter_num):
            if args.grad_down_lr and ite > out_iter_num / 10  * sign:
                lr /= 10
                sign += 1
            b = np.zeros((worker_num, cp_rank))
            P = cp_gene_method(dim, cp_rank)
            if args.dual_re_proj and ite > 0:
                lbd = multiply_with_row(P.T, lbd)
            if args.lbd_clip:
                lbd = np.clip(lbd, a_min=-args.lbd_clip, a_max=args.lbd_clip)
            for in_ite in range(inner_iter_num):
                #grad = ls_full_grad_dist_compression(data, label, Weight + multiply_with_row(P, b), P) + \
                    #grad_noise * np.random.randn(worker_num, cp_rank)
                grad = ls_full_grad_dist(data, label, Weight + multiply_with_row(P, b)) + \
                    grad_noise * np.random.randn(worker_num, args.dim)
                grad = multiply_with_row(P.T, grad)
                if args.grad_clip:
                    grad = np.clip(grad, a_min=-args.grad_clip, a_max=args.grad_clip)
                if args.grad_divide:
                    grad /= args.grad_divide 
                b -= lr * (grad + lbd / (lr * inner_iter_num))
                msd[ite * inner_iter_num + in_ite] = np.sum(((Weight + multiply_with_row(P, b))-Ws)**2) / (np.sum((Ws)**2))
            Weight += multiply_with_row(P, comm_matrix.dot(b))
            lbd += b - comm_matrix.dot(b)
            if args.dual_re_proj:
                lbd = multiply_with_row(P, lbd)
        msd_mean += msd
    msd_mean = msd_mean / epochs
    return msd_mean.squeeze()


def subopt(data, label, worker_num, lr, grad_noise, cp_gene_method, cp_rank, inner_iter_num, out_iter_num, epochs, args):
    comm_matrix = np.ones((worker_num, worker_num)) / worker_num
    _, dim = data.shape
    total_iter_num = out_iter_num * inner_iter_num
    msd_mean = np.zeros((total_iter_num, 1))
    if args.grad_down_lr:
        lr_record = lr 
    for e in range(epochs):
        print('Centralized SGD epoch index:', e + 1)
        if args.grad_down_lr:
            lr = lr_record
            sign = 1
        Weight = np.zeros((worker_num, dim))
        msd = np.zeros((total_iter_num, 1))
        for ite in range(out_iter_num):
            if args.grad_down_lr and ite > out_iter_num / 10  * sign:
                lr /= 10
                sign += 1
            b = np.zeros((worker_num, cp_rank))
            P = cp_gene_method(dim, cp_rank)
            for in_ite in range(inner_iter_num):
                grad = ls_full_grad_dist(data, label, Weight + multiply_with_row(P, b)) + \
                    grad_noise * np.random.randn(worker_num, args.dim)
                grad = multiply_with_row(P.T, grad)
                if args.grad_divide:
                    grad /= args.grad_divide 
                b -= lr * grad
                msd[ite * inner_iter_num + in_ite] = np.sum(((Weight + multiply_with_row(P, b))-Ws)**2) / (np.sum((Ws)**2))
            Weight += multiply_with_row(P, comm_matrix.dot(b))
        msd_mean += msd
    msd_mean = msd_mean / epochs
    return msd_mean.squeeze()

def fedavg(data, label, worker_num, lr, grad_noise, inner_iter_num, out_iter_num, epochs, args):
    comm_matrix = np.ones((worker_num, worker_num)) / worker_num
    _, dim = data.shape
    total_iter_num = out_iter_num * inner_iter_num
    msd_mean = np.zeros((total_iter_num, 1))
    if args.grad_down_lr:
        lr_record = lr 
    for e in range(epochs):
        print('Centralized SGD epoch index:', e + 1)
        if args.grad_down_lr:
            lr = lr_record
            sign = 1
        Weight = np.zeros((worker_num, dim))
        msd = np.zeros((total_iter_num, 1))
        for ite in range(out_iter_num):
            if args.grad_down_lr and ite > out_iter_num / 10  * sign:
                lr /= 10
                sign += 1
            for in_ite in range(inner_iter_num):
                grad = ls_full_grad_dist(data, label, Weight) + grad_noise * np.random.randn(worker_num, args.dim)
                Weight -= lr * grad
                msd[ite * inner_iter_num + in_ite] = np.sum((Weight -Ws)**2) / (np.sum((Ws)**2))
            Weight = comm_matrix.dot(Weight)
        msd_mean += msd
    msd_mean = msd_mean / epochs
    return msd_mean.squeeze()

def plot(data, title, output_name):
    plt.figure(figsize=(6,4))
    plt.title(title)
    plt.grid(False)
    plt.xlabel('Communication Rounds')
    plt.ylabel('Error')
    plt.yticks(fontsize=12, rotation=45)
    for name, value in data.items():
        x = np.arange(len(value))
        plt.plot(x, value, linewidth=2, alpha=0.3)
        color = plt.gca().lines[-1].get_color()
        ewm = pd.Series(value).ewm(alpha=0.1).mean()
        plt.plot(x, ewm, color=color, linewidth=1, label=name)    
        plt.axhline(y=ewm.iat[-1], color=color, linestyle='--')
        plt.yscale('log')
    plt.legend()
    plt.savefig(f"LinearRegExp/{output_name}.png")
    plt.close()

def get_outer_round_value(inner_iter_num, *value):
    res = [] 
    for v in value:
        res.append(v[inner_iter_num-1::inner_iter_num])
    return res



if __name__ == "__main__":
    args = parser_args(None)

    Ws, data, label = gene_data(args)
    cp_method = {
                'id': gene_ide_matrix,
                 'rd': gene_random_matrix,
                 'cd': Coordinate_descend_genep,
                 'ss': Spherical_smoothing_genep
                 }

    gene_method = cp_method[args.cp_gene_method]

    #full = Scaf(data, label, args.worker_num, None, 0.01, 0.01, 0, 10, 4000, 1).squeeze()

    #sub_opt = subopt(data, label, args.worker_num, args.lr, args.grad_noise, 
                     #Coordinate_descend_genep, args.cp_rank, args.inner_iter_num, args.out_iter_num,
                     #args.epochs, args)
    #fed_avg = fedavg(data, label, args.worker_num, args.lr, args.grad_noise, args.inner_iter_num,
                     #args.out_iter_num, args.epochs, args)
    #args.dual_re_proj = True
    #cd = subscaf(data, label, args.worker_num, args.lr, args.grad_noise,
                 #Coordinate_descend_genep, args.cp_rank, args.inner_iter_num, args.out_iter_num, 
                 #args.epochs, args)
    #args.dual_re_proj = False

    #dis_dict = {
                #'subopt': sub_opt,
                #'fedavg': fed_avg,
                #'SubScaf with CD': cd,
                #}

    #plot(dis_dict, 'Different Generation Method', 'DiffGene')

    # experiment 1
    #rd = subscaf(data, label, args.worker_num, args.lr, args.grad_noise,
                 #gene_random_matrix, args.cp_rank, args.inner_iter_num, args.out_iter_num, 
                 #args.epochs, args)
    #args.dual_re_proj = True
    #cd = subscaf(data, label, args.worker_num, args.lr, args.grad_noise,
                 #Coordinate_descend_genep, args.cp_rank, args.inner_iter_num, args.out_iter_num, 
                 #args.epochs, args)
    #args.dual_re_proj = False
    #ss = subscaf(data, label, args.worker_num, args.lr, args.grad_noise,
                 #Spherical_smoothing_genep, args.cp_rank, args.inner_iter_num, args.out_iter_num, 
                 #args.epochs, args)
    #args.grad_down_lr = False
    #ide = subscaf(data, label, args.worker_num, args.lr, args.grad_noise,
                 #gene_ide_matrix, args.dim, args.inner_iter_num, args.out_iter_num, 
                 #args.epochs, args)
    #args.grad_down_lr = True 
    #rd, cd, ss, ide = get_outer_round_value(args.inner_iter_num, rd, cd, ss, ide)
    #exp1 = {
            #'our-CD': cd,
            #'our-RD': rd,
            #'our-SS': ss,
            #r'our-$P^k$=I': ide,
            #}

    #plot(exp1, '', 'exp1')

    # experiment2
    #args.dual_re_proj = True
    #cd = subscaf(data, label, args.worker_num, args.lr, args.grad_noise,
                 #Coordinate_descend_genep, args.cp_rank, args.inner_iter_num, args.out_iter_num, 
                 #args.epochs, args)
    #args.dual_re_proj = False
    #sub_opt = subopt(data, label, args.worker_num, args.lr, args.grad_noise, 
                     #Coordinate_descend_genep, args.cp_rank, args.inner_iter_num, args.out_iter_num,
                     #args.epochs, args)
    #fed_avg = fedavg(data, label, args.worker_num, args.lr, args.grad_noise, args.inner_iter_num,
                     #args.out_iter_num, args.epochs, args)
    #args.grad_down_lr = False
    #ide = subscaf(data, label, args.worker_num, args.lr, args.grad_noise,
                 #gene_ide_matrix, args.dim, args.inner_iter_num, args.out_iter_num, 
                 #args.epochs, args)
    #args.grad_down_lr = True
    #cd, ide, sub_opt, fed_avg = get_outer_round_value(args.inner_iter_num, cd, ide, sub_opt, fed_avg)
    #exp2 = {
        #"our-CD": cd,
        #r'our-$P^k$=I': ide,
        #"FedAvg-CD": sub_opt,
        #"FedAvg": fed_avg,
    #}

    #plot(exp2, '', 'exp2')

    # experiment3

    #args.dual_re_proj = True
    #cd1 = subscaf(data, label, args.worker_num, args.lr * 10, args.grad_noise,
                 #Coordinate_descend_genep, args.cp_rank, 1, args.out_iter_num , 
                 #args.epochs, args)
    #cd5 = subscaf(data, label, args.worker_num, args.lr * 2, args.grad_noise,
                 #Coordinate_descend_genep, args.cp_rank, 5, args.out_iter_num, 
                 #args.epochs, args)
    #cd10 = subscaf(data, label, args.worker_num, args.lr, args.grad_noise,
                 #Coordinate_descend_genep, args.cp_rank, args.inner_iter_num, args.out_iter_num , 
                 #args.epochs, args)
    #cd20 = subscaf(data, label, args.worker_num, args.lr / 2, args.grad_noise,
                 #Coordinate_descend_genep, args.cp_rank, 20, int(args.out_iter_num), 
                 #args.epochs, args)
    #cd50 = subscaf(data, label, args.worker_num, args.lr / 5, args.grad_noise,
                 #Coordinate_descend_genep, args.cp_rank, 50, int(args.out_iter_num), 
                 #args.epochs, args)
    #args.dual_re_proj = False
    #args.grad_down_lr = False
    #ide = subscaf(data, label, args.worker_num, args.lr, args.grad_noise,
                 #gene_ide_matrix, args.dim, args.inner_iter_num, args.out_iter_num, 
                 #args.epochs, args)
    #args.grad_down_lr = True

    #cd1 = get_outer_round_value(1, cd1)[0]
    #cd5 = get_outer_round_value(5, cd5)[0]
    #cd10 = get_outer_round_value(10, cd10)[0]
    #cd20 = get_outer_round_value(20, cd20)[0]
    #cd50 = get_outer_round_value(50, cd50)[0]
    #ide = get_outer_round_value(10, ide)[0]
    #exp3 = {
        #"tau=1": cd1,
        #"tau=5": cd5,
        #"tau=10": cd10,
        #"tau=20": cd20,
        #"tau=50": cd50,
        #r'our-$P^k$=I': ide,
    #}

    #plot(exp3, '', 'exp3')


    args.dual_re_proj = True
    cd1 = subscaf(data, label, args.worker_num, args.lr, args.grad_noise,
                 Coordinate_descend_genep, 1, args.inner_iter_num, args.out_iter_num , 
                 args.epochs, args)
    cd3 = subscaf(data, label, args.worker_num, args.lr, args.grad_noise,
                 Coordinate_descend_genep, 3, args.inner_iter_num, args.out_iter_num , 
                 args.epochs, args)
    cd5 = subscaf(data, label, args.worker_num, args.lr, args.grad_noise,
                 Coordinate_descend_genep, args.cp_rank, args.inner_iter_num, args.out_iter_num , 
                 args.epochs, args)
    cd7 = subscaf(data, label, args.worker_num, args.lr, args.grad_noise,
                 Coordinate_descend_genep, 7, args.inner_iter_num, args.out_iter_num , 
                 args.epochs, args)
    cd10 = subscaf(data, label, args.worker_num, args.lr, args.grad_noise,
                 Coordinate_descend_genep, 10, args.inner_iter_num, args.out_iter_num , 
                 args.epochs, args)
    args.dual_re_proj = False

    cd1 = get_outer_round_value(10, cd1)[0]
    cd3 = get_outer_round_value(10, cd3)[0]
    cd5 = get_outer_round_value(10, cd5)[0]
    cd7 = get_outer_round_value(10, cd7)[0]
    cd10 = get_outer_round_value(10, cd10)[0]
    exp3 = {
        "CPDim=1": cd1,
        "CPDim=3": cd3,
        "CPDim=5": cd5,
        "CPDim=7": cd7,
        "CPDim=10": cd10,
    }

    plot(exp3, '', 'test')





