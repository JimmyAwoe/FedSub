import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# data generation utility
def genLS(total_sample_size, d):
    a = np.random.randn(total_sample_size, d)
    ans = np.random.randn(d, 1)
    b = a @ ans + 0.1 * np.random.randn(total_sample_size, 1)
    return a, b

def genHeLS(total_sample_size, d, worker_num):
    assert total_sample_size % worker_num == 0, "please make sure total_sample can be divided for each worker equally"
    mean = np.random.uniform(low=-1, high=1, size=worker_num)
    scale = np.sqrt(np.random.uniform(low=0.5, high=1.5, size=worker_num))
    a_list = []
    for i in range(worker_num):
        a_list.append(np.random.normal(loc=mean[i], scale=scale[i], size=(total_sample_size // worker_num, d)))
    a = np.concatenate(a_list, axis=0)
    ans = np.random.randn(d, 1)
    b = a @ ans + 0.1 * np.random.randn(total_sample_size, 1)
    return a, b


# Solution utility
def solLS(A, b):
    x_sol = np.linalg.inv(A.T@A)@(A.T@b)
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
            print('Centralized SGD epoch index:', e)
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

def SubScaf(X, y, N, A, alpha, noise, compress_rank, inner_iter_num=10, maxite=500, epochs=10):
    """
    X: 输入数据
    y: label
    N: worker数量
    alpha: 学习率
    alpha_bar: 外部学习率
    noise: 梯度的噪声
    compress_rank: 压缩维度
    inner_iter_num: 内部迭代次数
    maxite: 外部迭代次数
    epochs: 进行多少次重复来平均msd
    """
    R = np.ones((N,N))/N
    _, M = X.shape
    msd_mean = np.zeros((maxite * inner_iter_num, 1))
    
    for e in range(epochs):
        
        if (e+1)%1 == 0:
            print('Centralized SGD epoch index:', e)
    
        W = np.zeros((N, M))
        msd = np.zeros((maxite * inner_iter_num, 1))
        lbd = np.zeros((N, compress_rank))
        for ite in range(maxite):
            b = np.zeros((N, compress_rank))
            P = gene_random_matrix(M, compress_rank)
            for in_ite in range(inner_iter_num):
                G = ls_full_grad_dist_compression(X, y, W + multiply_with_row(P, b), P) + noise * np.random.randn(N, compress_rank)
                b -= alpha * (G + lbd / (alpha * inner_iter_num))
                msd[ite * inner_iter_num + in_ite] = np.sum(((W + multiply_with_row(P, b))-Ws)**2)/N / (np.sum((Ws)**2)/N)
            W += multiply_with_row(P, R.dot(b))
            lbd += b - R.dot(b)
        msd_mean += msd
    msd_mean = msd_mean/epochs
    return msd_mean


def SubScafLowGrad(X, y, N, A, alpha, noise, compress_rank, inner_iter_num=10, maxite=500, epochs=10):
    """
    梯度会除500
    X: 输入数据
    y: label
    N: worker数量
    alpha: 学习率
    alpha_bar: 外部学习率
    noise: 梯度的噪声
    compress_rank: 压缩维度
    inner_iter_num: 内部迭代次数
    maxite: 外部迭代次数
    epochs: 进行多少次重复来平均msd
    """

    R = np.ones((N,N))/N
    _, M = X.shape
    msd_mean = np.zeros((maxite * inner_iter_num, 1))
    
    for e in range(epochs):
        
        if (e+1)%1 == 0:
            print('Centralized SGD epoch index:', e)
    
        W = np.zeros((N, M))
        msd = np.zeros((maxite * inner_iter_num, 1))
        lbd = np.zeros((N, compress_rank))
        for ite in range(maxite):
            b = np.zeros((N, compress_rank))
            P = gene_random_matrix(M, compress_rank)
            for in_ite in range(inner_iter_num):
                G = ls_full_grad_dist_compression(X, y, W + multiply_with_row(P, b), P) + noise * np.random.randn(N, compress_rank)
                G /= 500
                b -= alpha * (G + lbd / (alpha * inner_iter_num))
                msd[ite * inner_iter_num + in_ite] = np.sum(((W + multiply_with_row(P, b))-Ws)**2)/N / (np.sum((Ws)**2)/N)
            W += multiply_with_row(P, R.dot(b))
            lbd += b - R.dot(b)
        msd_mean += msd
    msd_mean = msd_mean/epochs
    return msd_mean

def SubScafLowGradLowLr(X, y, N, A, alpha, noise, compress_rank, inner_iter_num=10, maxite=500, epochs=10):
    """
    学习率会逐步下降
    X: 输入数据
    y: label
    N: worker数量
    alpha: 学习率
    alpha_bar: 外部学习率
    noise: 梯度的噪声
    compress_rank: 压缩维度
    inner_iter_num: 内部迭代次数
    maxite: 外部迭代次数
    epochs: 进行多少次重复来平均msd
    """

    R = np.ones((N,N))/N
    _, M = X.shape
    msd_mean = np.zeros((maxite * inner_iter_num, 1))
    alpha_record = alpha
    for e in range(epochs):
        alpha = alpha_record
        if (e+1)%1 == 0:
            print('Centralized SGD epoch index:', e)
    
        W = np.zeros((N, M))
        msd = np.zeros((maxite * inner_iter_num, 1))
        lbd = np.zeros((N, compress_rank))
        sign = 1
        for ite in range(maxite):
            if ite > maxite / 10  * sign:
                alpha /= 10
                sign += 1
            b = np.zeros((N, compress_rank))
            P = gene_random_matrix(M, compress_rank)
            for in_ite in range(inner_iter_num):
                G = ls_full_grad_dist_compression(X, y, W + multiply_with_row(P, b), P) + noise * np.random.randn(N, compress_rank)
                G /= 500
                b -= alpha * (G + lbd / (alpha * inner_iter_num))
                msd[ite * inner_iter_num + in_ite] = np.sum(((W + multiply_with_row(P, b))-Ws)**2)/N / (np.sum((Ws)**2)/N)
            W += multiply_with_row(P, R.dot(b))
            lbd += b - R.dot(b)
        msd_mean += msd
    msd_mean = msd_mean/epochs
    return msd_mean



def SubScafLowGradLowLrDualReProj(X, y, N, gene_fun, alpha, noise, compress_rank, inner_iter_num=10, maxite=500, epochs=10):
    """
    阶段性降低学习率，梯度/500，对偶变量重投影
    X: 输入数据
    y: label
    N: worker数量
    gene_fun: 生成随机矩阵的方式
    alpha: 学习率
    alpha_bar: 外部学习率
    noise: 梯度的噪声
    compress_rank: 压缩维度
    inner_iter_num: 内部迭代次数
    maxite: 外部迭代次数
    epochs: 进行多少次重复来平均msd
    """

    R = np.ones((N,N))/N
    _, M = X.shape
    msd_mean = np.zeros((maxite * inner_iter_num, 1))
    alpha_record = alpha
    for e in range(epochs):
        alpha = alpha_record
        if (e+1)%1 == 0:
            print('Centralized SGD epoch index:', e)
    
        W = np.zeros((N, M))
        msd = np.zeros((maxite * inner_iter_num, 1))
        lbd = np.zeros((N, compress_rank))
        sign = 1
        for ite in range(maxite):
            if ite > maxite / 10  * sign:
                alpha /= 10
                sign += 1
            b = np.zeros((N, compress_rank))
            P = gene_fun(M, compress_rank)
            if ite > 0:
                lbd = multiply_with_row(P.T, lbd)
            for in_ite in range(inner_iter_num):
                G = ls_full_grad_dist_compression(X, y, W + multiply_with_row(P, b), P) + noise * np.random.randn(N, compress_rank)
                G /= 500
                b -= alpha * (G + lbd / (alpha * inner_iter_num))
                msd[ite * inner_iter_num + in_ite] = np.sum(((W + multiply_with_row(P, b))-Ws)**2)/N / (np.sum((Ws)**2)/N)
            W += multiply_with_row(P, R.dot(b))
            lbd += b - R.dot(b)
            lbd = multiply_with_row(P, lbd)
        msd_mean += msd
    msd_mean = msd_mean/epochs
    return msd_mean
    

def SubScafLowGradLowLrSOTA(X, y, N, gene_fun, alpha, noise, compress_rank, subspace_iter_num, inner_iter_num=10, maxite=500, epochs=10):
    """

    X: 输入数据
    y: label
    N: worker数量
    gene_fun: 生成随机矩阵的方式
    alpha: 学习率
    alpha_bar: 外部学习率
    noise: 梯度的噪声
    compress_rank: 压缩维度
    inner_iter_num: 内部迭代次数
    maxite: 外部迭代次数
    epochs: 进行多少次重复来平均msd
    """

    R = np.ones((N,N))/N
    _, M = X.shape
    msd_mean = np.zeros((maxite * inner_iter_num, 1))
    alpha_record = alpha
    for e in range(epochs):
        alpha = alpha_record
        if (e+1)%1 == 0:
            print('Centralized SGD epoch index:', e)
    
        W = np.zeros((N, M))
        msd = np.zeros((maxite * inner_iter_num, 1))
        lbd = np.zeros((N, compress_rank))
        sign = 1
        for ite in range(maxite):
            if ite > maxite / 5  * sign:
                alpha /= 10
                sign += 1
            b = np.zeros((N, compress_rank))
            if ite % subspace_iter_num == 0:
                if ite > 0:
                    lbd = multiply_with_row(P, lbd)
                P = gene_fun(M, compress_rank)
                if ite > 0:
                    lbd = multiply_with_row(P.T, lbd)
            for in_ite in range(inner_iter_num):
                G = ls_full_grad_dist_compression(X, y, W + multiply_with_row(P, b), P) + noise * np.random.randn(N, compress_rank)
                G /= 500
                #f = sign / (sign + 1)
                b -= alpha * (G + lbd / (alpha * inner_iter_num))
                msd[ite * inner_iter_num + in_ite] = np.sum(((W + multiply_with_row(P, b))-Ws)**2)/N / (np.sum((Ws)**2)/N)
            W += multiply_with_row(P, R.dot(b))
            lbd += b - R.dot(b)
        msd_mean += msd
    msd_mean = msd_mean/epochs
    return msd_mean


def SubScafLowGradLowLrIntOut(X, y, N, A, alpha, noise, compress_rank, 
                              subspace_iter_num, inner_iter_num=10, maxite=500, epochs=10):
    """
    学习率会逐步下降
    X: 输入数据
    y: label
    N: worker数量
    alpha: 学习率
    alpha_bar: 外部学习率
    noise: 梯度的噪声
    compress_rank: 压缩维度
    inner_iter_num: 内部迭代次数
    maxite: 外部迭代次数
    epochs: 进行多少次重复来平均msd
    """

    R = np.ones((N,N))/N
    _, M = X.shape
    msd_mean = np.zeros((maxite * inner_iter_num, 1))
    alpha_record = alpha
    for e in range(epochs):
        alpha = alpha_record
        if (e+1)%1 == 0:
            print('Centralized SGD epoch index:', e)
    
        W = np.zeros((N, M))
        msd = np.zeros((maxite * inner_iter_num, 1))
        lbd = np.zeros((N, compress_rank))
        sign = 1
        for ite in range(maxite):
            if ite > maxite / 10  * sign:
                alpha /= 10
                sign += 1
            b = np.zeros((N, compress_rank))
            if ite % subspace_iter_num == 0:
                P = gene_random_matrix(M, compress_rank)
            for in_ite in range(inner_iter_num):
                G = ls_full_grad_dist_compression(X, y, W + multiply_with_row(P, b), P) + noise * np.random.randn(N, compress_rank)
                G /= 500
                b -= alpha * (G + lbd / (alpha * inner_iter_num))
                msd[ite * inner_iter_num + in_ite] = np.sum(((W + multiply_with_row(P, b))-Ws)**2)/N / (np.sum((Ws)**2)/N)
            W += multiply_with_row(P, R.dot(b))
            lbd += b - R.dot(b)
        msd_mean += msd
    msd_mean = msd_mean/epochs
    return msd_mean


def SubScafLowGradLowLrCP(X, y, N, gene_fun, alpha, noise, compress_rank, inner_iter_num=10, maxite=500, epochs=10):
    """
    学习率会逐步下降
    X: 输入数据
    y: label
    N: worker数量
    alpha: 学习率
    alpha_bar: 外部学习率
    noise: 梯度的噪声
    compress_rank: 压缩维度
    inner_iter_num: 内部迭代次数
    maxite: 外部迭代次数
    epochs: 进行多少次重复来平均msd
    """
    R = np.ones((N,N))/N
    _, M = X.shape
    msd_mean = np.zeros((maxite * inner_iter_num, 1))
    alpha_record = alpha
    for e in range(epochs):
        alpha = alpha_record
        if (e+1)%1 == 0:
            print('Centralized SGD epoch index:', e)
    
        W = np.zeros((N, M))
        msd = np.zeros((maxite * inner_iter_num, 1))
        lbd = np.zeros((N, compress_rank))
        sign = 1
        for ite in range(maxite):
            if ite > maxite / 10  * sign:
                alpha /= 10
                sign += 1
            b = np.zeros((N, compress_rank))
            P = gene_fun(M, compress_rank)
            for in_ite in range(inner_iter_num):
                G = ls_full_grad_dist_compression(X, y, W + multiply_with_row(P, b), P) 
                G /= 500
                b -= alpha * (G + lbd / (alpha * inner_iter_num))
                msd[ite * inner_iter_num + in_ite] = np.sum(((W + multiply_with_row(P, b))-Ws)**2)/N / (np.sum((Ws)**2)/N)
            W += multiply_with_row(P, R.dot(b))
            lbd += b - R.dot(b)
        msd_mean += msd
    msd_mean = msd_mean/epochs
    return msd_mean


def SubScafLowGradLowLrVarIn(X, y, N, gene_fun, alpha, noise, compress_rank, inner_iter_num=10, maxite=500, epochs=10):
    """
    学习率会逐步下降
    X: 输入数据
    y: label
    N: worker数量
    gene_fun: 生成压缩矩阵方式
    alpha: 学习率
    alpha_bar: 外部学习率
    noise: 梯度的噪声
    compress_rank: 压缩维度
    inner_iter_num: 内部迭代次数
    maxite: 外部迭代次数
    epochs: 进行多少次重复来平均msd
    """
    R = np.ones((N,N))/N
    _, M = X.shape
    msd_mean = np.zeros((maxite * 50, 1))
    alpha_record = alpha
    in_record = inner_iter_num
    for e in range(epochs):
        length = 0
        alpha = alpha_record
        inner_iter_num = in_record
        if (e+1)%1 == 0:
            print('Centralized SGD epoch index:', e)
    
        W = np.zeros((N, M))
        msd = np.zeros((maxite * 50, 1))
        lbd = np.zeros((N, compress_rank))
        sign = 1
        factor = 1
        for ite in range(maxite):
            if ite > maxite / 10  * sign:
                factor *= 2
                sign += 1
            b = np.zeros((N, compress_rank))
            P = gene_fun(M, compress_rank)
            for in_ite in range(inner_iter_num):
                G = ls_full_grad_dist_compression(X, y, W + multiply_with_row(P, b), P) 
                G /= 500
                b -= alpha * (G + lbd / (alpha * inner_iter_num))
                msd[length] = np.sum(((W + multiply_with_row(P, b))-Ws)**2)/N / (np.sum((Ws)**2)/N)
                length += 1
            W += multiply_with_row(P, R.dot(b))
            lbd += b - R.dot(b)
        msd_mean[:length] += msd[:length]
    msd_mean = msd_mean[:length] / epochs
    return msd_mean


def SubScafNoCP(X, y, N, A, alpha, noise, compress_rank, inner_iter_num=10, maxite=500, epochs=10):
    """
    学习率会逐步下降
    X: 输入数据
    y: label
    N: worker数量
    alpha: 学习率
    alpha_bar: 外部学习率
    noise: 梯度的噪声
    compress_rank: 压缩维度
    inner_iter_num: 内部迭代次数
    maxite: 外部迭代次数
    epochs: 进行多少次重复来平均msd
    """

    R = np.ones((N,N))/N
    _, M = X.shape
    msd_mean = np.zeros((maxite * inner_iter_num, 1))
    
    for e in range(epochs):
        
        if (e+1)%1 == 0:
            print('Centralized SGD epoch index:', e)
    
        W = np.zeros((N, M))
        msd = np.zeros((maxite * inner_iter_num, 1))
        lbd = np.zeros((N, compress_rank))
        for ite in range(maxite):
            b = np.zeros((N, compress_rank))
            for in_ite in range(inner_iter_num):
                G  = ls_full_grad_dist(X, y, W + b)
                G /= 500
                b -= alpha * (G + lbd / (alpha * inner_iter_num))
                msd[ite * inner_iter_num + in_ite] = np.sum(((W + b)-Ws)**2)/N / (np.sum((Ws)**2)/N)
            W += R.dot(b)
            lbd += b - R.dot(b)
        msd_mean += msd
    msd_mean = msd_mean/epochs
    return msd_mean 

def plot(data, title, output_name):
    plt.figure(figsize=(6,4))
    plt.title(title)
    plt.grid(True)
    plt.xlabel('Step')
    plt.ylabel('Relative Distance')
    for name, value in data.items():
        x = np.arange(len(value))
        plt.plot(x, value, linewidth=2, alpha=0.3)
        color = plt.gca().lines[-1].get_color()
        ewm = pd.Series(value).ewm(alpha=0.1).mean()
        plt.plot(x, ewm, color=color, linewidth=1, label=name)    
        plt.axhline(y=ewm.iat[-1], color=color, linestyle='--')
        plt.yscale('log')
    plt.legend()
    plt.savefig(f"./LinearRegV3/{output_name}.png")
    plt.close()



np.random.seed(2022)
N, M = 20, 10 # N is the network size, M is the local data dimension
total_sample_size = 10000 # 8000 is the local data size at each node
X, y = genLS(total_sample_size, M)
w_sol = solLS(X, y)
Ws = np.ones((N,1))@w_sol.T


#msd_mean_subscaf = SubScaf(X, y, N, None, 0.01, 0, 5, 10, 3, 5).squeeze()
msd_mean_scaf = Scaf(X, y, N, None, 0.01, 0.01, 0, 10, 10000, 1).squeeze()


dis_dict = {
            'NonSubspace Scaffold': msd_mean_scaf,
            }

plot(dis_dict, 'test', 'test')


#dis_dict = {'Subspace Scaffold': msd_mean_subscaf,
            #'NonSubspace Scaffold': msd_mean_scaf}

#plot(dis_dict, 'Convergency Test', 'Convergenct_Test')


#msd_mean_subscaf_lowg = SubScafLowGrad(X, y, N, None, 0.01, 0, 5, 10, 5000, 5).squeeze()


#dis_dict = {'SubScaf LowGrad': msd_mean_subscaf_lowg,
            #'NonSubspace Scaffold': msd_mean_scaf}

#plot(dis_dict, 'Lower Gradient Lead to Convergenct', 'Lower_gradient')


#msd_mean_subscaf_lowg_lowlr = SubScafLowGradLowLr(X, y, N, None, 0.01, 0, 5, 10, 5000, 5).squeeze()

#dis_dict = {'SubScaf LowGrad & LowLr': msd_mean_subscaf_lowg_lowlr,
            #'SubScaf LowGrad': msd_mean_subscaf_lowg,
            #'NonSubspace Scaffold': msd_mean_scaf,
            #}


#plot(dis_dict, 'Gradually Lower Lr Improve Behavior', 'Improved_lr')

#msd_mean_subscaf_cp = SubScafNoCP(X, y, N, None, 0.01, 0, 10, 10, 5000, 5).squeeze()
#msd_mean_subscaf_lowg_lowlr = SubScafLowGradLowLr(X, y, N, None, 0.01, 0, 10, 10, 5000, 5).squeeze()

#dis_dict = {'SubScaf without CP': msd_mean_subscaf_cp,
            #'SubScaf LowGrad & LowLr & CPDim10': msd_mean_subscaf_lowg_lowlr,
            #'NonSubspace Scaffold': msd_mean_scaf,
            #}

#plot(dis_dict, 'No Compression', 'NoComp')


#msd_mean_subscaf_lowg_lowlr10 = SubScafLowGradLowLr(X, y, N, None, 0.01, 0, 10, 10, 5000, 5).squeeze()
#msd_mean_subscaf_lowg_lowlr8 = SubScafLowGradLowLr(X, y, N, None, 0.01, 0, 8, 10, 5000, 5).squeeze()
#msd_mean_subscaf_lowg_lowlr5 = SubScafLowGradLowLr(X, y, N, None, 0.01, 0, 5, 10, 5000, 5).squeeze()
#msd_mean_subscaf_lowg_lowlr3 = SubScafLowGradLowLr(X, y, N, None, 0.01, 0, 3, 10, 5000, 5).squeeze()

#dis_dict = {
            #'SubScaf LowGrad & LowLr & CPDim10': msd_mean_subscaf_lowg_lowlr10,
            #'SubScaf LowGrad & LowLr & CPDim8': msd_mean_subscaf_lowg_lowlr8,
            #'SubScaf LowGrad & LowLr & CPDim5': msd_mean_subscaf_lowg_lowlr5,
            #'SubScaf LowGrad & LowLr & CPDim3': msd_mean_subscaf_lowg_lowlr3,
            #'NonSubspace Scaffold': msd_mean_scaf,
            #}

#plot(dis_dict, 'Different CP Dimention', 'DiffCPdim')

#msd_mean_subscaf_lowg_lowlr2 = SubScafLowGradLowLr(X, y, N, None, 0.01, 0, 5, 2, 25000, 5).squeeze()
#msd_mean_subscaf_lowg_lowlr5 = SubScafLowGradLowLr(X, y, N, None, 0.01, 0, 5, 5, 10000, 5).squeeze()
#msd_mean_subscaf_lowg_lowlr10 = SubScafLowGradLowLr(X, y, N, None, 0.01, 0, 5, 10, 5000, 5).squeeze()
#msd_mean_subscaf_lowg_lowlr20 = SubScafLowGradLowLr(X, y, N, None, 0.01, 0, 5, 20, 2500, 5).squeeze()
#msd_mean_subscaf_lowg_lowlr50 = SubScafLowGradLowLr(X, y, N, None, 0.01, 0, 5, 50, 1000, 5).squeeze()

#dis_dict = {
            #'SubScaf LowGrad & LowLr & InStep2': msd_mean_subscaf_lowg_lowlr2,
            #'SubScaf LowGrad & LowLr & InStep5': msd_mean_subscaf_lowg_lowlr5,
            #'SubScaf LowGrad & LowLr & InStep10': msd_mean_subscaf_lowg_lowlr10,
            #'SubScaf LowGrad & LowLr & InStep20': msd_mean_subscaf_lowg_lowlr20,
            #'SubScaf LowGrad & LowLr & InStep50': msd_mean_subscaf_lowg_lowlr50,
            #'NonSubspace Scaffold': msd_mean_scaf,
            #}

#plot(dis_dict, 'Different Inner Step', 'DiffInStep')


#msd_mean_subscaf_lowg_lowlr_rd = SubScafLowGradLowLrCP(X, y, N, 
                                                       #gene_random_matrix, 0.01, 0, 10, 10, 5000, 5).squeeze()
#msd_mean_subscaf_lowg_lowlr_cd = SubScafLowGradLowLrCP(X, y, N, 
                                                       #Coordinate_descend_genep, 0.01, 0, 10, 10, 5000, 5).squeeze()
#msd_mean_subscaf_lowg_lowlr_ss = SubScafLowGradLowLrCP(X, y, N, 
                                                       #Spherical_smoothing_genep, 0.01, 0, 10, 10, 5000, 5).squeeze()
#msd_mean_subscaf_lowg_lowlr_ide = SubScafLowGradLowLrCP(X, y, N, 
                                                       #gene_ide_matrix, 0.01, 0, 10, 10, 5000, 5).squeeze()
#msd_mean_subscaf_cp = SubScafNoCP(X, y, N, None, 0.01, 0, 10, 10, 5000, 5).squeeze()

#dis_dict = {
            #'SubScaf with RD': msd_mean_subscaf_lowg_lowlr_rd,
            #'SubScaf with CD': msd_mean_subscaf_lowg_lowlr_cd,
            #'SubScaf with SS': msd_mean_subscaf_lowg_lowlr_ss,
            #'SubScaf with IDE': msd_mean_subscaf_lowg_lowlr_ide,
            #}

#plot(dis_dict, 'Different Generation Method', 'DiffGene')



#msd_mean_subscaf_lowg_lowlr_rd = SubScafLowGradLowLrCP(X, y, N, 
                                                       #gene_random_matrix, 0.01, 0, 5, 10, 5000, 5).squeeze()
#msd_mean_subscaf_lowg_lowlr_cd = SubScafLowGradLowLrCP(X, y, N, 
                                                       #Coordinate_descend_genep, 0.01, 0, 5, 10, 5000, 5).squeeze()
#msd_mean_subscaf_lowg_lowlr_ss = SubScafLowGradLowLrCP(X, y, N, 
                                                       #Spherical_smoothing_genep, 0.01, 0, 5, 10, 5000, 5).squeeze()

#dis_dict = {
            #'SubScaf with RD': msd_mean_subscaf_lowg_lowlr_rd,
            #'SubScaf with CD': msd_mean_subscaf_lowg_lowlr_cd,
            #'SubScaf with SS': msd_mean_subscaf_lowg_lowlr_ss,
            #}

#plot(dis_dict, 'Different Generation Method in CPDim 5', 'DiffGeneIn5')


#msd_mean_subscaf_lowg_lowlrIntOut1 = SubScafLowGradLowLrIntOut(X, y, N, None, 0.01, 0, 5, 1, 10, 5000, 5).squeeze()
#msd_mean_subscaf_lowg_lowlrIntOut5 = SubScafLowGradLowLrIntOut(X, y, N, None, 0.01, 0, 5, 5, 10, 5000, 5).squeeze()
#msd_mean_subscaf_lowg_lowlrIntOut10 = SubScafLowGradLowLrIntOut(X, y, N, None, 0.01, 0, 5, 10, 10, 5000, 5).squeeze()
#msd_mean_subscaf_lowg_lowlrIntOut20 = SubScafLowGradLowLrIntOut(X, y, N, None, 0.01, 0, 5, 20, 10, 5000, 5).squeeze()
#msd_mean_subscaf_lowg_lowlrIntOut50 = SubScafLowGradLowLrIntOut(X, y, N, None, 0.01, 0, 5, 50, 10, 5000, 5).squeeze()

#dis_dict = {
            #'SubScaf LowGrad & LowLr & SubChange1': msd_mean_subscaf_lowg_lowlrIntOut1,
            #'SubScaf LowGrad & LowLr & SubChange5': msd_mean_subscaf_lowg_lowlrIntOut5,
            #'SubScaf LowGrad & LowLr & SubChange10': msd_mean_subscaf_lowg_lowlrIntOut10,
            #'SubScaf LowGrad & LowLr & SubChange20': msd_mean_subscaf_lowg_lowlrIntOut20,
            #'SubScaf LowGrad & LowLr & SubChange50': msd_mean_subscaf_lowg_lowlrIntOut50,
            #'NonSubspace Scaffold': msd_mean_scaf,
            #}

#plot(dis_dict, 'Different Subspace Change Frequency', 'DiffSubFreq')

#msd_mean_subscaf_lowg_lowlr_DualReproj = SubScafLowGradLowLrDualReProj(X, y, N, 
                                                                       #Coordinate_descend_genep, 0.01, 0, 5, 10, 5000, 5).squeeze()
#msd_mean_subscaf_lowg_lowlr = SubScafLowGradLowLr(X, y, N, None, 0.01, 0, 5, 10, 5000, 5).squeeze()

#dis_dict = {
            #'SubScaf DualReproj': msd_mean_subscaf_lowg_lowlr_DualReproj,
            #'SubScaf LowGrad & LowLr ': msd_mean_subscaf_lowg_lowlr,
            ##'NonSubspace Scaffold': msd_mean_scaf,
            #}

#plot(dis_dict, 'Reproject Dual Variable', 'ReProj')


#msd_mean_subscaf_lowg_lowlr_rd = SubScafLowGradLowLrVarIn(X, y, N, 
                                                       #gene_random_matrix, 0.01, 0, 10, 10, 5000, 1).squeeze()
#msd_mean_subscaf_lowg_lowlr_cd = SubScafLowGradLowLrVarIn(X, y, N, 
                                                       #Coordinate_descend_genep, 0.01, 0, 10, 10, 5000, 1).squeeze()
#msd_mean_subscaf_lowg_lowlr_ss = SubScafLowGradLowLrVarIn(X, y, N, 
                                                       #Spherical_smoothing_genep, 0.01, 0, 10, 10, 5000, 1).squeeze()
#msd_mean_subscaf_lowg_lowlr_ide = SubScafLowGradLowLrVarIn(X, y, N, 
                                                       #gene_ide_matrix, 0.01, 0, 10, 10, 5000, 1).squeeze()

#dis_dict = {
            #'SubScaf with RD': msd_mean_subscaf_lowg_lowlr_rd,
            #'SubScaf with CD': msd_mean_subscaf_lowg_lowlr_cd,
            #'SubScaf with SS': msd_mean_subscaf_lowg_lowlr_ss,
            #'SubScaf with IDE': msd_mean_subscaf_lowg_lowlr_ide,
            #}

#plot(dis_dict, 'Different Generation Method', 'DiffGene')


#msd_mean_subscaf_lowg_lowlr_sota = SubScafLowGradLowLrSOTA(X, y, N, Coordinate_descend_genep, 0.01, 0, 5, 
                                                                       #50, 2, 50000, 1).squeeze()
#msd_mean_subscaf_lowg_lowlr = SubScafLowGradLowLr(X, y, N, None, 0.01, 0, 5, 10, 10000, 1).squeeze()

#dis_dict = {
            #'SubScaf SOTA': msd_mean_subscaf_lowg_lowlr_sota,
            #'SubScaf LowGrad & LowLr ': msd_mean_subscaf_lowg_lowlr,
            #'NonSubspace Scaffold': msd_mean_scaf,
            #}

#plot(dis_dict, 'SOTA', 'sota')