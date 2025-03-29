import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# data generation utility
def genLS(total_sample_size, d):
    a = np.random.randn(total_sample_size, d)
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


def SubScafNoCP(X, y, N, A, alpha, noise, compress_rank, inner_iter_num=10, maxite=500, epochs=10):

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
    plt.savefig(f"./LinearRegV2/{output_name}.png")
    plt.close()



np.random.seed(2022)
N, M = 20, 10 # N is the network size, M is the local data dimension
total_sample_size = 10000 # 8000 is the local data size at each node
X, y = genLS(total_sample_size, M)
w_sol = solLS(X, y)
Ws = np.ones((N,1))@w_sol.T


#msd_mean_subscaf = SubScaf(X, y, N, None, 0.01, 0, 5, 10, 3, 5).squeeze()
msd_mean_scaf = Scaf(X, y, N, None, 0.01, 0.01, 0, 10, 5000, 5).squeeze()



#dis_dict = {'Subspace Scaffold': msd_mean_subscaf,
            #'NonSubspace Scaffold': msd_mean_scaf}

#plot(dis_dict, 'Convergency Test', 'Convergenct_Test')


msd_mean_subscaf_lowg = SubScafLowGrad(X, y, N, None, 0.01, 0, 5, 10, 5000, 5).squeeze()


#dis_dict = {'SubScaf LowGrad': msd_mean_subscaf_lowg,
            #'NonSubspace Scaffold': msd_mean_scaf}

#plot(dis_dict, 'Lower Gradient Lead to Convergenct', 'Lower_gradient')


msd_mean_subscaf_lowg_lowlr = SubScafLowGradLowLr(X, y, N, None, 0.01, 0, 5, 10, 5000, 5).squeeze()

dis_dict = {'SubScaf LowGrad & LowLr': msd_mean_subscaf_lowg_lowlr,
            'SubScaf LowGrad': msd_mean_subscaf_lowg,
            'NonSubspace Scaffold': msd_mean_scaf,
            }


plot(dis_dict, 'Gradually Lower Lr Improve Behavior', 'Improved_lr')

msd_mean_subscaf_cp = SubScafNoCP(X, y, N, None, 0.01, 0, 10, 10, 5000, 5).squeeze()
msd_mean_subscaf_lowg_lowlr = SubScafLowGradLowLr(X, y, N, None, 0.01, 0, 10, 10, 5000, 5).squeeze()

dis_dict = {'SubScaf without CP': msd_mean_subscaf_cp,
            'SubScaf LowGrad & LowLr & CPDim10': msd_mean_subscaf_lowg_lowlr,
            'NonSubspace Scaffold': msd_mean_scaf,
            }

plot(dis_dict, 'No Compression', 'NoComp')