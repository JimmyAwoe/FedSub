from __future__ import division

import numpy as np
import matplotlib.pyplot as plt
# from sklearn.linear_model import LogisticRegression

#from topology import *

import networkx as nx
import matplotlib

import copy

import scipy.io as sio
from scipy.sparse import diags

# Data generation utility
# Sigma_n is set to be 0.1.
def genLS(total_sample_size, d):
    N=total_sample_size
    A = np.random.randn(total_sample_size, d)
    b = A @ np.random.randn(d, 1) + 0.1 * np.random.randn(N, 1)

#     xa = np.random.randn(1, d)
#     xb = a @ np.random.randn(d, 1) + 0.1 * np.random.randn(1)
    
#     A = np.ones((total_sample_size,1))@xa
#     b = np.ones((total_sample_size,1))*xb
    
    return A, b

# Solution utility
def solLS(A, b):
    
    x_sol = np.linalg.inv(A.T@A)@(A.T@b)
    #print(A.T@(A@x_sol-b))
    
    return x_sol
# Gradient utility
def solP(A,P,b):
    x_sol = np.linalg.inv(P.T@A.T@A@P)@(P.T@A.T@b)
    return x_sol
    
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
def ls_full_lowrank_grad_dist(X,y,W,P,B):
    n, m = X.shape
    r=P.shape[1]
    Q = W.shape[0]
    N_agent = n//Q    
    G = np.zeros((Q, r))
    
    for k in range(Q):
        bk = B[k,:].reshape(r, 1)
        wk = W[k,:].reshape(m, 1)        
        Xk = X[k*N_agent:(k+1)*N_agent, :]
        yk = y[k*N_agent:(k+1)*N_agent].reshape(N_agent, 1)

        grad = P.T@Xk.T@(Xk@(wk+P@bk)-yk)
        G[k,:] = grad.T
    
    return G


def get_grad_and_loss(X,y,W):
    n, m = X.shape
    Q = W.shape[0]
    N_agent = n//Q    
    G = np.zeros((Q, m))
    loss=0.0
    w_mean=np.mean(W,axis=0)
    for k in range(Q):
         
        wk = W[k,:].reshape(m, 1)        
        Xk = X[k*N_agent:(k+1)*N_agent, :]
        yk = y[k*N_agent:(k+1)*N_agent].reshape(N_agent, 1)

        grad = Xk.T@(Xk@wk-yk)
        G[k,:] = grad.T
        loss+=0.5*np.sum((Xk@w_mean-yk)**2)
    
    return G,loss/Q

# Cyclic network
def generate_cycle_network(N):
    # N is the number of agents
    
    diag_m1 = (1./3) * np.ones(N-1)
    diag_0  = (1./3) * np.ones(N)
    diag_p1 = (1./3) * np.ones(N-1)
    
    A = diags([diag_m1, diag_0, diag_p1], [-1, 0, 1]).toarray()
    A[0,-1] = 1./3
    A[-1,0] = 1./3
    
    return A
# Grid network
def MeshGrid2DGraph(size, shape = None):
    """Generate 2D MeshGrid structure of graph.
    Assume shape = (nrow, ncol), when shape is provided, a meshgrid of nrow*ncol will be generated.
    when shape is not provided, nrow and ncol will be the two closest factors of size.
    For example: size = 24, nrow and ncol will be 4 and 6, respectively.
    We assume  nrow will be equal to or smaller than ncol.
    If size is a prime number, nrow will be 1, and ncol will be size, which degrades the topology
    into a linear one.
    Example: A MeshGrid2DGraph with 16 nodes:
    .. plot::
        :context: close-figs
        >>> import networkx as nx
        >>> from bluefog.common import topology_util
        >>> G = topology_util.MeshGrid2DGraph(16)
        >>> nx.draw_spring(G)
    """

    assert size > 0
    if shape is None:
        i = int(np.sqrt(size))
        while size % i != 0:
            i -= 1
        shape = (i, size//i)
    nrow, ncol = shape
    assert size == nrow*ncol, "The shape doesn't match the size provided."
    topo = np.zeros((size, size))
    for i in range(size):
        topo[i][i] = 1.0
        if (i+1) % ncol != 0:
            topo[i][i+1] = 1.0
            topo[i+1][i] = 1.0
        if i+ncol < size:
            topo[i][i+ncol] = 1.0
            topo[i+ncol][i] = 1.0

    # According to Hasting rule (Policy 1) in https://arxiv.org/pdf/1702.05122.pdf
    # The neighbor definition in the paper is different from our implementation,
    # which includes the self node.
    topo_neighbor_with_self = [np.nonzero(topo[i])[0] for i in range(size)]
    for i in range(size):
        for j in topo_neighbor_with_self[i]:
            if i != j:
                topo[i][j] = 1.0/max(len(topo_neighbor_with_self[i]),
                                     len(topo_neighbor_with_self[j]))
        topo[i][i] = 2.0-topo[i].sum()
    G = nx.from_numpy_array(topo, create_using=nx.DiGraph)
    return G
# exponential network
def PowerTwoRing(N):
    incidenceMat = np.zeros((N,N))
    
    base_row = np.zeros(N)
    count_ones = 0
    for i in range(N):
        if i&(i-1) == 0:
            base_row[i] = 1
            count_ones += 1
            
    for i in range(N):
        incidenceMat[i,:] = np.roll(base_row,i)
        
    W = incidenceMat/count_ones
    Num_neighbor = count_ones
    
    return W, incidenceMat, Num_neighbor
# random network
def generate_network(N_node=20, method='metropolis_rule', **kwargs):
    '''
        Wrap function to generate topology. 
        Then by using avg or metropolis rule to generate the combination matrix
        current support key words:   prob----------connected probability

        method only support 'average_rule' and 'metropolis_rule' now
    '''
    
    # start with 1 because assuming every node has self-loop
    indegree = np.ones((N_node,1))
    G = generate_topology(N_node, kwargs.get('prob', 0.25))
    for edge in G.edges():
        indegree[edge[0]] += 1
        indegree[edge[1]] += 1

    def avg_rule(G, indegree):
        N_node = indegree.shape[0]
        A = np.zeros((N_node,N_node))
        for e1, e2 in G.edges():
            A[e1,e2] = 1./indegree[e2]
            A[e2,e1] = 1./indegree[e1]

        for i in range(N_node):
            A[i,i] = 1. - np.sum(A[:,i])
        return A

    def metropolis_rule(G, indegree):
        N_node = indegree.shape[0]
        A = np.zeros((N_node,N_node))
        for e1, e2 in G.edges():
            A[e1,e2] = 1./max(indegree[e1], indegree[e2])
            A[e2,e1] = 1./max(indegree[e1], indegree[e2])

        for i in range(N_node):
            A[i,i] = 1. - np.sum(A[:,i])
        return A 

    option = {'average_rule': avg_rule,
              'metropolis_rule': metropolis_rule}

    if method not in option:
        print ('Currently, only support "average_rule" and "metropolis_rule"')

    return option[method](G, indegree), G 
def generat_P(m,N,r):
    """
    生成一个大小为 (m+1, N, r) 的张量，每个 N * r 的子矩阵都从 St(N, r) 随机生成。
    """
    # 初始化 P，大小为 (m, N, r)
    P = np.zeros((m+1, N, r))/r
    
    # 对于每个 m 的子矩阵生成一个对应的随机矩阵 Z
    for i in range(m+1):
        # Step 1: 生成 N * R 的标准正态分布矩阵 Z
        Z = np.random.randn(N, r)/r
        
        # Step 2: 计算 (Z @ Z.T) 的逆的平方根
        ##ZTZ_inv_sqrt = np.linalg.inv(Z.T@ Z) ** -1/2
        
        # Step 3: 生成对应的随机矩阵 X
        ##X = Z @ ZTZ_inv_sqrt
        
        # 将生成的 X 存储到 P 中
        P[i] = Z
   # print(P)
    return P
def generate_Stiefel_tensor(m, N, r):
    """
    生成一个大小为 (m+1, N, r) 的张量，其中每个 N * r 的子矩阵
    都是从 Stiefel 流形 St(N, r) 随机生成的。

    参数：
    m -- 额外的维度，使得张量的第一个维度为 m+1
    N -- 矩阵的行数
    r -- 矩阵的列数 (r <= N)

    返回：
    tensor -- 形状为 (m+1, N, r) 的 NumPy 张量
    """
    tensor = np.zeros((m+1, N, r))

    for i in range(m+1):
        Z = np.random.randn(N, r)  # 生成 N × r 的标准正态分布矩阵
        Q, _ = np.linalg.qr(Z)  # 进行 QR 分解，仅取 Q
        tensor[i] = Q

    return tensor
def csgd_stochastic(X, y,sol, N, A, alpha, noise, maxite=500, epochs=10):
    M=X.shape[1]
     
    total_sample_size, M = X.shape
     
    con_err_record = []
    loss_record=[]
    dist_record=[]
    for e in range(epochs):
        
        if (e+1)%1 == 0:
            print('Centralized SGD epoch index:', e)
    
        w = np.zeros((M, ))
         
        
        for ite in range(maxite):

            
           
             
            G,loss=get_grad_and_loss(X, y, np.tile(w, (N, 1))   )
            G = G + noise * np.random.randn(N, M)
            w =  w - alpha * np.mean(G ,axis=0)
            dist=np.mean((w-sol)**2)*M
            dist_record.append(dist) 
            
            
            con_err_record.append( 0.0) 
            
            
            loss_record.append(loss)
        
    return  loss_record, dist_record, con_err_record
def dsgd_stochastic(X, y, sol,N, A, alpha, noise, maxite=500, epochs=10):
    M=X.shape[1]
    total_sample_size, M = X.shape
     
    con_err_record = []
    loss_record=[]
    dist_record=[]
    for e in range(epochs):
        
        if (e+1)%1 == 0:
            print('Decentralized SGD epoch index:', e)
    
        W = np.zeros((N, M))
        msd = np.zeros((maxite, 1))
        
        for ite in range(maxite):

            G,loss=get_grad_and_loss(X, y, W)
            G = G + noise * np.random.randn(N, M)
            W = A.dot(W - alpha * G)
            Wbar=np.mean(W, axis=0)
            dist=np.mean((Wbar-sol)**2)*W.shape[1]
            dist_record.append(dist) 
            
            
            con_err_record.append( np.sum((W-Wbar)**2)/N) 
            
            
            loss_record.append(loss)
             
       
        
     
        
    return loss_record, dist_record, con_err_record
def gradient_tracking_stochastic(X, y, sol, N, A, alpha, noise, maxite=500, epochs=10):
    M=X.shape[1]
     
    con_err_record = []
    loss_record=[]
    dist_record=[]
    aggre_con_error=[]
    for e in range(epochs):
        
        if (e+1)%1 == 0:
            print('Gradient-tracking epoch index:', e)
    
        alpha_used = alpha
        W = np.zeros((N, M))
        msd = np.zeros((maxite, 1))
        G_pre = np.zeros((N, M))
        Y = np.zeros((N, M))
        S=np.zeros((N, M))
        R= np.ones((N, N))/N
        for ite in range(maxite):

            
            G,loss=get_grad_and_loss(X, y, W)
            G = G + noise * np.random.randn(N, M)
            Y = A.dot(Y + G - G_pre)
            W = A.dot(W - alpha * Y)
            S+=W
            aggre_drift=np.sum((S-R.dot(S))**2)
            aggre_con_error.append(aggre_drift)
            Wbar=np.mean(W, axis=0)
            dist=np.mean((Wbar-sol)**2)*W.shape[1]
            dist_record.append(dist) 
           

            con_err_record.append( np.sum((W-Wbar)**2)/N) 
            G_pre = np.copy(G)
            
            loss_record.append(loss)
             
    
   
        
    return  loss_record, dist_record, con_err_record,aggre_con_error

def suda(X, y, sol, N, beta,A, B, C, alpha, noise, maxite=500, epochs=10):
    M=X.shape[1]
     
    con_err_record = []
    loss_record=[]
    dist_record=[]
    aggre_con_error=[]
    for e in range(epochs):
        
        if (e+1)%1 == 0:
            print('SUDA epoch index:', e)
    
        alpha_used = alpha
        W = np.zeros((N, M))
        msd = np.zeros((maxite, 1))
        G_pre = np.zeros((N, M))
        
        S=np.zeros((N, M))
        Z=np.zeros((N, M))
        R= np.ones((N, N))/N
        for ite in range(maxite):

            
            G,loss=get_grad_and_loss(X, y, W)
            G = G + noise * np.random.randn(N, M)
            
            
            W,Z = (A@C-B@B)@W - alpha * A@G -B.dot(Z), beta*Z+B@W
            
            S+=W
            aggre_drift=np.sum((S-R.dot(S))**2)
            aggre_con_error.append(aggre_drift)
            Wbar=np.mean(W, axis=0)
            dist=np.mean((Wbar-sol)**2)*W.shape[1]
            dist_record.append(dist) 
           

            con_err_record.append( np.sum((W-Wbar)**2)/N) 
             
            
            loss_record.append(loss)
             
    
   
        
    return  loss_record, dist_record, con_err_record,aggre_con_error


def lora_csgd (X,y,sol,N,P,alpha,noise,maxite=500,epochs=10):
    #N:node 个数，M：单个节点参数维度
    M=X.shape[1]
    
    con_err_record = []
    loss_record=[]
    dist_record=[]
    maxite_P=maxite//(P.shape[0]-1)
    subspace_n=P.shape[0]-1
    #print(subspace_n)
    r=P.shape[2]
    for e in range(epochs):
        
        if (e+1)%1 == 0:
            print('Lowrank-Gradient-tracking epoch index:', e)
    
        alpha_used = alpha
        w = np.zeros((M, ))
        con_err = np.zeros((maxite, 1))
        G_pre = np.zeros((N, r))
        G_pre_next = np.zeros((N, r))
        H = np.zeros((N, r))
        H_next = np.zeros((N, r))
        i=0
         
        for i in range (subspace_n):
            
            Pi=P[i,:,:]
            
            
            
            ite=0
            #Y = np.zeros((N, r))
            for ite in range(maxite_P):

                
                G,loss=get_grad_and_loss(X, y, np.tile(w, (N, 1)))
                
                 
                G = G + noise * np.random.randn(N, M)                
                
                w=w-alpha * np.mean(G@Pi@Pi.T ,axis=0)
                
                Wbar=w
                dist=np.mean((Wbar-sol)**2)*M
                dist_record.append(dist) 

                con_err_record.append( 0.0)  
                loss_record.append(loss)   
                
   

    return loss_record, dist_record, con_err_record





def lora_GT (X,y,sol,N,A,P,alpha,noise,maxite=500,epochs=10,next_tracking=True):
    #N:node 个数，M：单个节点参数维度
    M=X.shape[1]
     
    con_err_record = []
    loss_record=[]
    dist_record=[]
    aggre_con_error=[]
    maxite_P=maxite//(P.shape[0]-1)
    subspace_n=P.shape[0]-1
     
    r=P.shape[2]
    for e in range(epochs):
        
        if (e+1)%1 == 0:
            print('Lowrank-Gradient-tracking epoch index:', e)
    
        alpha_used = alpha
        W = np.zeros((N, M))
        con_err = np.zeros((maxite, 1))
        G_pre = np.zeros((N, r))
        G_pre_next = np.zeros((N, r))
        H = np.zeros((N, r))
        H_next = np.zeros((N, r))
        S=np.zeros((N, M))
        R= np.ones((N, N))/N 
        for i in range (subspace_n):
            
            Pi=P[i,:,:]
            PW=W@Pi
            W_res=W-PW@Pi.T
            
            
            ite=0
            #Y = np.zeros((N, r))
            for ite in range(maxite_P):

                
                G,loss=get_grad_and_loss(X, y, W)
                Wbar=np.mean(W, axis=0)
                dist=np.mean((Wbar-sol)**2)*W.shape[1]
                dist_record.append(dist)
                #G = ls_full_lowrank_grad_dist(X, y, W, Pi, B) + noise * np.random.randn(N, r)
                G = G + noise * np.random.randn(N, M)
                l=1.0
                
                
                H = A.dot(l*H + G@Pi - l*G_pre)
                H_next=A.dot(H_next + G@P[i+1,:,:] - G_pre_next)
                d=H
                PW = A.dot(PW - alpha *d)
                
                W=PW@Pi.T+W_res#-alpha*(G-G@Pi@Pi.T)
                 
                S+=W
                aggre_drift=np.sum((S-R.dot(S))**2)
                aggre_con_error.append(aggre_drift)

                con_err_record.append( np.sum((W-Wbar)**2)/N)  
                loss_record.append(loss)   
                
                G_pre = np.copy(G@Pi)
                G_pre_next = np.copy(G@P[i+1,:,:])
                
            
            
            #H = np.zeros((N, r))
            #G_pre = np.zeros((N, r))
            

            if next_tracking:
                H=H_next.copy()
                G_pre= G_pre_next.copy()
            else:
                H=H.copy()@Pi.copy().T@P[i+1,:,:]
                G_pre=G_pre.copy()@Pi.copy().T@P[i+1,:,:]
        
            
        
          
        
    

    return loss_record, dist_record, con_err_record,aggre_con_error

def find_orthogonal_complement(P):
    """
    输入正交矩阵 P (形状 n×r, r < n)
    返回正交补矩阵 Q (形状 n×(n−r)), 使得 Q 的列与 P 的列正交
    """
    n, r = P.shape
    # 1. 生成随机矩阵
    B = np.random.randn(n, n - r)
    
    # 2. 投影到正交补空间 (避免构造 n×n 矩阵)
    C = B - P @ (P.T @ B)  # 等价于 (I - P P^T)B
    
    # 3. 对投影后的矩阵进行QR分解
    Q, _ = np.linalg.qr(C, mode='reduced')
    return Q

def lora_suda(X, y, sol, N, beta,A, B, C,P, alpha,ite_P,decay=1.0, noise=0.0, epochs=10):
    M=X.shape[1]
     
    con_err_record = []
    loss_record=[]
    dist_record=[]
    aggre_con_error=[]
    
    subspace_n=P.shape[0]-1
     
    r=P.shape[2]
    
    
    
     
    W = np.zeros((N, M))
    
    W = np.zeros((N, M))
     
    

    S=np.zeros((N, M))
    PZ=np.zeros((N, r))
    Z=np.zeros((N, M))
    R= np.ones((N, N))/N 
    Pnext_Z_res =np.zeros((N, r))
    print('Lora_SUDA start') 
    alpha=alpha
    for i in range (subspace_n):
        
        
        alpha=alpha/(1+decay )   
        Pi=P[i,:,:]
        PW=W@Pi
        W_res=W-PW@Pi.T
        PZ=Z@Pi
        Z_res=Z-PZ@Pi.T 
        Q=find_orthogonal_complement(Pi)
        Z_res=( (Z_res@Q).mean(1).reshape(N,1) )@( Q.sum(1).reshape(1,M) )
        maxite_P=ite_P(i)
        for ite in range(maxite_P):

            
            G,loss=get_grad_and_loss(X, y, W)
            G = G + noise * np.random.randn(N, M)
            
            
            PW,PZ = (A@C-B@B)@PW - alpha * A@(G@Pi) -B.dot(PZ), beta*PZ+B@PW
            
            W=PW@Pi.T+W_res
            Z=PZ@Pi.T+Z_res
            S+=W
            aggre_drift=np.sum((S-R.dot(S))**2)
            aggre_con_error.append(aggre_drift)
            Wbar=np.mean(W, axis=0)
            dist=np.mean((Wbar-sol)**2)*W.shape[1]
            dist_record.append(dist) 
           

            con_err_record.append(np.sum((W-Wbar)**2)/N) 
             
            
            loss_record.append(loss)
             
         
       # con_err_record 记录每个worker得到的W和平均W之间的差距
       # loss_record 记录loss
       # dist_record  worker平均后与sol的mse
       # aggre_con_error 记录
        
    return  loss_record, dist_record, con_err_record,aggre_con_error