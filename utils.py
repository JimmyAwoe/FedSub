import torch.optim as optim
import torch.nn as nn
import torch
import torch.distributed as dist

class SubScaffoldSGD(optim.Optimizer):
    def __init__(self, params, lr, compressmat, lbd, tao):
        defaults = dict(lr=lr, p=compressmat, lbd=lbd, tao=tao)
        super().__init__(params=params, defaults=defaults)
    
    def step(self):
        for group in self.param_groups:
            p = group['p']
            lbd = group['lbd']
            tao = group['tao']
            lr = group['lr']
            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                update = d_p + lbd / (lr * tao)
                p.data.add_(update, alpha=-lr)


class SubScaffoldAdam(optim.Optimizer):
    def __init__(self, params, lr, compressmat, lbd, tao, epi=1e-8, betas=(0.9, 0.999)):
        defaults = dict(lr=lr, 
                        p=compressmat, 
                        lbd=lbd, 
                        tao=tao, 
                        m=None, 
                        v=None, 
                        epi=epi, 
                        betas=betas)
        super().__init__(params=params, defaults=defaults)
    
    def step(self):
        for group in self.param_groups:
            p = group['p']
            lbd = group['lbd']
            tao = group['tao']
            lr = group['lr']
            beta1 = group['betas'][0]
            beta2 = group['betas'][1]
            epi = group['epi']
            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                update = d_p + lbd / (lr * tao)
                state = self.state[p]
                # initialize state
                if len(state) == 0:
                    state['step'] = 0
                    state['m'] = update.clone()
                    state['v'] = update * update
                
                state['step'] += 1
                m = state['m']
                v = state['v']

                m.mul_(beta1).add_(update, alpha=1-beta1)
                v.mul_(beta2).addcmul_(update, update, value=1-beta2)

                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']

                lr_t = lr * (bias_correction2 ** 0.5) / bias_correction1

                denom = v.sqrt().add_(epi)

                p.data.addcdiv_(m, denom, value=-lr_t)

def generate_compression_mat_random(in_feature, out_feature):
    compressmat = torch.normal(0, 1 / out_feature, size=(in_feature, out_feature))
    return compressmat 

def generate_compression_mat_svd(grad, rank):
    u, _, v = torch.linalg.svd(grad)
    if len(grad.shape) == 1:
        m = 0 
        n = 1
    else:
        m, n = grad.shape
    if m <= n:
        return u[:, :rank]
    else:
        return v[:, :rank]

class SubScafLinearClassifier(nn.Module):
    def __init__(self, rank, weight, compressmat, compression_mat_gene=generate_compression_mat_random):
        super().__init__()
        self.weight = weight
        self.comp_mat_gene = compression_mat_gene
        self.p = compressmat
        self.rank = rank
        self.b = nn.Parameter(torch.zeros(rank))

    def refresh(self, *args):
        with torch.no_grad():
            self.b.zero_()
            new_p = self.comp_mat_gene(*args).to(self.weight.device)
            # dist.all_reduce(new_p, op=dist.ReduceOp.AVG)
            dist.broadcast(new_p, src=0)
            self.p = new_p

    def forward(self, x):
        if len(x) == 1:
            bias = torch.ones(1).to(x.device)
            return torch.matmul(torch.cat((x, bias), dim=0), self.weight + self.p @ self.b)
        else:
            bias = torch.ones(x.shape[0], 1).to(x.device)
            return torch.matmul(torch.cat((x, bias), dim=1), self.weight + self.p @ self.b)

class SubScafAvgCPMatLinearClassifier(nn.Module):
    def __init__(self, rank, weight, compressmat, compression_mat_gene=generate_compression_mat_random):
        super().__init__()
        self.weight = weight
        self.comp_mat_gene = compression_mat_gene
        self.p = compressmat
        self.rank = rank
        self.b = nn.Parameter(torch.zeros(rank))

    def refresh(self, *args):
        with torch.no_grad():
            self.b.zero_()
            new_p = self.comp_mat_gene(*args).to(self.weight.device)
            dist.all_reduce(new_p, op=dist.ReduceOp.AVG)
            self.p = new_p

    def forward(self, x):
        if len(x) == 1:
            bias = torch.ones(1).to(x.device)
            return torch.matmul(torch.cat((x, bias), dim=0), self.weight + self.p @ self.b)
        else:
            bias = torch.ones(x.shape[0], 1).to(x.device)
            return torch.matmul(torch.cat((x, bias), dim=1), self.weight + self.p @ self.b)

class SubScafGaloreLinearClassifier(SubScafLinearClassifier):
    def __init__(self, rank, weight, compressmat, compression_mat_gene=generate_compression_mat_svd):
        if len(weight.shape) == 1:
            m = 0
            n = 1
        else:
            m, n = weight.shape
        if m <= n:
            self.aggregate = lambda weight, p, b: weight + p @ b
        else:
            self.aggregate = lambda weight, p, b: weight + b @ p.T
        super().__init__(rank, weight, compressmat, compression_mat_gene)
    
    def refresh(self, data, label, loss_fun):
        with torch.enable_grad():
            output = self.forward(data)
            loss = loss_fun(output, label)
            loss.backward()
        self.b.zero_()
        for p in self.parameters():
            new_p = self.comp_mat_gene(p.grad.data, self.rank).to(self.weight.device)
        # dist.all_reduce(new_p, op=dist.ReduceOp.AVG)
        dist.broadcast(new_p, src=0)
        self.p = new_p
    
    def forward(self, x):
        if len(x) == 1:
            bias = torch.ones(1).to(x.device)
            return torch.matmul(torch.cat((x, bias), dim=0), self.aggregate(self.weight, self.p, self.b))
        else:
            bias = torch.ones(x.shape[0], 1).to(x.device)
            return torch.matmul(torch.cat((x, bias), dim=1), self.aggregate(self.weight, self.p, self.b))


class basemodel(nn.Module):
    def __init__(self, in_feature, out_feature):
        super().__init__()
        self.linear = nn.Linear(in_feature, out_feature, bias=False)
        
    def forward(self, x):
        if len(x) == 1:
            bias = torch.ones(1).to(x.device)
            return self.linear(torch.cat((x, bias), dim=0)).squeeze(1)
        else:
            bias = torch.ones(x.shape[0], 1).to(x.device)
            return self.linear(torch.cat((x,bias), dim=1)).squeeze(1)


def ddp_setup():
    dist.init_process_group(backend="nccl")