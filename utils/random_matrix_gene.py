import torch
import math

def gene_random_matrix(comp_dim, dim, method='cd', gradient=None):
    if method.lower() == 'rd':
        return Random_Normal_genep(comp_dim, dim)
    elif method.lower() == 'cd':
        return Coordinate_descend_genep(comp_dim, dim)
    elif method.lower() == 'ss':
        return Spherical_smoothing_genep(comp_dim, dim)
    elif method.lower() == 'svd':
        return SVD_gene(comp_dim, dim, gradient)
    elif method.lower() == 'idx':
        return IDX_gene(comp_dim, dim)
    else:
        assert False, "haven't define chosen compression matrix generation method"

def IDX_gene(comp_dim, dim):
    assert comp_dim == dim, "The compression dimension must fit dimension"
    return torch.eye(dim)

def Random_Normal_genep(comp_dim, dim):
    return torch.randn(comp_dim, dim) / math.sqrt(dim)

def Coordinate_descend_genep(comp_dim, dim):
    assert dim >= comp_dim, "compression dimension must be smaller than dimension"
    ide = torch.eye(dim)
    select_row = torch.randperm(dim)[:comp_dim]#sort().values
    sign = torch.randint(0, 2, (comp_dim, ))
    #sign = sign * 2 - 1
    #P = torch.sqrt(torch.tensor(dim / comp_dim)) * ide[select_row, :] * sign.unsqueeze(1)
    P = ide[select_row, :] * sign.unsqueeze(1)
    return P

def Spherical_smoothing_genep(comp_dim, dim):
    z = torch.randn(dim, dim)
    Q, R = torch.linalg.qr(z)
    D = torch.diag(torch.sign(torch.diag(R)))
    Q = torch.matmul(Q, D)
    R = torch.matmul(D, R)
    assert torch.allclose(torch.matmul(Q, R), z, atol=1e-5), "the QR decomposion is not accuracy"
    P = torch.sqrt(torch.tensor(dim / comp_dim)) * Q[:comp_dim, :]
    return P

def SVD_gene(comp_dim, dim, gradient):
    _, _, V = torch.linalg.svd(gradient, full_matrices=True)
    V_k = V[:comp_dim, :].contiguous()
    return V_k
    