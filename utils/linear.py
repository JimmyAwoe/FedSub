from transformers.activations import ACT2FN
import torch.nn as nn
import torch
import torch.nn.functional as F


class SubScafLinear(nn.Linear):
    """
    Linear network with compressed dimension
    """
    def __init__(self, comp_dim: int, comp_mat: torch.Tensor, wraped_model: nn.Linear):
        self.comp_mat = comp_mat
        self.comp_dim = comp_dim
        device = wraped_model.weight.device
        dtype = wraped_model.weight.dtype
        factory_kwargs = {'device': device, 'dtype': dtype}
        bias = wraped_model.bias is not None
        super().__init__(wraped_model.in_features, wraped_model.out_features, 
                         bias, device, dtype)
        self.x = self.weight.detach().clone()
        #self.b = nn.Parameter(torch.zeros((comp_dim, wraped_model.in_features), **factory_kwargs))
        self.b = nn.Parameter(torch.zeros((wraped_model.out_features, comp_dim), **factory_kwargs))
        #nn.init.kaiming_uniform_(self.b, a=0, mode='fan_in', nonlinearity='relu')
        self.layer = _subscaflinear.apply
        del self.weight

    def forward(self, input):
        #return F.linear(input, self.b @ self.comp_mat + self.x, self.bias)
        return self.layer(input, self.comp_mat, self.b, self.x)
    
    def update(self, comp_mat=None, x=None, b=False):
        """
        Update compression matrix, x or b
        
        Be careful when update compressino before update x because that need the
        compressino matrix.
        """
        with torch.no_grad():
            if x is not None:
                self.x = x
            if comp_mat is not None:
                self.comp_mat = comp_mat
            if b:
                self.b.data = torch.zeros_like(self.b.data)

class _subscaflinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, comp_mat, b, x):
        ctx.save_for_backward(input @ comp_mat.T, b @ comp_mat + x)
        return F.linear(input, b @ comp_mat + x)

    @staticmethod
    def backward(ctx, grad_output):
        act_for_b, act_for_input = ctx.saved_tensors
        grad_comp_mat = grad_x = None
        grad_input = grad_output @ act_for_input
        grad_b = grad_output.transpose(1, 2) @ act_for_b
        return grad_input, grad_b, grad_comp_mat, grad_x


        
