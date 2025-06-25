from torch.nn.common_types import  _size_2_t
from typing import Optional, Union
from torch import Tensor
from torch.nn import functional as F
import torch.nn as nn
import torch
from torch.nn.modules.conv import _ConvNd

def _pair(x):
    """
    transfer input as a two-tuples
    
    can also import by using
        'from torch.nn.modules.utils import _pair'
    
    we implement it by hand in case the function changes further
    """
    if isinstance(x, tuple):
        return x
    return (x, x)

class SubScafConv2d(_ConvNd):
    def __init__(
        self,
        comp_dim,
        comp_mat,
        wraped_model,
    ) -> None:
        factory_kwargs = {"device": wraped_model.weight.device, "dtype": wraped_model.weight.dtype}
        kernel_size_ = _pair(wraped_model.kernel_size)
        stride_ = _pair(wraped_model.stride)
        padding_ = wraped_model.padding if isinstance(wraped_model.padding, str) else _pair(wraped_model.padding)
        dilation_ = _pair(wraped_model.dilation)
        super().__init__(
            wraped_model.in_channels,
            wraped_model.out_channels,
            kernel_size_,
            stride_,
            padding_,
            dilation_,
            False,
            _pair(0),
            wraped_model.groups,
            wraped_model.bias,
            wraped_model.padding_mode,
            **factory_kwargs,
        )
        # replace weight 
        del self.weight
        self.comp_dim = comp_dim
        self.comp_mat = comp_mat
        self.x = wraped_model.weight.detach().clone()
        self.b = nn.Parameter(torch.zeros((self.out_channels, self.in_channels // self.groups, kernel_size_[0], comp_dim), **factory_kwargs))
        del wraped_model
    
    @property
    def weight(self):
        return self.x + self.b @ self.comp_mat
    
    
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
    
    def extra_repr(self):
        return f"{self.out_channels}, {self.in_channels}, kernel_size=(7, {self.comp_dim}), stride={self.stride}, padding={self.padding}, bias={self.bias}"
        

    def _conv_forward(self, input: Tensor, weight: Tensor, bias: Optional[Tensor]):
        if self.padding_mode != "zeros":
            return F.conv2d(
                F.pad(
                    input, self._reversed_padding_repeated_twice, mode=self.padding_mode
                ),
                weight,
                bias,
                self.stride,
                _pair(0),
                self.dilation,
                self.groups,
            )
        return F.conv2d(
            input, weight, bias, self.stride, self.padding, self.dilation, self.groups
        )

    def forward(self, input: Tensor) -> Tensor:
        return self._conv_forward(input, self.weight, self.bias)