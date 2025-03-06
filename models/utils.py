import os
import numpy as np
import torch
import torch.nn as nn


def DilateBinaryMask(x, conv, n:int=1, dtype=None):
    dtype = x.type() if dtype is None else dtype
    if n > 0:
        for i in range(n):
            x = conv(x)
        return torch.where(x > 0., 1., 0.).type(dtype)
    else:
        return x
    

def ErodeBinaryMask(x, conv, n:int=1, dtype=None):
    dtype = x.type() if dtype is None else dtype
    if n > 0:
        x = torch.where(x > 0., 0., 1.).type(dtype)
        for i in range(n):
            x = conv(x)
        return torch.where(x > 0., 0., 1.)
    else:
        return x

    
def GaussianKernel(sigma:float, ndims:int=3):
    window = np.round(sigma * 3) * 2 + 1
    center = (window - 1)/2

    mesh = [(-0.5 * pow(torch.arange(window) - center, 2))] * ndims
    mesh = torch.stack(torch.meshgrid(*mesh, indexing='ij'), dim=-1)
    kernel = (1 / pow(2 * torch.pi * sigma**2, 1.5)) * \
        torch.exp(-(pow(mesh, 2).sum(dim=-1)) / (2*sigma**2))

    return kernel / kernel.sum()


def InitializeConvolution(in_shape, out_shape, conv_weight_data,
                          kernel_size:int=3, stride:int=1,
                          padding:int=1, dilation:int=1,
                          bias=False, ndims:int=3, device=None,
                          requires_grad:bool=False,
):
    in_channels = in_shape[1]
    out_channels = out_shape[1]
    device = 'cpu' if device is None else device
    
    while len(conv_weight_data.shape) < len(in_shape):
        conv_weight_data = conv_weight_data.unsqueeze(0)

    conv_fn = eval('nn.Conv%dd' % ndims)(in_channels=in_channels, out_channels=out_channels,
                                         kernel_size=kernel_size, stride=stride,
                                         padding=padding, dilation=dilation,
                                         bias=False, device=device,
    )
    conv_fn.weight.data = conv_weight_data.to(torch.float).to(device)
    conv_fn.weight.requires_grad = requires_grad
    return conv_fn


def UnsqueezeAndRepeat(x, unsqueeze_dims:list[int], repeats=None):
    unsqueeze_dims = [unsqueeze_dims] if not isinstance(unsqueeze_dims, list) else unsqueeze_dims
    if repeats is not None: assert len(repeats) == len(x.shape) + len(unsqueeze_dims), \
       "In utils.UnsqueezeAndRepeat(), len(repeats) must equal len(x.shape) + len(unsqueeze_dims)"
    for d in unsqueeze_dims: x.unsqueeze(d)
    return x if repeats is None else x.repeat(repeats)

