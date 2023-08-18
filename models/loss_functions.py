import sys
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import monai
import interpol
# from .denoisers import TV2d
from scipy.ndimage import histogram, distance_transform_edt

def nll_loss(outputs, targets, weights=1, reduction='mean', **kwargs):
    breakpoint()
    weights = torch.as_tensor(weights, device=outputs.device)
    numer = torch.sum(weights * -(torch.xlogy(targets, outputs)), keepdim=True, axis=1)
    denom = torch.sum(weights * (targets), keepdim=True, axis=1)

    if reduction == 'none':
        return numer
    
    return torch.sum(numer) / torch.sum(denom)

def cce_loss(outputs, targets, weights=1, reduction='mean', **kwargs):
    weights = torch.as_tensor(weights, device=outputs.device)
    numer = torch.sum(weights * -(targets * torch.log_softmax(outputs, dim=1)), keepdim=True, axis=1)
    denom = torch.sum(weights * (targets), keepdim=True, axis=1)

    if reduction == 'none':
        return numer
    
    return torch.sum(numer) / torch.sum(denom)

def dce_loss(outputs, targets, weights=1, reduction='mean', **kwargs):
    outputs = F.log_softmax(outputs, dim=1)
    axes = [0] + list(range(2, outputs.ndim))
    numer = torch.sum(targets * -outputs, keepdim=True, axis=axes)
    denom = torch.sum(targets, keepdim=True, axis=axes)

    return torch.mean(numer / denom)

def mse_loss(outputs, targets, weights=1, reduction='mean', **kwargs):
    weights = torch.as_tensor(weights, device=outputs.device)
    outputs = F.softmax(outputs, dim=1)
    numer = torch.sum(weights * (targets - outputs) ** 2, keepdim=True, axis=1)
    denom = torch.sum(weights * (targets), keepdim=True, axis=1)

    return torch.sum(numer) / torch.sum(denom)

def dice_nll_loss(outputs, targets, **kwargs):
    return dice_loss_safe(outputs, targets, start_idx=1, softmax=False) + nll_loss(outputs, targets)

def dice_cce_loss(outputs, targets, softmax=True, **kwargs):
    return dice_loss_safe(outputs, targets, start_idx=1) + cce_loss(outputs, targets)

def dice_loss(outputs, targets, weights=1, start_idx=0, softmax=True, **kwargs):
    weights = torch.as_tensor(weights, device=outputs.device)
    outputs = F.softmax(outputs, dim=1) if softmax else outputs
    axes = [0] + list(range(2, outputs.ndim))
    numer = torch.sum(weights * (outputs * targets * 2), axis=axes) + 1e-5
    denom = torch.sum(weights * (outputs * outputs + targets * targets), axis=axes) + 1e-5

    return torch.mean(1 - numer[start_idx:] / (denom[start_idx:] + 1e-8))

def dice_loss_safe(outputs, targets, weights=1, start_idx=1, softmax=True, **kwargs):
    weights = torch.as_tensor(weights, device=outputs.device)
    outputs = F.softmax(outputs, dim=1) if softmax else outputs
    axes = [0] + list(range(2, outputs.ndim))
    numer = 2 * torch.sum(weights * (outputs * targets), axis=axes) + 1e-5
    denom = 1 * torch.sum(weights * (outputs + targets), axis=axes) + 1e-5

    return torch.mean(1 - numer[start_idx:] / (denom[start_idx:] + 1e-8))

def dice_coef(outputs, targets, weights=1, reduction='mean', **kwargs):
    outputs = torch.as_tensor(outputs == torch.max(outputs, keepdim=True, axis=1)[0], dtype=torch.float, device=outputs.device)  #outputs.argmax(dim=1, keepdim=True) #  F.softmax(outputs, dim=1)
    axes = [0] + list(range(2, outputs.ndim))
    numer = torch.sum(outputs * targets, axis=axes)
    denom = torch.sum(outputs * outputs + targets * targets, axis=axes)
    ratio = 2 * numer / (denom + 1)

    if reduction=='none':
        return ratio

    return torch.mean(ratio)

def jacc_loss(outputs, targets, weights=1, ignore_index=255, reduction='mean', **kwargs):
    weights = torch.as_tensor(weights, device=outputs.device)
    outputs = F.softmax(outputs, dim=1)
    axes = [0] + list(range(2, outputs.ndim))
    mask = torch.sum(targets, keepdim=True, axis=1) > 0
    numer = torch.sum(mask * weights * (outputs * targets), axis=axes)
    denom = torch.sum(mask * weights * (outputs * outputs + targets * targets - outputs * targets), axis=axes)
    ratio = numer / denom

    return -torch.mean(ratio[denom > 0])

def jacc_coef(outputs, targets, weights=1, ignore_index=255, reduction='mean', **kwargs):
    weights = torch.as_tensor(weights, device=outputs.device)
    outputs = torch.as_tensor(outputs == torch.max(outputs, keepdim=True, axis=1)[0], dtype=torch.float, device=outputs.device)  #outputs.argmax(dim=1, keepdim=True) #  F.softmax(outputs, dim=1)
    axes = [0] + list(range(2, outputs.ndim))
    mask = torch.sum(targets, keepdim=True, axis=1)
    numer = torch.sum(mask * (outputs * targets), axis=axes)
    denom = torch.sum(mask * (outputs * outputs + targets * targets - outputs * targets), axis=axes)
    ratio = numer / denom.clamp(min=1.0)

    if reduction=='none':
        return ratio

    return torch.mean(ratio) #ratio) #[denom > 0])

def dice_coef_class(outputs, targets, weights=1, ignore_index=255, **kwargs):
    return dice_coef(outputs, targets, weights, ignore_index, reduction='none')

def surf_dist(outputs, targets, p=2, reduction='sum', **kwargs):
    count = torch.zeros(outputs.shape[1], device=outputs.device)
    sumsq = torch.zeros(outputs.shape[1], device=outputs.device)

    outputs = outputs.argmax(dim=1).long().cpu()
    targets = targets.argmax(dim=1).long().cpu()

    for i in range(count.shape[0]):
        h_target = torch.as_tensor(distance_transform_edt(targets != i) - 0) \
                 - torch.as_tensor(distance_transform_edt(targets == i) - 1)

        h_output = torch.as_tensor(distance_transform_edt(outputs != i) - 0) \
                 - torch.as_tensor(distance_transform_edt(outputs == i) - 1)

        count[i] = count[i] + (h_output == 0).sum()
        sumsq[i] = sumsq[i] + (h_target[h_output == 0].abs() ** p).sum()

        count[i] = count[i] + (h_target == 0).sum()
        sumsq[i] = sumsq[i] + (h_output[h_target == 0].abs() ** p).sum()

    if reduction=='none':
        return (sumsq / (count + 1e-10)) ** (1/p)

    if reduction=='mean':
        return (sumsq / (count + 1e-10)).mean() ** (1/p)

    if reduction=='sum':
        return (sumsq.sum() / count.sum()) ** (1/p)

def surf_dist_class(outputs, targets, p=2, **kwargs):
    return surf_dist(outputs, targets, p, reduction='none', **kwargs)

def haus_dist(outputs, targets, p=8, **kwargs):
    return surf_dist(outputs, targets, p, **kwargs)
