import sys, math
import torch
import torch.nn.functional as F
import numpy as np
from scipy.ndimage import distance_transform_edt as edt



def Dice(output, target, exclude_background=False, eps=1e-5, **kwargs):
    start_idx = 1 if exclude_background else 0
    output = torch.flatten(F.softmax(output[:, start_idx:, ...], dim=1) if compute_softmax \
                           else output[:, start_idx:, ...])
    target = torch.flatten(target[:, start_idx:, ...])
    numer = 2 * torch.sum(output * target)
    denom = torch.sum(output + target)
    
    dice = (numer + eps) / (denom + eps)
    return dice



def MeanDice(output, target, exclude_background=False, eps=1e-5, **kwargs):
    start_idx = 1 if exclude_background else 0
    output = F.softmax(output, dim=1) if compute_softmax else output
    numer = torch.stack([2 * torch.sum(output[:,i,...] * target[:,i,...]) \
                         for i in range(start_idx, target.shape[1])])
    denom = torch.stack([torch.sum(output[:,i,...] + target[:,i,...]) + 1e-8 \
                         for i in range(start_idx, target.shape[1])])

    dice = torch.mean((numer + eps) / (denom + eps))
    breakpoint()
    return dice



def HausDist(y_pred, y, p=2, reduction='sum', **kwargs):
    count = torch.zeros(y_pred.shape[1], device=y_pred.device)
    sumsq = torch.zeros(y_pred.shape[1], device=y_pred.device)

    for i in range(count.shape[0]):
        h_y = torch.from_numpy(edt(y.cpu()!=i) - edt(y.cpu()==i) + 1).to(device=y_pred.device)
        h_y_pred = torch.from_numpy(edt(y_pred.cpu()!=i) - edt(y_pred.cpu()==i) + 1).to(device=y_pred.device)

        count[i] = count[i] + (h_y_pred==0).sum()
        sumsq[i] = sumsq[i] + (h_y[h_y_pred==0].abs() ** p).sum()

        count[i] = count[i] + (h_y==0).sum()
        sumsq[i] = sumsq[i] + (h_y_pred[h_y==0].abs() ** p).sum()

    if reduction=='none':
        return (sumsq / (count + 1e-10)) ** (1/p)

    if reduction=='mean':
        return (sumsq / (count + 1e-10)).mean() ** (1/p)

    if reduction=='sum':
        return (sumsq.sum() / count.sum()) ** (1/p)
