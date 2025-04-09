import sys
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import interpol
from scipy.ndimage import distance_transform_edt

import utils


def mean_cce(output, cls, cls_all, weight:float=1., reduction:str='none'):
    target = torch.tensor(
        np.array(cls_all) == cls, dtype=torch.float, device=output.device
    ).view(output.shape)
    loss = (-target * output.log_softmax(dim=1)).sum(dim=1) / target.sum(dim=1)

    return weight * loss


def mean_dice_across_classes(
        output, target, weight:float=1., eps:float=1e-8,
        compute_softmax:bool=True, exclude_background:bool=False
):
    nB, nC = target.shape[:2]
    idx = 1 if exclude_background else 0
    output = F.softmax(output, dim=1) if compute_softmax else output
    output = output.view(nB, nC, -1)[:, idx:, ...]
    target = target.view(nB, nC, -1)[:, idx:, ...]

    loss = 1 - (
        (2 * output * target).sum(dim=-1) / (output + target + eps).sum(dim=-1)
    ).mean(dim=1)
    return weight * loss


def mean_dice_across_classes_diff(
        output, target, weight:float=1., eps:float=1e-8, dim:int=2,
        compute_softmax:bool=True, exclude_background:bool=False
):
    nB, nC = target.shape[:2]
    idx = 1 if exclude_background else 0
    output = (
        F.softmax(output, dim=1) if compute_softmax else output
    ).diff(dim=dim).abs().view(nB, nC, -1)[:, idx:, ...]
    target = target.diff(dim=dim).abs().view(nB, nC, -1)[:, idx:, ...]

    loss = 1 - (
        (2 * output * target).sum(dim=-1) / (output + target + eps).sum(dim=-1)
    ).mean(dim=1)
    return weight * loss



def mean_mse_across_classes_diff(
        output, target, weight=1, factor:float=1, dim:int=2, eps:float=1e-8,
        compute_softmax:bool=False, exclude_background=False
):
    nB, nC = target.shape[:2]
    idx = 1 if exclude_background else 0

    output = utils.min_max_norm(
        (output.softmax(dim=1) if compute_softmax else output
        ).diff(dim=dim).view(nB, nC, -1)[:, idx:], -factor, factor
    )
    target = target.diff(dim=dim).view(nB, nC, -1)[:, idx:]

    loss = (
        ((output - target) ** 2).sum(dim=-1) / (target.sum(dim=-1) + eps)
    ).mean(dim=1).abs()
    return weight * loss


def mse_change(output, target, dim:int=2, compute_softmax:bool=True):
    loss = (output - target) ** 2
    return torch.mean(loss.flatten())



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

    
def haus_dist(outputs, targets, p=8, **kwargs):
    return surf_dist(outputs, targets, p, **kwargs)
