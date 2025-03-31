import sys
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import interpol
from scipy.ndimage import distance_transform_edt



## CCE ##

def cce_loss(output, target, weight=1, compute_softmax=True, **kwargs):
    log_output = torch.log_softmax(output, dim=1) if compute_softmax else torch.log(output)
    loss = -1 * (log_output * target).sum() / target.sum()
    return weight * loss



## Dice ##

def dice_loss(output, target, weight=1, compute_softmax=True, exclude_background=False, **kwargs):
    start_idx = 1 if exclude_background else 0
    output = torch.flatten(F.softmax(output[:, start_idx:, ...], dim=1) if compute_softmax \
                           else output[:, start_idx:, ...])
    target = torch.flatten(target[:, start_idx:, ...])
    numer = 2 * torch.sum(output * target) + 1e-5
    denom = torch.sum(output + target) + 1e-5

    loss = 1 - (numer/denom)
    return weight * loss


def dice_loss_yesbackground(output, target, weight=1, **kwargs):
    loss = dice_loss(output, target, weight, exclude_background=False)
    return loss


def dice_loss_nobackground(output, target, weight=1, **kwargs):
    loss = dice_loss(output, target, weight, exclude_background=True)
    return loss


def mean_dice_loss(output, target, weight=1, compute_softmax=True, exclude_background=False, **kwargs):
    start_idx = 1 if exclude_background else 0
    output = F.softmax(output, dim=1) if compute_softmax else output
    numer = torch.stack([2 * torch.sum(output[:,i,...] * target[:,i,...]) \
                         for i in range(start_idx, target.shape[1])])
    denom = torch.stack([torch.sum(output[:,i,...] + target[:,i,...]) + 1e-8 \
                         for i in range(start_idx, target.shape[1])])

    loss = 1 - torch.mean(numer[denom!=0]/denom[denom!=0])
    return weight * loss


def mean_dice_loss_yesbackground(output, target, weight=1, **kwargs):
    loss = mean_dice_loss(output, target, weight, exclude_background=False)
    return loss


def mean_dice_loss_nobackground(output, target, weight=1, **kwargs):
    loss = mean_dice_loss(output, target, weight, exclude_background=True)
    return loss



## MSE ##

def mse_loss(output, target, weight=1, compute_softmax=True, exclude_background=True, **kwargs):
    start_idx = 1 if exclude_background else 0
    output = F.softmax(output, dim=1) if compute_softmax else output
    loss = ((output[:,start_idx:,...] - target[:,start_idx:,...]) ** 2).sum() / target[:,start_idx:,...].sum()

    return weight * loss



def mse_loss_logits(output, target, weight=1, rescale_factor=15, exclude_background=True, **kwargs):
    start_idx = 1 if exclude_background else 0
    target_rescale = rescale_factor * (2 * target - 1)
    loss = ((output[:,start_idx:,...] - target_rescale[:,start_idx:,...]) ** 2).sum() / target[:,start_idx:,...].sum()

    return weight * loss


def mean_mse_loss_logits(output, target, weight=1, rescale_factor=15, exclude_background=False, **kwargs):
    start_idx = 1 if exclude_background else 0
    output_rescale = rescale_factor * (2 * (output - output.min())/ (output.max() - output.min()) - 1)
    target_rescale = rescale_factor * (2 * target - 1)

    numer = torch.stack([((output_rescale[:,i,...] - target_rescale[:,i,...]) ** 2).sum() \
                       for i in range(start_idx, target.shape[1])])
    denom = torch.stack([target[:,i,...].sum() for i in range(start_idx, target.shape[1])])

    loss = torch.mean(numer[denom!=0] / denom[denom!=0])
    return weight * loss


def mse_loss_logits_yesbackground(output, target, weight=1, **kwargs):
    loss = mse_loss_logits(output, target, weight, exclude_background=False)
    return loss

def mse_loss_logits_nobackground(output, target, weight=1, **kwargs):
    loss = mse_loss_logits(output, target, weight, exclude_background=True)
    return loss

def mean_mse_loss_logits_yesbackground(output, target, weight=1, **kwargs):
    loss = mean_mse_loss_logits(output, target, weight, exclude_background=False)
    return loss

def mean_mse_loss_logits_nobackground(output, target, weight=1, **kwargs):
    loss = mean_mse_loss_logits(output, target, weight, exclude_background=True)
    return loss



## Surface distances ##

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
