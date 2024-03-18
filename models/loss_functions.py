import sys
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
#import interpol

#from scipy.ndimage import distance_transform_edt


def cce_loss(output, target, weight=1, compute_softmax=True, **kwargs):
    log_output = torch.log_softmax(output, dim=1) if compute_softmax else torch.log(output)
    loss = -1 * (log_output * target).sum() / target.sum()
    return weight * loss


def dice_loss(output, target, weight=1, start_idx=0, compute_softmax=True, exclude_background=False, **kwargs):
    start_idx = 1 if exclude_background else 0
    output = torch.flatten(F.softmax(output[:, start_idx:, ...], dim=1) if compute_softmax \
                           else output[:, start_idx:, ...])
    target = torch.flatten(target[:, start_idx:, ...])
    numer = 2 * torch.sum(output * target) + 1e-5
    denom = torch.sum(output + target) + 1e-5

    loss = 1 - (numer/denom)
    return weight * loss


def mean_dice_loss(output, target, weight=1, compute_softmax=True, exclude_background=False, **kwargs):
    start_idx = 1 if exclude_background else 0
    output = F.softmax(output, dim=1) if compute_softmax else output
    numer = torch.stack([2 * torch.sum(output[:,i,...] * target[:,i,...]) \
                         for i in range(start_idx, target.shape[1])])
    denom = torch.stack([torch.sum(output[:,i,...] + target[:,i,...]) + 1e-8 \
                         for i in range(start_idx, target.shape[1])])
    loss = 1 - torch.mean(numer/denom)
    return weight * loss


def mse_loss(output, target, weight=1, compute_softmax=True, exclude_background=True, **kwargs):
    start_idx = 1 if exclude_background else 0
    output = F.softmax(output, dim=1) if compute_softmax else output
    loss = ((output[:,start_idx:,...] - target[:,start_idx:,...]) ** 2).sum() / target[:,start_idx:,...].sum()

    return weight * loss


#def mean_mse_loss(output, target, weight=1, compute_softmax=True, exclude_background=True, **kwargs):


def mse_loss_logits(output, target, weight=1, rescale_factor=5, exclude_background=True, **kwargs):
    start_idx = 1 if exclude_background else 0
    target_rescale = rescale_factor * (2 * target - 1)
    loss = ((output[:,start_idx:,...] - target_rescale[:,start_idx:,...]) ** 2).sum() / target[:,start_idx:,...].sum()
    
    return weight * loss


def mean_mse_loss_logits(output, target, weight=1, rescale_factor=1, exclude_background=False, **kwargs):
    start_idx = 1 if exclude_background else 0
    output_rescale = rescale_factor * (2 * (output - output.min())/ (output.max() - output.min()) - 1)
    target_rescale = rescale_factor * (2 * target - 1)
    mse = torch.stack([((output_rescale[:,i,...] - target_rescale[:,i,...]) ** 2).sum() / \
                       target[:,i,...].sum() for i in range(start_idx, target.shape[1])])
    loss = torch.mean(mse)
    
    return weight * loss
