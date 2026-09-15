import sys
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import interpol
from scipy.ndimage import distance_transform_edt

import utils


# --------------------------------------------------------------------------------------------------
# Cross-entropy losses

def bce_loss(output, target, possible, weight=1., reduction='mean'):
    nB = output.shape[0]
    target = torch.tensor(
        [x == target for x in possible], dtype=torch.float, device=output.device
    ).view(output.shape)
    loss = nn.BCEWithLogitsLoss(reduction=reduction)(output, target)
    return weight * loss


def cce_loss(output, target, possible, weight=1., reduction='mean'):
    target = [target] if not isinstance(target, (list, tuple)) else target
    target = torch.tensor(
        [possible.index(x) for x in target], dtype=torch.long, device=output.device
    )
    loss = nn.CrossEntropyLoss(reduction=reduction)(output, target)
    return weight * loss


# --------------------------------------------------------------------------------------------------
# Dice losses

def mean_dice_loss(output, target, weight=1., eps=1e-5, compute_softmax=True, compute_diff=False,
                   exclude_background=False):
    nB, nC, nT = target.shape[:3]
    idx = 1 if exclude_background else 0

    output = output.softmax(dim=1) if compute_softmax else output
    output = output.reshape(nB, nC, nT, -1)[:, idx:, ...]
    output = output.diff(dim=2).abs() if compute_diff else output

    target = target.reshape(nB, nC, nT, -1)[:, idx:, ...].to(output.device)
    target = target.diff(dim=2).abs() if compute_diff else target

    numer = (2 * output * target).sum(dim=-1)
    denom = (output + target).sum(dim=-1)
    dice = (numer + eps) / (denom + eps)

    loss = (1 - dice).mean()
    return weight * loss


def mean_dice_diff_loss(output, target, weight=1., eps=1e-5, compute_softmax=True,
                        exclude_background=False):
    nB, nC, nT = target.shape[:3]
    idx = 1 if exclude_background else 0

    output = (
        F.softmax(output, dim=2) if compute_softmax else output
    ).diff(dim=2).abs().reshape(nB, nC, nT - 1, -1)[:, idx:, ...]
    target = target.diff(dim=2).abs().reshape(nB, nC, nT - 1, -1)[:, idx:, ...].to(output.device)

    numer = (2 * output * target).sum(dim=-1)
    denom = (output + target).sum(dim=-1)

    loss = (1 - (numer + eps) / (denom + eps)).mean()
    return weight * loss
