import sys
import math
import torch
import torch.nn.functional as F


def MeanDice(output, target):
    num = (output==target).sum()
    denom = torch.numel(target) + torch.numel(output)
    dice = 2 * num/denom
    
    return dice


def LabelDice(output, target, label_list:list[int]=None):
    labels = target.unique.numpy() if label_list is None else label_list
    dice = [None] * labels.size
    
    for i in range(labels):
        num = ((output==labels[i]) * (target==labels[i])).sum()
        denom = (output==labels[i]).sum() + (target==labels[i]).sum()
        dice[i] = 2 * num / denom

    return dice



def MeanLabelDice(output, target, label_list:list[int]=None, background_label:int=None):
    labels = target.unique.numpy() if label_list is None else label_list
    labels = labels if background_label is None else [label for label in labels if label != background_label]
    dice = np.mean(LabelDice(output, target, label_list=labels))

    return dice
