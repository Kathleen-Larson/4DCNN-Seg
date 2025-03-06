import os
import time
import pathlib as Path
import numpy as np
import surfa as sf
import random

import torch
import torch.optim
import torch.nn as nn
import torch.nn.functional as F

from models import utils

###################################################################################################
class SynthLongitudinal(nn.Module):
    def __init__(self,
                 labels_in,
                 input_shape:tuple[int],             # # expected shape of input
                 slist:dict=None,                    # dictionary containing labels and % atrophy
                 subsample_atrophy:float=1.0,        # % to subsamle atrophy
                 synth_image_seeds:int=0,            # seed for random image synthesis parameters
                 synth_image_bg_labels:list[int]=0,  # labels to set as background of synth image
                 include_lesions:bool=False,         # flag to turn on/off lesion insertion
                 lesion_label:int=77,                # int value for lesion label
                 max_n_lesions:int=0,                # max number of lesions to insert
                 lesion_in_labels:list[int]=[2, 41], # list of labels to insert lesions within
                 n_dims:int=3,                       # number of image dimesions (2 or 3)
                 n_timepoints:int=2,                 # number of timepoints to synthesize
                 T:int=2,                            # index of temporal dimension in input data
                 device=None
    ):
        super(SynthLongitudinal, self).__init__()
        self.synth_atrophy_model = nn.Module()
        self.device = 'cpu' if device is None else device
        
        # Parse args
        self.labels_in = labels_in
        self.slist = slist
        self.input_shape = input_shape
        self.subsample_atrophy = subsample_atrophy
        self.synth_image_seeds = synth_image_seeds
        self.synth_image_bg_labels = synth_image_bg_labels
        self.include_lesions = include_lesions
        self.lesion_label = lesion_label
        self.max_n_lesions = max_n_lesions
        self.lesion_in_labels = lesion_in_labels
        self.n_dims = n_dims
        self.n_timepoints = n_timepoints
        self.T = T

        # Generate models for each timepoint
        model_initial = self._init_timepoint_model(add_lesions=self.include_lesions,
                                                    resize_labels=False
        )
        self.synth_atrophy_model.add_module('SynthModel0', model_initial)

        for t in range(self.n_timepoints-1):
            model_atrophy = self._init_timepoint_model(add_lesions=self.include_lesions,
                                                       resize_labels=True,
            )
            self.synth_atrophy_model.add_module('SynthModel%d' % (t+1), model_atrophy)

        # Initialize new tensors w/ timepoint dimension
        new_shape = self.input_shape[:-self.n_dims] \
            + (self.n_timepoints+1,) + self.input_shape[-self.n_dims:]

        self.X_init = torch.zeros(new_shape, device=self.device, dtype=torch.float)
        self.y_init = torch.zeros(new_shape, device=self.device, dtype=torch.int)        
        
        
    def _init_timepoint_model(self, add_lesions:bool=True, resize_labels=True):
        timepoint_model = nn.Module()
        
        # Add lesions?
        if add_lesions and self.max_n_lesions > 0:
            timepoint_model.add_lesions = True
            add_lesions = _AddLesions(input_shape=self.input_shape,
                                      lesion_label=self.lesion_label,
                                      max_n_lesions=self.max_n_lesions,
                                      background_labels=self.lesion_in_labels,
                                      device=self.device
            )
            timepoint_model.add_module('AddLesions', add_lesions)
        else:
            timepoint_model.add_lesions = False
            
        # Induce synthetic atrophy
        if resize_labels and self.slist is not None:
            timepoint_model.resize_labels = True
            resize_labels = _ResizeLabels(slist=self.slist,
                                          subsample=self.subsample_atrophy,
                                          input_shape=self.input_shape,
                                          device=self.device,
            )
            timepoint_model.add_module('ResizeLabels', resize_labels)
        else:
            timepoint_model.resize_labels = False
            
        # Synthesize intensity images from label maps
        labels_to_image = _LabelsToImage(labels_in=self.labels_in,
                                         input_shape=self.input_shape,
                                         background_labels=self.synth_image_bg_labels,
                                         device=self.device
        )
        timepoint_model.add_module('LabelsToImage', labels_to_image)

        return timepoint_model


    def forward(self, y_in):
        X = [None] * self.n_timepoints #self.X_init.clone()
        y = [None] * self.n_timepoints #self.y_init.clone()

        for t in range(self.n_timepoints):
            synth_model = self.synth_atrophy_model.__getattr__('SynthModel%d' % t)
            y[t] = y_in if t == 0 else y[t-1].clone()

            if synth_model.add_lesions: y[t] = synth_model.AddLesions(y[t])
            if synth_model.resize_labels: y[t] = synth_model.ResizeLabels(y[t])
            X[t] = synth_model.LabelsToImage(y[t])

        return torch.stack(X, dim=self.T), torch.stack(y, dim=self.T)

    
        

class _AddLesions(nn.Module):
    def __init__(self, 
                 input_shape=None,
                 lesion_label:int=77,                # int value for lesion label
                 chance:float=0.5,                   # probability of inserting lesions
                 max_n_lesions:int=1,                # max number of lesions to insert
                 background_labels:list[int]=[2,41], # list of labels to insert lesions within
                 background_buffer:int=3,
                 max_lesion_vol:float=20.0,
                 device=None
    ):
        super(_AddLesions, self).__init__()
        self.device = 'cpu' if device is None else device
        
        self.lesion_label = lesion_label
        self.chance = chance
        self.max_n_lesions = max_n_lesions
        self.background_buffer = background_buffer
        self.background_labels = background_labels
        self.max_lesion_vol = max_lesion_vol

        assert input_shape is not None, "Must provide input shape to _AddLesions"
        
        # Set up morphology operations
        structuring_element = torch.zeros((3, 3, 3), dtype=float)
        structuring_element[torch.tensor([1,1,2,0,1,1,1]),
                                 torch.tensor([1,1,1,1,1,0,2]),
                                 torch.tensor([1,2,1,1,0,1,1])] = 1.

        self.dilation_conv = utils.InitializeConvolution(in_shape=input_shape,
                                                         out_shape=input_shape,
                                                         conv_weight_data=structuring_element,
                                                         kernel_size=3, stride=1, padding=1, dilation=1,
                                                         bias=False, device=self.device,
                                                         requires_grad=False
        )


    def _insert_lesions(self, x):
        # Create eroded labels masks
        n_bg_labels = len(self.background_labels)
        M_bg = [None] * n_bg_labels
        M_bg_idxs = [None] * n_bg_labels

        t_start = time.time()
        for n in range(n_bg_labels):
            M_bg[n] = self._make_label_mask(x, self.background_labels[n]).float()
            M_bg_idxs[n] = torch.where(M_bg[n] > 0.)
            
            
        # Lesion insertion
        n_lesions = 8 #random.randint(1, self.max_n_lesions)
        rand_idxs = torch.randint(n_bg_labels, size=(n_lesions,)).tolist()
        lesion_vols = (self.max_lesion_vol * torch.rand(size=(n_lesions,))).tolist()
        M_les = torch.zeros(x.shape, device=x.device, dtype=x.dtype)
        
        for n, idx in enumerate(rand_idxs):
            t0 = time.time()
            x_les = M_bg[idx].clone()
            lesion_slist = {2: [lesion_vols[n], 1]}
            
            # Introduce 1 vox sized lesion
            vox_idx = random.randint(0, int(M_bg[idx].sum())-1)
            idx = [idxs[vox_idx].item() for idxs in M_bg_idxs[idx]]
            x_les[..., idx[-3], idx[-2], idx[-1]] = 2

            # Dilate lesion and update
            grow_lesion = _ResizeLabels(lesion_slist, x.shape, apply_dropout=True)
            x_les = grow_lesion(x_les)
            M_les[x_les==2] = 1.

            t1 = time.time()
            #print(f'{(t1-t0):>.2f}s for volume={lesion_vols[n]:>.1f}')
            
        x[torch.where(M_les > 0.)] = self.lesion_label
        return x

    
    def _make_label_mask(self, x, labels=None):
        background_labels = self.background_labels if labels is None else labels
        mask = torch.zeros(x.shape, device=x.device, dtype=x.dtype)
        for n in self.background_labels:
            mask[x==n] = 1.
        return mask
    

    def forward(self, x):
        if random.uniform(0., 1.) <= self.chance:
            x = self._insert_lesions(x)
        return x
    
    

class _ResizeLabels(nn.Module):
    def __init__(self,
                 slist:dict={},              # dictionary of structures to atrophy
                 input_shape=None,           # dimensions of input volume
                 max_perc_atrophy:float=0.5, # upper bound for atrophy percent sampler 
                 subsample:float=1.0,        # amount to subsample image voxels
                 apply_dropout=False,        # flag to apply dropout filter to dilation masks
                 dropout_rate:float=0.2,     # dropout rate for dilation masks
                 device=None,
    ):
        super(_ResizeLabels, self).__init__()
        self.device = 'cpu' if device is None else device

        self.structure_list = slist
        self.subsample = subsample
        self.n_channels = input_shape[1]
        self.dropout_rate = dropout_rate if apply_dropout else 0.
        
        # Set up dilation
        structuring_element = torch.zeros((3, 3, 3), dtype=float)
        structuring_element[torch.tensor([1,1,2,0,1,1,1]),
                                 torch.tensor([1,1,1,1,1,0,2]),
                                 torch.tensor([1,2,1,1,0,1,1])] = 1.
        
        self.dilation_conv = utils.InitializeConvolution(in_shape=input_shape,
                                                         out_shape=input_shape,
                                                         conv_weight_data=structuring_element,
                                                         kernel_size=3, stride=1, padding=1, dilation=1,
                                                         bias=False, device=self.device,
                                                         requires_grad=False
        )


    def _apply_dropout_to_mask(self, x, dr):
        m = torch.zeros(x.shape, dtype=x.dtype, device=x.device)
        m[x.nonzero(as_tuple=True)] = torch.rand((int(x.sum())), dtype=x.dtype, device=x.device)
        return torch.where(m >= dr, x, 0)
        
        
    def _is_adjacent(self, x, label1, label2):
        mask1 = utils.DilateBinaryMask((x == label1).float(), self.dilation_conv)
        mask2 = utils.DilateBinaryMask((x == label2).float(), self.dilation_conv)
        return True if (mask1 * mask2).sum() > 0 else False


    def _shift_label_boundary(self, mask1, mask2, max_vol_change:float=None):
        dil = utils.DilateBinaryMask(mask1, self.dilation_conv) * mask2

        # Apply dropout if necessary
        if self.dropout_rate > 0.:
            dil = self._apply_dropout_to_mask(dil, self.dropout_rate)

        # Ensure dilation mask is not inducing too much change
        if max_vol_change is not None:
            vol_change = dil.sum()/mask2.sum()
            if vol_change > abs(max_vol_change):
                dr = (vol_change - abs(max_vol_change)) / vol_change
                dil = self._apply_dropout_to_mask(dil, dr)

        return mask1 + dil, mask2 - dil
    

    def _resize_labels(self, x, slist):
        it = 0
        for label in slist:
            # Create mask of specified adjacent neighbor labels
            nbr_list = slist[label][1:]
            nbr_mask = torch.zeros(x.shape, device=x.device).float()

            for idx, neighbor in enumerate(nbr_list):
                if not self._is_adjacent(x, label, neighbor):  nbr_list.pop(idx)
            
            # Resize target
            trg_mask = (x == label).float()
            trg_mask_orig = trg_mask.clone()

            target_vol_change = slist[label][0]
            vol_change = 0.
            
            while abs(vol_change) < abs(target_vol_change):
                change_remaining = target_vol_change - vol_change

                for neighbor in nbr_list:
                    nbr_mask = (x == neighbor).float()
                    if target_vol_change < 0:
                        nbr_mask, trg_mask = \
                            self._shift_label_boundary(nbr_mask, trg_mask,
                                                       max_vol_change=change_remaining)
                        x = torch.where(nbr_mask.bool(), neighbor, x)
                    else:
                        trg_mask, nbr_mask = \
                            self._shift_label_boundary(trg_mask, nbr_mask,
                                                       max_vol_change=change_remaining)
                        x = torch.where(trg_mask.bool(), label, x)
                        
                vol_change = ((trg_mask.sum() - trg_mask_orig.sum()) \
                              / trg_mask_orig.sum()).item()

        return x


    def _randomize_atrophy(self):
        n_atrophy = random.randint(2, len(self.structure_list))
        idxs_all = torch.arange(0, len(self.structure_list))
        idxs = torch.multinomial(idxs_all.to(torch.float), n_atrophy)

        slist = {}
        for n, label in enumerate(self.structure_list.keys()):
            max_perc = self.structure_list[label][0]
            perc = random.uniform(max_perc, 0.) if max_perc < 0 \
                else random.uniform(0., max_perc)
            slist[label] = [perc] + self.structure_list[label][1:]        

        return slist


    def forward(self, x):
        slist = self._randomize_atrophy()
        x = self._resize_labels(x, slist)
        return x


    
class _LabelsToImage(nn.Module):
    """
    Synthesize an intensity image from a label map
    """
    def __init__(self,
                 labels_in,
                 input_shape:list[int],
                 background_labels:list[int]=None,
                 max_intensity:int=255,
                 blur_std:float=0.5,
                 noise_std=0.15,
                 device=None
    ):
        super(_LabelsToImage, self).__init__()
        self.device = 'cpu' if device is None else device
        
        self.labels_in = labels_in
        self.background_labels = [background_labels] if not isinstance(background_labels, list) \
            else background_labels
        self.max_intensity = max_intensity
        self.noise_std = noise_std

        ## Set up Gaussian blur
        kernel = utils.GaussianKernel(blur_std)
        self.gaussian_blur = utils.InitializeConvolution(in_shape=input_shape,
                                                         out_shape=input_shape,
                                                         conv_weight_data=kernel,
                                                         kernel_size=kernel.shape,
                                                         padding=[(k-1)//2 for k in kernel.shape],
                                                         stride=1, dilation=1, bias=False,
                                                         device=self.device, requires_grad=False,
        )
        
        
    def _convert_labels(self, x):
        y = x.clone()
        for label in self.labels_in:
            if self.labels_in[label] != label: y[x==label] = self.labels_in[label]
        return y


    def _add_noise(self, x):
        sigma = random.uniform(0., self.noise_std)
        noise = torch.normal(mean=0., std=sigma, size=tuple(x.shape),
                             dtype=x.dtype, device=x.device)
        return x + noise


    def _min_max_norm(self, x, minim=0., maxim=255.):
        return (maxim - minim) * (x - x.min()) / (x.max() - x.min()) + minim
    
    

    def _synthesize_intensities(self, x):
        y = torch.zeros(x.shape, dtype=torch.float, device=x.device)
        labels = x.unique().tolist()
        for idx, val in enumerate(labels):
            if val in self.background_labels:  labels.pop(idx)

        means = torch.rand(len(labels)).to(y.dtype)
        for idx, label in enumerate(labels):
            inds = torch.where(x==label)
            y[inds] = means[idx]

        return y


        
    def forward(self, y):
        X = self._convert_labels(y)
        X = self._synthesize_intensities(X)
        X = self._add_noise(X)
        X = self.gaussian_blur(X)
        X = self._min_max_norm(X)
        return X




class _RemoveLabels(nn.Module):
    """
    Remove specified labels from an intensity map
    """
    def __init__(self,
                 labels_keep=None,
                 labels_remove=None,
    ):
        super(_RemoveLabels, self).__init__()
    
        if labels_keep is None and labels_remove is None:
            print('_RemoveLabels isnt going to do anything without inputs...')
            

    def forward(self, x):
        return x


        
