import os
import time
import pathlib as Path
import numpy as np
import surfa as sf
import random
import yaml

import torch
import torch.optim
import torch.nn as nn
import torch.nn.functional as F

import utils

#------------------------------------------------------------------------------

class SynthLongitudinal(nn.Module):
    def __init__(self,
                 sdict:dict,                  # dict of labels to atrophy
                 IOdict:dict,                 # dict of in->out label mappings
                 in_shape:list[int],           # no. expected shape of input
                 bg_labels:list[int]=0,       # background label val
                 do_insert_lsns:bool=False,   # flag to insert lesions
                 do_resize_labels:bool=True,  # flag to do synthetic atrophy
                 lsns_in_label:list[int]=2,   # labels to insert lesions
                 lsns_label:int=77,           # lesion label val
                 max_n_lsns:int=0,            # max no. lesions to insert
                 n_dims:int=3,                # no. image dimesions (2 or 3)
                 n_tps:int=2,                 # no. timepoints to synthesize
                 subsample:float=1.0,         # % to subsamle atrophy
                 T:int=2,                     # index of temporal dim.
                 device=None,
    ):
        super(SynthLongitudinal, self).__init__()
        
        # Parse class attributes
        self.device = 'cpu' if device is None else device
        self.n_timepoints = n_tps
        self.n_dims = n_dims

        max_n_lsns = max_n_lsns if do_insert_lsns else 0
        sdict = sdict if do_resize_labels else None
        
        # Check inputs
        in_shape = ((int(in_shape),) * n_dims
                    if isinstance(in_shape, int) or isinstance(in_shape, float)
                    else in_shape
        )
        while len(in_shape) < n_dims + 2:
            in_shape = (1,) + tuple(in_shape)

        # Initialize models for each timepoint
        self.synth_atrophy = nn.Module()        
        for t in range(n_tps):
            add_lesions = _AddLesions(
                in_shape=in_shape, lesions_label=lsns_label,
                max_n_lesions=max_n_lsns, bg_labels=lsns_in,
                device=device
            ) if max_n_lsns > 0 else nn.Identity()

            resize_labels = _ResizeLabels(
                sdict=sdict, subsample=subsample_atrophy,
                in_shape=in_shape, device=device
            ) if sdict is not None and t > 0 else nn.Identity()

            labels_to_image = _LabelsToImage(
                IOdict=IOdict, in_shape=in_shape,
                bg_labels=bg_labels, device=device
            )
            
            model = nn.Module()
            model.add_module('AddLesions', add_lesions)
            model.add_module('ResizeLabels', resize_labels)
            model.add_module('LabelsToImage', labels_to_image)
            self.synth_atrophy.add_module(f'SynthModel{t}', model)
            

    def forward(self, y_in):
        X = [None] * self.n_timepoints
        y = [None] * self.n_timepoints

        for t in range(self.n_timepoints):
            model = self.synth_atrophy.__getattr__(f'SynthModel{t}')
            y[t] = model.AddLesions(y_in if t == 0 else y[t-1].clone())
            y[t] = model.ResizeLabels(y[t])
            X[t] = model.LabelsToImage(y[t])

        X = torch.stack(X, dim=-(self.n_dims+1))
        y = torch.stack(y, dim=-(self.n_dims+1))
        return X, y

    
#-----------------------------------------------------------------------------

class _AddLesions(nn.Module):
    """
    Class to add a random number of lesions (up to max_n_lesions) within a 
    label image. Lesions will have a maximum volume of max_lesions_vol.
    """
    def __init__(self, 
                 in_shape=None,
                 lesions_label:int=77,        # value for lesion label
                 chance:float=0.5,            # prob. of inserting lesions
                 max_n_lesions:int=1,         # max no. of lesions
                 bg_labels:list[int]=[2,41],  # labels to insert lesions
                 bg_buffer:int=3,             # size of buffer (no. voxels)
                 max_lesions_vol:float=20.0,  # max lesion volume
                 n_dims:int=3,                # no. image dims
                 device=None
    ):
        super(_AddLesions, self).__init__()
        self.device = 'cpu' if device is None else device
        
        self.lesions_label = lesions_label
        self.chance = chance
        self.max_n_lesions = max_n_lesions
        self.bg_buffer = bg_buffer
        self.bg_labels = bg_labels
        self.max_lesions_vol = max_lesions_vol
        self.n_dims = n_dims

        if in_shape is None:
            utils.fatal('Must provide input shape to _AddLesions')
        
        # Set up morphology operations
        structuring_element = torch.zeros((3, 3, 3), dtype=float)
        structuring_element[torch.tensor([1,1,2,0,1,1,1]),
                                 torch.tensor([1,1,1,1,1,0,2]),
                                 torch.tensor([1,2,1,1,0,1,1])] = 1.

        self.dilation_conv = init_convolution(
            in_shape=in_shape, out_shape=in_shape,
            conv_weight_data=structuring_element,
            kernel_size=3, stride=1, padding=1, dilation=1,
            bias=False, device=self.device,
            requires_grad=False
        )


    def _insert_lesions(self, x):
        """
        Insert lesions within a volume
        """
        # Get mask of background labels
        M = torch.zeros(x.shape, device=x.device).to(float)
        for bg in self.bg_labels:
            M[x==bg] = 1.
        bg_idxs = torch.stack(torch.where(M > 0.))
        
        # Initialize up to max_n_lesions lesions (1 vox in size)        
        n_lesions = random.randint(1, self.max_n_lesions)
        M[tuple(bg_idxs[:,torch.randint(0, int(M.sum()), size=(n_lesions,))]
        )] = torch.arange(2, n_lesions+2).to(float).to(M.device)
        
        # Create sdict and grow lesions w/ _ResizeLabels
        lesions_sdict = {}
        for n in range(n_lesions):
            lesions_sdict[n+2] = [random.uniform(0., self.max_lesions_vol), 1]
        M = _ResizeLabels(
            lesions_sdict, x.shape, device=self.device,
            apply_dropout=True, randomize=False,
        )(M)
        
        # Insert back into original volume
        x = torch.where(M > 1, self.lesions_label, x)
        return x

    
    def forward(self, x):
        """
        Calls function w/ probability of self.chance
        """
        if random.uniform(0., 1.) <= self.chance:
            x = self._insert_lesions(x)
        return x
    
    
#-----------------------------------------------------------------------------

class _ResizeLabels(nn.Module):
    """
    Class to randomly resize labels (synthetic atrophy/growth) based in an 
    input dictionary of structures self.sdict. Has the ability to randomize 
    the amount of atrophy and number of structures w/in the input dict on the 
    fly.
    """
    def __init__(self,
                 sdict:dict={},           # dict of structures to atrophy
                 in_shape=None,           # dims of input volume
                 subsample:float=1.0,     # amt to subsample image
                 apply_dropout=False,     # flag to do dropout (dil. masks)
                 dropout_rate:float=0.2,  # dropout rate for dil. masks
                 randomize:bool=True,     # flag to randomize
                 device=None,
    ):
        super(_ResizeLabels, self).__init__()
        self.device = 'cpu' if device is None else device

        # Parse input params
        self.sdict = sdict
        self.subsample = subsample
        self.n_channels = in_shape[1]
        self.dropout_rate = dropout_rate if apply_dropout else 0.
        self.randomize = randomize
        
        # Set up dilation convolutions
        structuring_element = torch.zeros((3, 3, 3), dtype=float)
        structuring_element[torch.tensor([1,1,2,0,1,1,1]),
                            torch.tensor([1,1,1,1,1,0,2]),
                            torch.tensor([1,2,1,1,0,1,1])] = 1.        
        self.dilation_conv = init_convolution(
            in_shape=in_shape, out_shape=in_shape,
            conv_weight_data=structuring_element,
            kernel_size=3, stride=1, padding=1, dilation=1,
            bias=False, device=self.device,
            requires_grad=False
        )

    def _apply_dropout_to_mask(self, x, dr):
        """
        Applies a dropout filter to an input mask (adds variability to the 
        label boundaries during dilation/erosion)
        """
        m = torch.zeros(x.shape, dtype=x.dtype, device=x.device)
        m[x.nonzero(as_tuple=True)] = torch.rand(
            (int(x.sum())), dtype=x.dtype, device=x.device
        )
        return torch.where(m >= dr, x, 0)
        
        
    def _is_adjacent(self, x, label1, label2):
        """
        Determines whether or not two labels in the same volume are adjacent
        """
        mask1 = dilate_binary_mask(
            (x == label1).float(), self.dilation_conv)
        mask2 = dilate_binary_mask(
            (x == label2).float(), self.dilation_conv)
        return True if (mask1 * mask2).sum() > 0 else False


    def _shift_label_boundary(self, mask1, mask2, max_vol_change:float=None):
        """
        Moves the boundary between two labels to induce a specific amount of 
        volumetric change (mask1 will grow, mask2 will shrink)
        """
        dil = dilate_binary_mask(mask1, self.dilation_conv) * mask2

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
    

    def _resize_labels(self, x, sdict):
        """
        Main function to resize the labels specified in sdict
        """
        it = 0
        for label in sdict:
            # Create mask of specified adjacent neighbor labels
            nbr_list = sdict[label][1:]
            nbr_mask = torch.zeros(x.shape, device=x.device).float()

            for idx, neighbor in enumerate(nbr_list):
                if not self._is_adjacent(x, label, neighbor):
                    nbr_list.pop(idx)
            
            # Resize target
            trg_mask = (x == label).float()
            trg_mask_orig = trg_mask.clone()

            target_vol_change = sdict[label][0]
            vol_change = 0.
            
            while abs(vol_change) < abs(target_vol_change):
                change_remaining = target_vol_change - vol_change

                for neighbor in nbr_list:
                    nbr_mask = (x == neighbor).float()
                    if target_vol_change < 0:
                        nbr_mask, trg_mask = self._shift_label_boundary(
                            nbr_mask, trg_mask, max_vol_change=change_remaining
                        )
                        x = torch.where(nbr_mask.bool(), neighbor, x)
                    else:
                        trg_mask, nbr_mask = \
                            self._shift_label_boundary(
                                trg_mask, nbr_mask,
                                max_vol_change=change_remaining
                            )
                        x = torch.where(trg_mask.bool(), label, x)
                        
                vol_change = ((trg_mask.sum() - trg_mask_orig.sum()) \
                              / trg_mask_orig.sum()).item()
                
        return x


    def _randomize_sdict(self):
        """
        Randomize the no. structures and maximum amount of atrophy included in
        the sdict (allows for randomization beyond class initialization)
        """
        # Get idxs of structures to atrophy
        n_atrophy = random.randint(0, len(self.sdict))
        idxs_all = torch.arange(0, len(self.sdict))
        idxs = torch.multinomial(idxs_all.to(torch.float), n_atrophy)

        # Randomize each amount to be atrophied
        sdict = {}
        for n, label in enumerate(self.sdict.keys()):
            max_perc = self.sdict[label][0]
            perc = random.uniform(max_perc, 0.) if max_perc < 0 \
                else random.uniform(0., max_perc)
            sdict[label] = [perc] + self.sdict[label][1:]        

        return sdict


    def forward(self, x):
        sdict = self._randomize_sdict() if self.randomize else self.sdict
        x = self._resize_labels(x, sdict)
        return x


    
class _LabelsToImage(nn.Module):
    """
    Synthesize an intensity image from a label map
    """
    def __init__(self,
                 IOdict,
                 in_shape:list[int],
                 bg_labels:list[int]=None,
                 max_intensity:int=255,
                 blur_std:float=0.2,
                 noise_std=0.15,
                 device=None
    ):
        super(_LabelsToImage, self).__init__()
        self.device = 'cpu' if device is None else device

        # Parse args
        self.IOdict = IOdict
        self.bg_labels = [bg_labels] if not isinstance(bg_labels, list) \
            else bg_labels
        self.max_intensity = max_intensity
        self.noise_std = noise_std

        ## Set up Gaussian blur
        kernel = gaussian_kernel(blur_std)
        self.gaussian_blur = init_convolution(
            in_shape=in_shape, out_shape=in_shape,
            conv_weight_data=kernel, kernel_size=kernel.shape,
            padding=[(k-1)//2 for k in kernel.shape],
            stride=1, dilation=1, bias=False,
            device=self.device, requires_grad=False,
        )

        
    def _convert_labels(self, x):
        """
        Given the dictionary self.IOdict, converts voxels equal to a key to
        its corresponding value
        """
        y = x.clone()
        for label in self.IOdict:
            if self.IOdict[label] != label:
                y[x==label] = self.IOdict[label]
        return y


    def _add_noise(self, x):
        """
        Add Gaussian white noise to input image
        """
        sigma = random.uniform(0., self.noise_std)
        noise = torch.normal(mean=0., std=sigma, size=tuple(x.shape),
                             dtype=x.dtype, device=x.device)
        return x + noise


    def _synthesize_intensities(self, x):
        """
        Create synthetic grayscale image from input label map
        """
        # Get list of foreground labels and intensity means
        labels = list(set(x.unique().tolist()) ^ set(self.bg_labels))
        means = torch.rand(len(labels)) * self.max_intensity

        # Fill y
        y = torch.zeros(x.shape, dtype=torch.float, device=x.device)
        for label, m in zip(labels, means):
            y[torch.where(x==label)] = m.item()
            
        return y

        
    def forward(self, y):
        X = self._convert_labels(y)
        X = self._synthesize_intensities(X)
        X = self._add_noise(X)
        X = self.gaussian_blur(X)
        X = utils._min_max_norm(X, m=0, M=255)
        return X

    
#------------------------------------------------------------------------------

def _config_synth_models(config:dict, include_lesions=True, device=None):
    """
    Configures all synth models given an input config dict
    """    
    # Load label lookup table
    if not 'lut' in config:
        utils.fatal('Error: must include path to lut in input config')
    lut = sf.load_label_lookup(config['lut'])

    # Get dict of corresponding right/left label values (d = {right: left})
    lr_dict = {}
    for Rkey, Rval in lut.items():
        if 'Right' in Rval.name:
            lr_dict[Rkey] = search_lut(
                lut, '-'.join(['Left'] + Rval.name.split('-')[1:])
            )[0]
    
    # Get label dictionaries
    synth_slists_config = yaml.safe_load(open(config['slist_config']))
    synth_slists = {}

    for _class, sdict in synth_slists_config.items():
        synth_slists[_class] = {}
        for label, slist in sdict.items():
            key = search_lut(lut, label)[0]
            synth_slists[_class][key] = [config['max_perc_atrophy']] + [
                search_lut(lut, name)[0] for name in slist
            ]

    # Build models
    synth_models = {}
    for _class, sdict in synth_slists.items():
        synth_models[_class] = SynthLongitudinal(
            sdict=sdict, IOdict=lr_dict, device=device,
            **config['model_config']
        )

    return synth_models
    


def dilate_binary_mask(x, conv, n:int=1, dtype=None):
    dtype = x.type() if dtype is None else dtype
    if n > 0:
        for i in range(n):
            x = conv(x)
        return torch.where(x > 0., 1., 0.).type(dtype)
    else:
        return x


def erode_binary_mask(x, conv, n:int=1, dtype=None):
    dtype = x.type() if dtype is None else dtype
    if n > 0:
        x = torch.where(x > 0., 0., 1.).type(dtype)
        for i in range(n):
            x = conv(x)
        return torch.where(x > 0., 0., 1.)
    else:
        return x


def gaussian_kernel(sigma:float, ndims:int=3):
    window = np.round(sigma * 3) * 2 + 1
    center = (window - 1)/2

    mesh = [(-0.5 * pow(torch.arange(window) - center, 2))] * ndims
    mesh = torch.stack(torch.meshgrid(*mesh, indexing='ij'), dim=-1)
    kernel = (1 / pow(2 * torch.pi * sigma**2, 1.5)) \
        * torch.exp(-(pow(mesh, 2).sum(dim=-1)) / (2*sigma**2))

    return kernel / kernel.sum()


def init_convolution(in_shape, out_shape, conv_weight_data,
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

        conv_fn = eval('nn.Conv%dd' % ndims)(
            in_channels=in_channels, out_channels=out_channels,
            kernel_size=kernel_size, stride=stride, padding=padding,
            dilation=dilation, bias=False, device=device,
        )
    conv_fn.weight.data = conv_weight_data.to(torch.float).to(device)
    conv_fn.weight.requires_grad = requires_grad
    return conv_fn


def search_lut(lut, string:str):
    """
    Returns the key corresponding to the input string=value.name
    """
    return [key for key, val in lut.items() if val.name == string]

