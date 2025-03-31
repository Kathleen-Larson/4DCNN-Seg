import sys
import numpy as np
from numpy import random as npr
import math
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import Compose

from utils import unsqueeze_repeat, fatal


#-----------------------------------------------------------------------------#
#                               Intensity funcs                               #
#-----------------------------------------------------------------------------#

class BiasField:
    """
    Simulates bias field in input tensor [y = x * exp(B)]
    """
    def __init__(self,
                 shape_factor:float=0.025,  # ratio of small field to img size
                 max_value:float=1.0,       # max value of bias field
                 std:float=0.3,             # std of bias field
                 randomize:bool=True,       # flag to randomize
                 X:int=3                    # number of spatial dims
    ):
        self.shape_factor = shape_factor
        self.std = std
        self.max_value = max_value
        self.randomize = randomize
        self.X = X
            
    def _apply_bias_field(self, x):
        sz_full = x.shape[-self.X:]
        nB, nC, nT = x.shape[:self.X]
        bf_full = torch.zeros(
            ((nB, nC, nT) + sz_full), dtype=x.dtype, device=x.device
        )
        
        for t in range(0, nT):
            sz_small = torch.ceil(
                torch.tensor(sz_full) * self.shape_factor
            ).to(torch.int)
            sz_small = (1, 1) + tuple(sz_small.tolist())
            bf_std = random.uniform(0., self.std) if self.randomize \
                else self.std
            
            bf_small = torch.normal(
                mean=0., std=bf_std, size=tuple(sz_small), device=x.device
            )
            bf_full[:, :, t, ...] = F.interpolate(
                bf_small, size=sz_full[-self.X:],
                mode='trilinear' if self.X == 3 else 'bilinear'
            )
        return x * torch.exp(bf_full)

        
    def __call__(self, inputs):
        inputs[0] = self._apply_bias_field(inputs[0])
        return inputs

    

class GammaTransform:
    """
    Gamma transform on input tensor [y = x ** exp(gamma)]
    """
    def __init__(self,
                 std:float=0.5,        # std of gamma value
                 randomize:bool=True,  # flag to randomize
    ):
        self.std = std
        self.randomize = randomize

        
    def _gamma_transform(self, x):
        gamma = torch.tensor(
            random.normal(0., self.std) if self.randomize else self.std
        ).to(x.device)
        return x.pow(torch.exp(gamma))

    
    def __call__(self, inputs):
        inputs[0] = self._gamma_transform(inputs[0])
        return inputs
    


class GaussianNoise:
    """
    Adds gaussian noise to input tensor [y = x + N]
    """
    def __init__(self,
                 std:float=21.0,       # std of gaussian noise
                 randomize:bool=True,  # flag to randomize
    ):
        self.std = std
        self.randomize = randomize

        
    def _noise(self, x):
        noise_std = random.uniform(0., self.std) if self.randomize \
            else self.std
        noise = torch.normal(mean=0., std=noise_std, size=tuple(x.size()))
        return x + noise.to(x.device)


    def __call__(self, inputs):
        inputs[0] = self._noise(inputs[0])
        return inputs

    

class MinMaxNorm:
    """
    Robust intensity and min-max normalization
    """
    def __init__(self,
                 min_int:float=0.,       # minimum intensity value
                 max_int:float=1.,       # maximum intensity value
                 min_perc:float=0.,      # minimum % to clip intensities
                 max_perc:float=0.95,    # maximum % to clip intensities
                 use_robust:bool=False,  # flag to use robust norm
                 X:int=3,                # no. spatial dims
    ):
        self.m = min_int
        self.M = max_int
        self.mperc = min_perc
        self.Mperc = max_perc
        self.use_robust = use_robust
        
        
    def _min_max_norm(self, x, m=None, M=None):
        m = self.m if m is None else m
        M = self.M if M is None else N
        return (M - m) * (x - x.min()) / (x.max() - x.min()) + m

    
    def _robust_norm(self, x):
        # Convert percentages to intensities
        full_sz = x.shape
        n_vox = np.prod([sz for sz in full_sz[:-self.X]])
        flat_sz = shape[:-self.X] + (torch.tensor(n_vox,))

        x_sorted, _ = x.reshape(flat_sz).sort()
        m = x_sorted[..., max(int(self.mperc * flat_sz[-1]), 0)][0]
        M = x_sorted[..., min(int(self.Mperc * flat_sz[-1]), flat_sz[-1]-1)][0]

        # Robust normalization
        _, nC, nT = full_sz[:-self.X]
        for c in range(nC):
            for t in range(nT):
                m, M = [min_val[c, t], max_val[c, t]]
                x[:, c, t, ...] = torch.clamp(
                    x[:, chn, ...], min=m[c, t], max=M[c, t]
                )
        return self._min_max_norm(x)
        
    
    def __call__(self, inputs):
        inputs[0] = self._robust_norm(inputs[0]) if self.use_robust \
             else self._min_max_norm(inputs[0])
        return inputs


#-----------------------------------------------------------------------------#
#                                Spatial funcs                                #
#-----------------------------------------------------------------------------#

class AffineElasticTransform:
    def __init__(self,
                 translations:[float,list]=0.0,  # translation bounds
                 rotations:[float,list]=15.0,    # rotation bounds (degrees)
                 shears:[float,list]=0.012,      # shearing bouinds
                 scales:[float,list]=0.15,       # scaling bounds
                 elastic_factor:float=0.0625,    # small SVF sz : full sz
                 elastic_std:float=3.,           # std of gaussian for SVF
                 n_elastic_steps:int=7,          # no. of integration steps
                 apply_affine:bool=True,         # perform affine trans
                 apply_elastic:bool=True,        # perform elastic trans
                 randomize:bool=True,            # randomize params?
                 X:int=3,                        # no. of spatial dims
    ):
        self.X = X
        self.apply_affine = apply_affine
        self.apply_elastic = apply_elastic

        # Parse affine transform parameters
        if self.apply_affine:
            def _parse_affine_param(param, center=0.):
                if isinstance(param, list):
                    if len(param) == 2:
                        param = [param] * self.X
                    elif len(param) == self.X:
                        param = [[-p, p] for p in param]
                    elif len(param) == self.X * 2:
                        param = param
                else:
                    param = [[-param if full else 0., param]] * X
                shift = torch.ones((self.X, 2), dtype=torch.float) * center
                return torch.tensor(param).reshape(self.X, 2) + shift

            self.translations = _parse_affine_param(translations)
            self.rotations = _parse_affine_param(rotations)
            self.shears = _parse_affine_param(shears, center=0.)
            self.zooms = _parse_affine_param(scales, center=1.)

        # Parse elastic transform parameters
        if self.apply_elastic:
            self.elastic_factor = elastic_factor
            self.elastic_std = elastic_std
            self.n_elastic_steps = n_elastic_steps


    def _make_affine(self, x, nT):
        def _sample_params(bounds, nT):
            bounds_range = torch.diff(bounds).squeeze()
            return torch.rand((nT, self.X)) * bounds_range + bounds[:,0]
        
        I = utils.unsqueeze_repeat(
            torch.eye(self.X+1, device=x.device),
            unsqueeze_dims=0, repeats=(nT, 1, 1)
        )

        # Translations
        T = I.clone()
        T[:, torch.arange(self.X), -1] = _sample_params(self.translations, nT)

        # Shears
        Sinds = torch.ones(
            (nT, self.X+1, self.X+1), dtype=torch.bool, device=x.device
        )
        Sinds[:, torch.eye(self.X+1, dtype=torch.bool)] = False
        Sinds[:, -1, :] = False
        Sinds[:, :, -1] = False
        S = I.clone()
        S[Sinds] = torch.cat([
            _sample_params(self.shears, nT),_sample_params(self.shears, nT)
        ], dim=-1).flatten()
        
        # Zooms
        Z = I.clone()
        Z[:, torch.arange(self.X),
          torch.arange(self.X)] = _sample_params(self.zooms, nT)
        
        # Rotations
        rotations = _sample_params(self.rotations, nT) * torch.pi/180
        c, s = [torch.cos(r), torch.sin(r)]
        [R1, R2, R3] = [I.clone(), I.clone(), I.clone()]
        if self.X == 2:
            R1[torch.tensor([0,1,0,1]),
              torch.tensor([0,0,1,1])] = torch.tensor([c, s, -s, c])
        else:
            R1[torch.tensor([1,2,1,2]),
               torch.tensor([1,1,2,2])] = torch.tensor(
                   [c[0], s[0], -s[0], c[0]]
               )
            R2[torch.tensor([0,2,0,2]),
               torch.tensor([0,0,2,2])] = torch.tensor(
                   [c[1], s[1], -s[1], c[1]]
               )
            R3[torch.tensor([0,1,0,1]),
               torch.tensor([0,0,1,1])] = torch.tensor(
                   [c[2], s[2], -s[2], c[2]]
               )
            
        ## Convert affine matrix to displacement field
        aff = (T @ R @ Z @ S).to(x.dtype).to(x.device)
        grid = torch.stack(
            torch.meshgrid(
                [torch.arange(-(s-1)/2, s/2, dtype=x.dtype, device=x.device)
                 for s in x.shape[-self.X:]
                ], indexing='ij'
            ), dim=-1
        )
        coords = torch.cat(
            [grid.view(-1, self.X), torch.ones(
                (grid.numel()//self.X, 1), dtype=x.dtype, device=x.device)
            ], dim=-1
        )
        aff_coords = (coords @ aff.transpose(-1,-2)).view(
            nT, *x.shape[-self.X:], 4)
                  
        coords = (
            torch.cat([grid.view(-1, self.X),
                       torch.ones((grid.numel()//self.X,1), device=x.device)
            ], dim=-1).to(x.device) @ aff.transpose(0,1)
        ).view(*x.shape[-self.X:], self.X + 1)
        
        disp = 2 * aff_coords[..., :self.X] / (
            torch.tensor(x.shape[-self.X:], device=x.device) - 1
        )
        return disp

    
    def _make_elastic(self, x, nT):
        f = self.elastic_factor
        n = self.n_elastic_steps

        # Field shapes
        sz_full = x.shape[-self.X:]
        sz_small = (1, 1) + tuple(torch.ceil(torch.tensor(sz_full) * f
        ).to(torch.int).tolist()) + (self.X,)
        sz_half = (1, 1) + tuple(torch.ceil(torch.tensor(sz_full) * 0.5
        ).to(torch.int).tolist()) + (self.X,)

        elastic = torch.zeros((x.shape[1:] + (self.X,)), dtype=x.dtype
        ).to(x.device)

        for t in range(nT):
            # Create SVF
            svf_small = torch.normal(
                mean=0., std=torch.rand(1).item(), size=tuple(sz_small)
            ).to(x.device)
            svf_half =  torch.stack([
                F.interpolate(svf_small[...,i],
                              size=tuple(sz_half[-self.X-1:-1]),
                              mode='trilinear' if self.X == 3 else 'bilinear'
                ) for i in range(self.X)], dim=-1
            )
            
            # Scaling and squaring
            svf_half = (svf_half / (2 ** self.n_elastic_steps)
            ).squeeze(1).movedim(-1,1)
            grid_half = torch.stack(
                torch.meshgrid(
                    [torch.arange(s, dtype=svf_half.dtype, device=x.device)
                     for s in svf_half.shape[-self.X:]
                    ], indexing='ij'
                ), dim=-1
            ).movedim(-1,0).unsqueeze(0)
            weights = (2 / torch.tensor(grid_half.shape[-self.X:])
            ).to(x.device)
            
            for _ in range(n - 1):
                grid_interp = (svf_half + grid_half).movedim(1,-1) * weights
                svf_half += F.grid_sample(
                    svf_half, grid_interp, align_corners=True
                )

            # Interpolate to full size
            elastic[:, t, ...] = F.interpolate(
                svf_half, size=sz_full[-self.X:],
                align_corners=True, mode='trilinear'
            ).movedim(1,-1)

        return elastic


    def __call__(self, inputs):
        _, _, nT = inputs[0].shape[:-self.X]

        if self.apply_affine and self.apply_elastic:
            A = self._make_affine(inputs[0]).unsqueeze(0)
            E = self._make_elastic(inputs[0])
            T = A + E
        elif self.apply_affine:
            T = self._make_affine(inputs[0]).unsqueeze(0)
        elif self.apply_elastic:
            T = self._make_elastic(inputs[0])
        else:
            T = None

        for t in range(nT):            
            inputs = inputs if T is not None else [
                F.grid_sample(
                    x[:,:,t,...], T[:,t,...].permute(0, 3, 2, 1, 4),
                    align_corners=True, mode='bilinear'
                ) for x in inputs
            ]
                       
        return inputs

    

class CropPatch:
    def __init__(self,
                 patch_sz:[int, list]=None,  # size of crop patch
                 randomize:bool=True,        # randomize crop patch ?
                 X:int=3,                    # no. image dims
    ):
        self.X = X
        self.patch_sz = [patch_sz] * X if isinstance(patch_sz, int)\
            else patch_sz
        
        if self.patch_sz is not None:
            if randomize:
                self._get_bounds = self._get_random_crop_bounds
            else:
                self._get_bounds = self._get_center_crop_bounds

                
    def _get_bounding_box(self, vol, bffr:int=4):
        vol = torch.squeeze(vol)
        bbox = [[0, vol.shape[i]-1] for i in range(len(vol.shape))]

        while vol[bbox[0][0],:,:].sum() == 0: bbox[0][0] += 1
        while vol[bbox[0][1],:,:].sum() == 0: bbox[0][1] -= 1
        while vol[:,bbox[1][0],:].sum() == 0: bbox[1][0] += 1
        while vol[:,bbox[1][1],:].sum() == 0: bbox[1][1] -= 1
        while vol[:,:,bbox[2][0]].sum() == 0: bbox[2][0] += 1
        while vol[:,:,bbox[2][1]].sum() == 0: bbox[2][1] -= 1

        bbox = [[bb[0] - (bffr), bb[1] + (bffr)] for bb in bbox]
        return bbox


    def _get_center_crop_bounds(self, vol_sz, bbox:list=None):
        vol_sz = vol_sz[-self.X:]
        patch_sz = self.patch_sz

        if bbox is not None:
            center = [(bb[0] + bb[1])//2 for bb in bbox]
        else:
            center = [vs//2 for vs in vol_sz]
        bounds = [[c - ps//2, c + ps//2] for c, ps in zip(center, patch_sz)]

        for i in range(self.X):
            if bounds[i][0] < 0:  shift = bounds[i][0]
            elif bounds[i][1] > vol_sz[i]:  shift = bounds[i][1] - vol_sz[i]
            else:  shift = 0
            bounds[i] = [bounds[i][0] - shift, bounds[i][1] - shift]

        return bounds


    def _get_random_crop_bounds(self, vol_sz, bbox:list=None,
                                return_bounds=False):
        vol_sz = vol_sz[-self.X:]
        patch_sz = self.patch_sz

        if bbox is not None:
            rand_bounds = [
                [max([0, bbox[i][1] - patch_sz[i] + 1]),
                 min([vol_sz[i] - patch_sz[i], bbox[i][0]])]
                for i in range(self.X)
            ]
        else:
            rand_bounds = [[0, vol_sz[i] - patch_sz[i]] for i in range(self.X)]

        start_idx = [
            rand_bounds[i][1] if rand_bounds[i][1] <= rand_bounds[i][0]
            else npr.randint(rand_bounds[i][0], rand_bounds[i][1])
            for i in range(self.X)
        ]
        bounds = [
            (start_idx[i], start_idx[i] + patch_sz[i]) for i in range(self.X)
        ]

        return bounds
    

    def _apply_crop(self, vol, bounds):
        if self.X == 2:
            h, w = bounds
            crop = vol[..., h[0]:h[1], w[0]:w[1]]
        elif self.X >= 3:
            h, w, d = bounds
            crop = vol[..., h[0]:h[1], w[0]:w[1], d[0]:d[1]]
        else:
            print(f'Invalid X (X=={self.X}')

        return crop

    
    def __call__(self, inputs, return_bounds=False):
        if self.patch_sz is not None:
            bbox = None
            bounds = self._get_bounds(inputs[0].shape, bbox)
            inputs = [self._apply_crop(x, bounds) for x in inputs]
        
        return inputs



class FlipTransform:
    def __init__(self,
                 flip_axis:int=None, # axis to flip
                 chance:float=0.5,   # probability of flipping
                 X=3                 # number of image dims
    ):
        self.X = X
        self.chance = chance \
            if (chance <= 1 and chance >= 0) or chance is not None \
               else fatal('Invalid chance (must be float between 0 and 1)')
        self.flip_axis = flip_axis \
            if isinstance(flip_axis, int) or flip_axis is None \
               else fatal('Invalid flip_axis (must be int between 0 and '
                          'number of spatial dimensions)')

    def __call__(self, inputs):
        if torch.rand((1)) < self.chance:
            inputs = [torch.fliplr(
                x.transpose(1,self.flip_axis)
            ).transpose(1,self.flip_axis) for x in inputs]
        return inputs



#-----------------------------------------------------------------------------#
#                                  Misc funcs                                 #
#-----------------------------------------------------------------------------#

class AssignOneHotLabels:
    """
    Performs one-hot encoding of an input label image
    """
    def __init__(self,
                 label_values:list=None, # Label values to encode
                 X:int=3,                # no. spatial dims
    ):
        self.label_values = label_values
        self.X = X

        
    def _one_hot_encode(self, x):
        if self.label_values == None:
            self.label_values = torch.unique(x)

        onehot = torch.zeros(x.shape).to(x.device)
        if self.X == 3:
            onehot = onehot.repeat(1,len(self.label_values),1,1,1,1)
        elif self.X == 2:
            onehot = onehot.repeat(1,len(self.label_values),1,1,1)

        x = x.squeeze()
        for i in range(0, len(self.label_values)):
            onehot[:,i,...] = x==self.label_values[i]

        return onehot

    
    def __call__(self, inputs):
        inputs[1] = self._one_hot_encode(inputs[1]).to(inputs[0].dtype)
        return inputs



class ComposeTransforms:
    def __init__(self, transform_list):
        self.transforms = [t for t in transform_list if t is not None]
        
    def __call__(self, inputs):
        for T in self.transforms:
            if T is not None:
                inputs = T(inputs)
        return inputs



