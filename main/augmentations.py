import sys, os
import numpy as np
from numpy import random as npr
import math
import random
import surfa as sf

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import Compose

import utils


#-----------------------------------------------------------------------------#
#                               Intensity funcs                               #
#-----------------------------------------------------------------------------#

class BiasField:
    """
    Simulates bias field in input tensor [y = x * exp(B)]. Applies a different
    field to each channel/timepoint
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
        # Get shapes
        _, nC, nT = x.shape[:-self.X]
        sz_full = tuple(x.shape[-self.X:])
        sz_small = (1, nC*nT) + resize_shape(sz_full, self.shape_factor)
        
        # Create small-sized field for each timepoint/channel
        bf_small = torch.normal(
            mean=0., size=sz_small,
            std=(self.std * random.random() if self.randomize else self.std)
        ).to(x.device)
        
        # Resize to x.shape and apply
        bf_full = F.interpolate(
            bf_small, size=sz_full, align_corners=True,
            mode='trilinear' if self.X == 3 else 'bilinear'
        ).view(x.shape)
        
        return x * torch.exp(bf_full)

        
    def __call__(self, inputs):
        inputs[0] = self._apply_bias_field(inputs[0])
        return inputs

    
#------------------------------------------------------------------------------

class GammaTransform:
    """
    Gamma transform on input tensor [y = x ** exp(gamma)]
    """
    def __init__(self,
                 std:float=0.5,        # std of gamma value
                 randomize:bool=True,  # flag to randomize
                 X:int=3               # no. spatial dims
    ):
        self.std = std
        self.randomize = randomize
        self.X = X

        
    def _gamma_transform(self, x):
        # Apply to each channel/timepoint
        torch.normal(
            mean=0., size=x.shape,
            std=(self.std * random.random() if self.randomize else self.std)
        ).to(x.device)

        return x.pow(torch.exp(gamma))
    
    
    def __call__(self, inputs):
        inputs[0] = self._gamma_transform(inputs[0])
        return inputs
    

#------------------------------------------------------------------------------

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
        # Apply to each channel/timepoint
        noise = torch.normal(
            mean=0., size=x.shape,
            std=(self.std * random.random() if self.randomize else self.std)
        ).to(x.device)
        
        return x + noise


    def __call__(self, inputs):
        inputs[0] = self._noise(inputs[0])
        return inputs


#------------------------------------------------------------------------------

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
        self.X = X
        
        
    def _robust_norm(self, x):
        # Get shapes
        full_sz = tuple(x.shape)
        _, nC, nT = full_sz[:-self.X]
        n_vox = np.prod(full_sz[-self.X:])
        flat_sz = full_sz[:-self.X] + (n_vox,)

        # Convert percentages to intensities
        x_sorted, _ = x.reshape(flat_sz).sort()
        m = x_sorted[..., max(int(self.mperc * flat_sz[-1]), 0)][0]
        M = x_sorted[..., min(int(self.Mperc * flat_sz[-1]), flat_sz[-1]-1)][0]

        # Robust normalization
        for c in range(nC):
            for t in range(nT):
                x[:, c, t, ...] = torch.clamp(
                    x[:, c, t, ...], min=m[c, t], max=M[c, t]
                )
        return utils._min_max_norm(x, m=self.m, M=self.M)
        
    
    def __call__(self, inputs):
        inputs[0] = self._robust_norm(inputs[0]) if self.use_robust \
             else self._min_max_norm(inputs[0])
        return inputs

    
#-----------------------------------------------------------------------------#
#                                Spatial funcs                                #
#-----------------------------------------------------------------------------#

class AffineElasticTransform:
    """
    Applies a spatial transform (affine + elastic) to an input tensor with 
    dimensions [B, C, T, ...]. The same transform should be applied across all
    channels, batches, and timepoints (unlike w/ the intensity augmentations).
    """
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
        self.randomize = randomize
        
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

            self.translation_bounds = _parse_affine_param(translations)
            self.rotation_bounds = _parse_affine_param(rotations)
            self.shear_bounds = _parse_affine_param(shears, center=0.)
            self.scale_bounds = _parse_affine_param(scales, center=1.)

        # Parse elastic transform parameters
        if self.apply_elastic:
            self.elastic_factor = elastic_factor
            self.elastic_std = elastic_std
            self.n_elastic_steps = n_elastic_steps


    def _AffineDisplacementField(self, x):
        """
        Randomly sample parameters (translations, rotations, shearing, and 
        scaling) and generate affine transform
        """
        def _sample_params(bounds):
            bounds_range = torch.diff(bounds).squeeze()
            return torch.rand(self.X) * bounds_range + bounds[:,0]

        n_vox = np.prod(tuple(x.shape[-self.X:]))
        I = torch.eye(self.X+1)

        # Translations
        T = I.clone()
        T[torch.arange(self.X),-1] = _sample_params(self.translation_bounds)

        # Shears
        Sinds = torch.ones((self.X+1,self.X+1), dtype=torch.bool)
        Sinds[torch.eye(self.X+1, dtype=torch.bool)] = False
        Sinds[-1,:] = False
        Sinds[:,-1] = False

        S = I.clone()
        S[Sinds] = torch.cat(
            [_sample_params(self.shear_bounds),
             _sample_params(self.shear_bounds)
            ], dim=-1
        )
        
        # Zooms
        Z = I.clone()
        Z[torch.arange(self.X),
          torch.arange(self.X)] = _sample_params(self.scale_bounds)

        # Rotations
        rotations = _sample_params(self.rotation_bounds) * torch.pi/180
        c, s = [torch.cos(rotations), torch.sin(rotations)]

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
                   [c[2], s[2], -s[2], c[2]])

        # Convert affine matrix to displacement field
        aff = (T @ R3 @ R2 @ R1 @ Z @ S).to(x.device)
        grid = self._meshgrid_coords(x)
        coords = torch.cat(
            [self._meshgrid_coords(x).view(-1, self.X),
             torch.ones((n_vox, 1)).to(x.device)
            ], dim=-1
        )
        coords_aff = coords @ aff.transpose(-2, -1)
        grid_aff = coords_aff[..., :self.X].view(*x.shape[-self.X:], -1)
        disp = grid_aff.unsqueeze(0) * (
            2 / (torch.tensor(x.shape[-self.X:]) - 1).to(x.device)
        )
        return disp


    def _ElasticDisplacementField(self, x):
        """
        Randomly generate an elastic deformation field
        """
        def _resize_shape(sz, mult):
            return tuple(
                torch.ceil(torch.tensor(sz) * mult).to(torch.int).tolist()
            )

        # Get field shapes
        sz_full = tuple(x.shape[-self.X:])
        sz_small = (1, self.X) + _resize_shape(sz_full, self.elastic_factor)
        sz_half = (1, self.X) + _resize_shape(sz_full, 0.5)

        # Create small sized SVF
        std = (random.uniform(0, self.elastic_std) if self.randomize
               else self.elastic_std
        )
        svf_small = torch.normal(mean=0., std=std, size=sz_small).to(x.device)

        # Resize to half of full shape
        svf_half =  F.interpolate(
            svf_small, size=sz_half[-self.X:],
            mode='trilinear' if self.X == 3 else 'bilinear'
        )

        # Integrate w/ scaling and squaring to smooth
        svf_half /= (2 ** self.n_elastic_steps)
        grid_half = self._meshgrid_coords(svf_half)

        weights = 2 / (torch.tensor(sz_half[-self.X:])).to(x.device)

        for _ in range(self.n_elastic_steps - 1):
            grid_interp = weights * (svf_half.movedim(1, -1) + grid_half)
            svf_half += F.grid_sample(
                svf_half, grid_interp, align_corners=True, mode='bilinear'
            )

        # Interpolate to full size
        elastic = F.interpolate(
            svf_half, size=sz_full[-self.X:], align_corners=True,
            mode='trilinear' if self.X == 3 else 'bilinear'
        )
        disp = elastic.movedim(1,-1) * (
            2 / (torch.tensor(x.shape[-self.X:]) - 1).to(x.device)
        )
        return disp


    def _meshgrid_coords(self, x):
        """
        Creates a meshgrid centered around origin for a tensor of
        shape=x.size=[N, C, H, W, D]
        """
        grid = torch.stack(
            torch.meshgrid(
                [torch.arange(0, s, dtype=x.dtype, device=x.device)
                 for s in x.shape[-self.X:]
                ], indexing='ij'
            ), dim=-1
        )
        grid -= ((grid.max() - grid.min()) / 2.)
        return grid


    def __call__(self, inputs):
        """
        """
        # Generate affine and elastic displacement fields
        A = (self._AffineDisplacementField(inputs[0]) if self.apply_affine
             else None)
        E = (self._ElasticDisplacementField(inputs[0]) if self.apply_elastic
             else None)
        A = None
        # Compose into single transform
        if A is None and E is None:
            T = None
        elif A is not None and E is not None:
            T = A + E
        else:
            T = E + self._meshgrid_coords(inputs[0]) * (
                2 / torch.tensor(inputs[0].shape[-self.X:])
            ).to(inputs[0].device) if A is None else A

        # Apply to each input
        for n, x in enumerate(inputs):
            sz = x.shape
            nB, nC, nT = sz[:-self.X]
            inputs[n] = F.grid_sample(
                x.view((nB, nC*nT) + sz[-self.X:]), T.permute(0, 3, 2, 1, 4),
                align_corners=True, mode='bilinear'
            ).view(sz)

        return inputs

    
#------------------------------------------------------------------------------
    
class CropPatch:
    def __init__(self,
                 patch_sz:[int, list]=None,  # size of crop patch
                 randomize:bool=True,        # randomize crop patch ?
                 X:int=3,                    # no. image dims
    ):
        self.X = X
        self.patch_sz = [patch_sz] * X if isinstance(patch_sz, int)\
            else patch_sz
        
        self._get_bounds = (
            self._get_random_crop_bounds if randomize
            else self._get_center_crop_bounds
        )


    def _get_center_crop_bounds(self, vol_sz, bbox:list=None):
        """
        Get bounds of crop window centered within input image
        """
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
                                return_bounds=False
    ):
        """
        Get bounds of randomly placed crop window within input image
        """
        vol_sz = vol_sz[-self.X:]
        patch_sz = self.patch_sz

        if bbox is not None:
            rand_bounds = [
                [max([0, bbox[i][1] - patch_sz[i] + 1]),
                 min([vol_sz[i] - patch_sz[i], bbox[i][0]])]
                for i in range(self.X)
            ]
        else:
            rand_bounds = [
                [0, vol_sz[i] - patch_sz[i]] for i in range(self.X)
            ]

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
        """
        Extract patch of input volume within bounds (same bounds for all 
        channels/timepoints)
        """
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


#------------------------------------------------------------------------------

class FlipTransform:
    """
    Flips input image across left/right axis
    """
    def __init__(self,
                 flip_axis:int=None, # axis to flip
                 chance:float=0.5,   # probability of flipping
                 X=3                 # number of image dims
    ):
        self.X = X
        self.chance = chance \
            if (chance <= 1 and chance >= 0) or chance is not None \
               else fatal('Invalid chance (must be float between 0 and 1)')
        self.flip_axis = flip_axis
        """
        self.flip_axis = flip_axis \
            if isinstance(flip_axis, int) or flip_axis is None \
               else fatal('Invalid flip_axis (must be int between 0 and '
                          'number of spatial dimensions)')
        """
        
    def __call__(self, inputs):
        if torch.rand((1)) < self.chance:
            axis = self.flip_axis + 3
            inputs = [torch.flip(x, dims=[axis]) for x in inputs]
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


#------------------------------------------------------------------------------

class ComposeTransforms:
    """
    Sequentially applies a list of transform classes to a list of input tensors
    """
    def __init__(self, transform_list):
        self.transforms = [t for t in transform_list if t is not None]
        
    def __call__(self, inputs):
        for T in self.transforms:
            if T is not None:
                inputs = T(
                    inputs if isinstance(inputs, list) else list(inputs)
                )
        return inputs


#------------------------------------------------------------------------------

def resize_shape(sz, mult):
    """
    Multiplies input size (tuple) and returns new size as tuple
    """
    return tuple(
        torch.ceil(torch.tensor(sz) * mult).to(torch.int).tolist()
    )


def visualize(x):
    """
    Visualize an input tensor image w/ Freeview
    """
    img = x.cpu().numpy().squeeze(0)
    fv = sf.vis.Freeview()
    for chn in range(img.shape[0]):
        fv.add_image(img[0, ...])
    fv.show()


def write(xlist, basenames,
          is_labels:list[bool]=[False],
          is_onehot:list[bool]=[False],
          n_dims:int=3
):
    """
    Write outputs of augmentation functions
    """
    for n, (x, fbase) in enumerate(zip(xlist, basenames)):
        x = torch.argmax(x, dim=1).unsqueeze(1) if is_onehot[n] else x

        if len(x.squeeze(0).squeeze(0).shape) == n_dims + 1:
            x = x.squeeze(0).squeeze(0).cpu().numpy().astype(
                np.int32 if is_labels[n] else np.float32
            )
            for t in range(x.shape[0]):
                img = sf.Volume(x[t, ...]).save(f'{fbase}.{t}.mgz')
        else:
            print(f'not yet implemented, cannot write {fbase}')
