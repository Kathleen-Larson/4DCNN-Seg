import sys
import os
import numpy as np
from numpy import random as npr
import math
import random
import surfa as sf
import gc

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import Compose

from typing import List

"""
This script is managed by Kathleen Larson (klarson9@mgh.harvard.edu). It doesn't appear in any
publications (yet!), but please let me know if you are planning to use it!

These are all classes for data augmentation updated to handle 6D tensors with dimensions 
[B, C, T=n_timepoints, H, W, D].
"""


# ------------------------------------------------------------------------------------------------ #
#                                  Intensity-based augmentations
# ------------------------------------------------------------------------------------------------ #

class BiasField:
    """
    Simulates bias field in input tensor [y = x * exp(B)]. Applies a different field to each
    channel/timepoint
    """
    def __init__(self,
                 shape_factor=0.025,  # ratio of small field to img size
                 max_value=1.0,       # max value of bias field
                 std=0.3,             # std of bias field
                 randomize=True,      # flag to randomize parameters(false = use max value)
                 X=3):                # number of spatial dims

        self.shape_factor = shape_factor
        self.std = std
        self.max_value = max_value
        self.randomize = randomize
        self.X = X

    def _apply_bias_field(self, x):
        # Get shapes
        _, nC, nT = x.shape[:-self.X]
        sz_full = tuple(x.shape[-self.X:])
        sz_small = (1, nC * nT) + resize_shape(sz_full, self.shape_factor)

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


class GammaTransform:
    """
    Gamma transform on input tensor [y = x ** exp(gamma)]
    """
    def __init__(self,
                 std=0.5,         # std of gamma value
                 randomize=True,  # flag to randomize
                 X=3):            # no. spatial dims

        self.std = std
        self.randomize = randomize
        self.X = X

    def _gamma_transform(self, x):
        # Randomly generate gamma values
        nC, nT = x.shape[1:-(self.X)]
        gamma = torch.normal(
            mean=0., size=(nC * nT,),
            std=(self.std * random.random() if self.randomize else self.std)
        ).to(x.device).view(nC, nT)

        # Apply to each channel/timepoint
        for c in range(nC):
            for t in range(nT):
                x[:, c, t, ...] = x[:, c, t, ...].pow(torch.exp(gamma[c, t]))

        return x

    def __call__(self, inputs):
        inputs[0] = self._gamma_transform(inputs[0])
        return inputs


class GaussianNoise:
    """
    Adds gaussian noise to input tensor [y = x + N]
    """
    def __init__(self,
                 std=21.0,        # std of gaussian noise
                 randomize=True,  # flag to randomize
                 X=3):            # no. spatial dims

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


class MinMaxNorm:
    """
    Robust intensity and min-max normalization
    """
    def __init__(self,
                 min_int=0.,        # minimum intensity value
                 max_int=1.,        # maximum intensity value
                 min_perc=0.,       # minimum % to clip intensities
                 max_perc=0.95,     # maximum % to clip intensities
                 use_robust=False,  # flag to use robust norm
                 X=3):              # no. spatial dims

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
        M = x_sorted[..., min(int(self.Mperc * flat_sz[-1]), flat_sz[-1] - 1)][0]

        # Robust normalization
        for c in range(nC):
            for t in range(nT):
                x[:, c, t, ...] = torch.clamp(
                    x[:, c, t, ...], min=m[c, t], max=M[c, t]
                )
        return min_max_norm(x, m=self.m, M=self.M)

    def __call__(self, inputs):
        inputs[0] = self._robust_norm(inputs[0])
        return inputs


# ------------------------------------------------------------------------------------------------ #
#                                     Spatial augmentations
# ------------------------------------------------------------------------------------------------ #

class AffineElasticTransform:
    """
    Applies a spatial transform (affine + elastic) to an input tensor with dimensions 
    [B, C, T, ...]. The same transform should be applied across all channels, batches, and 
    timepoints (unlike w/ the intensity augmentations).
    """
    def __init__(self,
                 translations=0.0,       # translation bounds
                 rotations=15.0,         # rotation bounds (degrees)
                 shears=0.012,           # shearing bouinds
                 scales=0.15,            # scaling bounds
                 elastic_factor=0.0625,  # small SVF sz : full sz
                 elastic_std=3.,         # std of gaussian for SVF
                 n_elastic_steps=7,      # no. of integration steps
                 apply_affine=True,      # perform affine trans
                 zero_center=False,      # make the composed warp 0 mean
                 apply_elastic=True,     # perform elastic trans
                 randomize=True,         # randomize params?
                 returns=['output'],     # what to return from _call_. Could be 'output' or 'warp'
                 X=3):                   # no. of spatial dims

        self.X = X
        self.apply_affine = apply_affine
        self.apply_elastic = apply_elastic
        self.randomize = randomize
        self.returns = returns
        self.zero_center = zero_center

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
                        fatal(
                            'Input affine parameter must be float OR list of '
                            'len==2 if same bounds for all image dims, '
                            'len==nDims where each dim has bounds [-val, val], '
                            'or len==(nDims*2) where each dim has bounds '
                            '[min, max].'
                        )
                else:
                    param = [[-param, param]] * X
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
            return torch.rand(self.X) * bounds_range + bounds[:, 0]

        n_vox = np.prod(tuple(x.shape[-self.X:]))

        # Translations
        T = torch.eye(self.X + 1)
        T[torch.arange(self.X), -1] = _sample_params(self.translation_bounds)

        # Shears
        Sinds = torch.ones((self.X + 1, self.X + 1), dtype=torch.bool)
        Sinds[torch.eye(self.X + 1, dtype=torch.bool)] = False
        Sinds[-1, :] = False
        Sinds[:, -1] = False

        S = torch.eye(self.X + 1)
        S[Sinds] = torch.cat(
            [_sample_params(self.shear_bounds), _sample_params(self.shear_bounds)], dim=-1
        )

        # Zooms
        Z = torch.eye(self.X + 1)
        Z[torch.arange(self.X),
          torch.arange(self.X)] = _sample_params(self.scale_bounds)

        # Rotations
        rotations = _sample_params(self.rotation_bounds) * torch.pi / 180
        c, s = [torch.cos(rotations), torch.sin(rotations)]

        [R1, R2, R3] = [torch.eye(self.X + 1) for n in range(3)]
        if self.X == 2:
            R1[torch.tensor([0, 1, 0, 1]),
               torch.tensor([0, 0, 1, 1])] = torch.tensor([c, s, -s, c])
        else:
            R1[torch.tensor([1, 2, 1, 2]),
               torch.tensor([1, 1, 2, 2])] = torch.tensor([c[0], s[0], -s[0], c[0]])
            R2[torch.tensor([0, 2, 0, 2]),
               torch.tensor([0, 0, 2, 2])] = torch.tensor([c[1], s[1], -s[1], c[1]])
            R3[torch.tensor([0, 1, 0, 1]),
               torch.tensor([0, 0, 1, 1])] = torch.tensor([c[2], s[2], -s[2], c[2]])

        # Convert affine matrix to displacement field
        aff = (T @ R3 @ R2 @ R1 @ Z @ S).to(x.device)
        grid = self._meshgrid_coords(x)
        coords = torch.cat(
            [self._meshgrid_coords(x).view(-1, self.X), torch.ones((n_vox, 1)).to(x.device)], dim=-1
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
        std = (random.uniform(0, self.elastic_std) if self.randomize else self.elastic_std)
        svf_small = torch.normal(mean=0., std=std, size=sz_small).to(x.device)

        # Resize to half of full shape
        svf_half = F.interpolate(
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
        disp = elastic.movedim(1, -1) * (
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
                [torch.arange(0, s, dtype=x.dtype, device=x.device) for s in x.shape[-self.X:]],
                indexing='ij'
            ), dim=-1
        )
        grid -= ((grid.max() - grid.min()) / 2.)
        return grid

    def coords_to_displacements(self, Tcoords, image_shape):
        '''
        Convert a tensor of image coords in [-1, 1] to  voxel displacements 
        (useful for SuperWarp-type learning),

        input warp shape should be (batch, ndim, x, y, z), which gets modified
        internally to (batch, x, y, z, ndim) as that is what gr0d_sample needs
        '''
        Tcoords = Tcoords.movedim(1, -1)
        sh = image_shape[-self.X:]
        center = ((torch.tensor(sh).to(Tcoords.device) - 1) / 2)[None, None, None]

        grid_x, grid_y, grid_z = torch.meshgrid(
            torch.arange(sh[0]), torch.arange(sh[1]), torch.arange(sh[2]), indexing='ij'
        )
        coords = torch.stack((grid_x, grid_y, grid_z), dim=-1).to(Tcoords.device)
        Tdisplacements = Tcoords * center + center - coords
        return Tdisplacements.movedim(-1, 1)

    def displacements_to_coords(self, Tdisplacements, image_shape): 
        '''
        Convert a tensor of voxel displacements to one of image coords in [-1, 1],
        which is what is needed by apply_warp

        input warp shape should be (batch, ndim, x, y, z), which gets modified
        internally to (batch, x, y, z, ndim) as that is what gr0d_sample needs
        '''
        Tdisplacements = Tdisplacements.movedim(1, -1)
        sh = image_shape[-self.X:]
        center = ((torch.tensor(sh).to(Tdisplacements.device) - 1) / 2)[None, None, None]
        grid_x, grid_y, grid_z = torch.meshgrid(
            torch.arange(sh[0]), torch.arange(sh[1]), torch.arange(sh[2]), indexing='ij'
        )
        coords = torch.stack((grid_x, grid_y, grid_z), dim=-1).to(Tdisplacements.device)
        Tcoords = (Tdisplacements + coords - center) / center
        return Tcoords.movedim(-1, 1)

    def apply_warp(self, Tcoords, inputs): 
        '''
        apply an input warp with shape (batch, ndim, x, y, z) to a list of inputs of shape
        [batch, channels, time, X, Y, Z]
        input warp Tcoords should be in relative coordinates format, NOT voxel displacements
        '''
        Tcoords = Tcoords.movedim(1, -1)
        sz = inputs[0].shape
        for n, x in enumerate(inputs):
            sz = x.shape
            nB, nC, nT = sz[:-self.X]
            inputs[n] = F.grid_sample(
                x.to(torch.float).view((nB, nC * nT) + sz[-self.X:]),
                Tcoords.permute(0, 3, 2, 1, 4),
                align_corners=True,
                mode='bilinear'
            ).view(sz)

        return inputs

    def __call__(self, inputs):
        # Generate affine and elastic displacement fields
        A = (self._AffineDisplacementField(inputs[0]) if self.apply_affine
             else None)
        E = (self._ElasticDisplacementField(inputs[0]) if self.apply_elastic
             else None)

        # Compose into single transform
        if A is None and E is None:
            T = None
        elif A is not None and E is not None:
            T = A + E
        else:
            T = E + self._meshgrid_coords(inputs[0]) * (
                2 / torch.tensor(inputs[0].shape[-self.X:])
            ).to(inputs[0].device) if A is None else A

        T = T.permute(0, 2, 1, 3) if self.X == 2 else T.permute(0, 3, 2, 1, 4)

        # Apply to each input
        if T is not None:
            if self.zero_center:
                T = T - T.mean()
            for n, x in enumerate(inputs):
                sz = x.shape
                nB, nC, nT = sz[:-self.X]
                inputs[n] = F.grid_sample(
                    x.to(torch.float).view((nB, nC * nT) + sz[-self.X:]),
                    T, align_corners=True, mode='bilinear'
                ).view(sz)

        returns = []

        if 'output' in self.returns:
            if len(self.returns) > 1:
                returns += [inputs]
            else:
                returns = inputs   # backwards compatibility

        if 'warp' in self.returns:
            returns += [T.movedim(-1, 1)]

        return returns


class CropPatch:
    def __init__(self,
                 patch_sz=None,   # size of crop patch
                 randomize=True,  # randomize crop patch location (false = use image center
                 use_com=True,    # flag to use center of mass (of binary FG mask) for center crop
                 X=3):            # no. image dims

        self.X = X
        self.patch_sz = [patch_sz] * X if isinstance(patch_sz, int) else patch_sz
        if len(self.patch_sz) != self.X:
            fatal(
                f'Error in augmentations.CropPatch: patch_sz must have X={X} number of dimensions '
                f'(input was patch_sz={patch_sz})'
            )

        # Random or center crop?
        self._get_bounds = (
            self._get_random_crop_bounds if randomize
            else self._get_center_crop_bounds
        )
        self.use_com = use_com if not randomize else False

    def _center_of_mass(self, x):
        """
        Center the crop bounds around foreground (e.g., binary mask of all foreground labels)
        """
        return [m.to(float).mean().to(int) for m in torch.where(x[:, 1:, ...] > 0.)[self.X:]]

    def _get_center_crop_bounds(self, full_sz, bbox=None, center=None):
        """
        Get bounds of crop window centered within input image
        """
        full_sz = full_sz[-self.X:]
        patch_sz = self.patch_sz

        center = (
            [vs // 2 for vs in full_sz] if center is None and bbox is None
            else [(bb[0] + bb[1]) // 2 for bb in bbox] if center is None
            else center
        )
        bounds = [[c - ps // 2, c + ps // 2] for c, ps in zip(center, patch_sz)]

        for i in range(self.X):
            if bounds[i][0] < 0:
                shift = bounds[i][0]
            elif bounds[i][1] > full_sz[i]:
                shift = bounds[i][1] - full_sz[i]
            else:
                shift = 0
            bounds[i] = [bounds[i][0] - shift, bounds[i][1] - shift]

        return bounds

    def _get_random_crop_bounds(self, full_sz, bbox=None, return_bounds=False, center=None):
        """
        Get bounds of randomly placed crop window within input image
        """
        full_sz = full_sz[-self.X:]
        patch_sz = self.patch_sz

        if bbox is not None:
            rand_bounds = [
                [max([0, bbox[i][1] - patch_sz[i] + 1]),
                 min([full_sz[i] - patch_sz[i], bbox[i][0]])]
                for i in range(self.X)
            ]
        else:
            rand_bounds = [[0, full_sz[i] - patch_sz[i]] for i in range(self.X)]

        start_idx = [
            rand_bounds[i][1] if rand_bounds[i][1] <= rand_bounds[i][0]
            else npr.randint(rand_bounds[i][0], rand_bounds[i][1])
            for i in range(self.X)
        ]
        bounds = [(start_idx[i], start_idx[i] + patch_sz[i]) for i in range(self.X)]
        return bounds

    def _apply_crop(self, x, bounds):
        """
        Extract patch of input volume within bounds (same bounds for all 
        channels/timepoints)
        """
        repeats = x.shape[:-self.X] + (1,) * self.X

        # Crop input volume
        if self.X == 2:
            h, w = bounds
            crop = x[..., h[0]:h[1], w[0]:w[1]]
        elif self.X >= 3:
            h, w, d = bounds
            crop = x[..., h[0]:h[1], w[0]:w[1], d[0]:d[1]]
        else:
            print(f'Invalid X (X=={self.X}')

        return crop

    def __call__(self, inputs, return_bounds=False):
        if self.patch_sz is not None:
            bounds = self._get_bounds(
                inputs[0].shape, 
                center=(self._center_of_mass(inputs[1]) if self.use_com else None)
            )
            inputs = [self._apply_crop(x, bounds) if x is not None else x for x in inputs]
        return inputs


class FlipTransform:
    """
    Flips input image across left/right axis
    """
    def __init__(self,
                 flip_axis=None,  # axis to flip
                 chance=0.5,      # probability of flipping
                 X=3):            # number of image dims

        self.X = X
        self.chance = (
            chance if (chance <= 1 and chance >= 0) or chance is not None
            else fatal('Invalid chance (must be float between 0 and 1)')
        )
        self.flip_axis = flip_axis

    def __call__(self, inputs):
        if torch.rand((1)) < self.chance and self.flip_axis is not None:
            axis = self.flip_axis + 3
            inputs = [torch.flip(x, dims=[axis]) if x is not None else x for x in inputs]
        return inputs


class Resample:
    """
    Resample input volume (tensor) to different resolution (given an input 
    sampling factor or target shape)
    """
    def __init__(self,
                 resample_factor=1.0,  # % by which to resample image resolution 
                 target_shape=None,    # target output shape
                 X=3):                 # no. spatial dims

        if target_shape is None and resample_factor == 1.0:
            print('Target shape is None and resample factor is 1.0... this will do nothing..')

        self.factor = resample_factor
        self.target_shape = target_shape
        self.X = X

    def __call__(self, inputs):
        # Get new shape
        B, C, T, H, W, D = inputs[0].shape
        Hi = int(H * self.factor)
        Wi = int(W * self.factor)
        Di = int(D * self.factor)
        
        outputs = [torch.zeros((B, C, T, Hi, Wi, Di), device=inputs[0].device)] * len(inputs)
        
        # Interpolate for each timepoint
        for i in range(len(inputs)):
            x = inputs[i]
            dtype = x.dtype

            if not torch.is_floating_point(inputs[i]):
                mode = 'nearest'
                x = x.to(torch.float)
            else:
                mode = 'trilinear' if self.X == 3 else 'bilinear' if self.X == 2 else 'linear'
                
            for t in range(x.shape[2]):
                outputs[i][:, :, t, ...] = F.interpolate(
                    x[:, :, t, ...], size=self.target_shape, scale_factor=self.factor, mode=mode
                ).to(dtype)

        return outputs


# ------------------------------------------------------------------------------------------------ #
#                              Miscellaneous functions/transforms
# ------------------------------------------------------------------------------------------------ #

class AssignOneHotLabels:
    """
    Performs one-hot encoding of an input label image
    """
    def __init__(self, label_values=None, X=3):
        self.label_values = label_values
        self.X = X

    def _one_hot_encode(self, x):
        label_values = x.unique() if self.label_values is None else self.label_values
        onehot = torch.cat(
            [(x == n).to(torch.float) for n in label_values], dim=1
        )
        return onehot

    def __call__(self, inputs):
        inputs[1] = self._one_hot_encode(inputs[1]).to(inputs[0].dtype)
        return inputs


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


def fatal(message):
    print(message)
    sys.exit(1)


def min_max_norm(x, m=0, M=1, eps=1e-8):
    """
    Min-max normalization
    """
    if type(x) is torch.Tensor:
        y = (M - m) * (x - x.min()) / (x.max() - x.min() + eps) + m
    elif type(x) is np.ndarray:
        y = (M - m) * (x - np.min(x)) / (np.max(x) - np.min(x) + eps) + m
    else:
        y = (M - m) * (x - min(x)) / (max(x) - min(x) + eps) + m
    return y


def resize_shape(sz, mult):
    """
    Multiplies input size (tuple) and returns new size as tuple
    """
    return tuple(torch.ceil(torch.tensor(sz) * mult).to(torch.int).tolist())


def visualize(x):
    """
    Visualize an input tensor image w/ Freeview
    """
    img = x.cpu().numpy().squeeze(0)
    fv = sf.vis.Freeview()
    for chn in range(img.shape[0]):
        fv.add_image(img[0, ...])
    fv.show()


def write(xlist,
          basenames,
          is_labels=[False],
          is_onehot=[False],
          n_dims=3):
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
