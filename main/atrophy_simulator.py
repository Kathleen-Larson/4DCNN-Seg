import os
import time
import pathlib as Path
import numpy as np
import pdb
import surfa as sf
import random
import yaml
from collections import OrderedDict

import torch
import torch.optim
import torch.nn as nn
import torch.nn.functional as F

from typing import List
import augmentations_longitudinal as aug

# debug = True
debug = False


"""
This script is managed by Kathleen Larson (klarson9@mgh.harvard.edu). It doesn't appear in any 
publications (yet!), but please let me know if you are planning to use it! 

This is based off of the SynthAtrophyPair function in the neurite_sandbox/tf/models.py script, but 
ported to operate in torch and improved (no offense!) to handle more than 2 timepoints at a time.

The main class here is SynthLongitudinal, which calls several other classes also contained within 
this file. You can initialize the required parameters and structure dictionaries using the 
_config_synth_models function (or just do it on your own). I have been configuring everything using 
a .yaml file, from which I can load a dictionary and input directly into _config_synth_models. The 
forward model for SynthLongitudinal intakes a single label images and performs the following 
operations (each contained within a different class):
1. _ResampleInput: resamples the input image to a different resolution using 
     torch.nn.functional.Interpolate (great for debugging if you need things to run more quickly).
2. _AddLesions: adds a random number of lesions into the label image. This is done by randomly 
     placing a 1 voxel sized lesion into a predefined background label (probably just WM), and then 
     dilating it until it reaches a randomized maximum volume.
3. _ResizeLabels: resizes a set of labels to induce synthetic atrophy. This is done by a series of 
     binary image morphology operations dilate the surrounding labels into the atrophied region to 
     acheive a randomly defined amount of volume loss. Labels to resize are defined in the sdict 
     variable.
4. _LabelsToImage: synthesizes an intensity image from the new label image. Intensities are 
     randomly generated between 0 and a user defined value (default is 255). Can also include a 
     Gaussian blur and white noise for further randomization (set the stds to 0 for these if you 
     don't want it. I have been inducing spatial transformations, bias field simulation and 
     contrast augmentation (via a gamma function) w/ a different script specifically focused on 
     data augmentation, but if you want me to include that here then please reach out and I can add 
     it in. 

At the end, it can also replace labels (if you don't want to include all of the ones used for image 
synthesize in the output (target) segmentation).
"""


class SynthLongitudinal(nn.Module):
    def __init__(
            self,
            sdict,                    # dict of labels to atrophy
            img_dict,                 # dict of LabelsToImage mappings
            neighbors_dict,           # dict of all possible atrophy targets + neighbors
            in_shape,                 # expected shape of input
            blur_std=0.25,            # std dev of gaussian blur in LabelsToImage
            bg_labels=0,              # background label value
            control_bounds=[-.02, .02],  # min/max change in non-target structures
            deform_images=True,       # flag to apply a spatial deformation in LabelsToImage
            do_CSF_augment=False,     # flag to add extra noise to CSF in intensity image
            do_insert_lesions=False,  # flag to insert lesions
            do_resample=False,        # flag to resample to new res
            do_resize_labels=True,    # flag to induce atrophy
            lesions_chance=0.5,       # probability that lesions will be inserted
            lesions_in_label=2,       # labels in which to insert lesions
            lesions_label=77,         # lesion label value
            max_lesions_vol=200,      # max volume of lesions
            max_n_lesions=0,          # max no. lesions to insert
            noise_std=0.05,           # gauss noise std dev in LabelsToImage (rel to max int)
            randomize=True,           # flag to randomize atrophy
            return_sdict=False,       # flag to output dict w/ atrophy labels
            resample_factor=0.5,      # factor for resampling img res
            same_across_tps=True,     # use same random seed for all timepoints
            seg_dict=None,            # dict for labels to include in output
            subsample=1.0,            # % to subsample atrophy
            T=2,                      # no. timepoints to synthesize
            different_lesion_means=False,  # create different means for each lesion label
            X=3,                      # no. spatial dims
            device=None
    ):
        super(SynthLongitudinal, self).__init__()

        # Parse class attributes
        self.device = 'cpu' if device is None else device
        self.T = T
        self.X = X

        self.different_lesion_means = different_lesion_means  # create diff means for each lesion
        '''
        sdict            - the dictionary that governs how labels grow/shrink
        synth_image_dict - the dictionary that allows combining labels so that they have the same
                           means after LabelsToImage
        IO_labels_dict   - the dictionary to remap labels at output

        if different_lesion_means is true there is an internal step in which each lesion is 
        given its own label so that they each have different intensity means, then these labels 
        are added automatically to IO_labels_dict to that they get mapped to the single lesion 
        label before returning the label maps to the caller.
        '''
        self.atrophy_dict = sdict
        self.synth_image_dict = img_dict.copy()  # dict for intensities synthesis
        self.IO_labels_dict = seg_dict.copy()    # dict for output labels (lesions->label)
        self.neighbors_dict = neighbors_dict.copy()  # dict with labels + neighbors
        self.lesions_label = lesions_label
        self.return_sdict = return_sdict
        self.lesions_label = lesions_label

        max_n_lesions = max_n_lesions if do_insert_lesions else 0
        sdict = sdict if do_resize_labels else None

        if self.different_lesion_means and sdict is not None:
            max_sd = np.array([*sdict.keys()]).max()
            max_seg = np.array([*self.IO_labels_dict.keys()]).max()
            for lno in range(2 * max_n_lesions + 2):
                sdict[lno + max_seg] = sdict[lesions_label]
                self.IO_labels_dict[lno + max_seg] = lesions_label

        # Check inputs
        in_shape = (
            (int(in_shape),) * self.X
            if isinstance(in_shape, int) or isinstance(in_shape, float)
            else in_shape
        )
        while len(in_shape) < self.X + 2:
            in_shape = (1,) + tuple(in_shape)

        # Resample input volume ?
        self.resample_input = _ResampleImage(
            resample_factor=resample_factor
        ) if do_resample and resample_factor != 1.0 else nn.Identity()

        # Initialize models for each timepoint
        self.synth_atrophy = nn.ModuleList([])

        for t in range(self.T):
            synth = nn.Module()
            synth.add_lesions = _AddLesions(
                in_shape=in_shape,
                lesions_label=lesions_label,
                max_n_lesions=max_n_lesions,
                max_lesions_vol=max_lesions_vol,
                bg_labels=lesions_in_label,
                different_lesion_means=different_lesion_means,
                chance=lesions_chance,
                device=device
            ) if max_n_lesions > 0 else nn.Identity()
            synth.resize_labels = _ResizeLabels(
                sdict=sdict,
                neighbors_dict=neighbors_dict,
                control_bounds=control_bounds,
                subsample=subsample,
                in_shape=in_shape,
                X=self.X,
                randomize=randomize,
                return_dict=self.return_sdict,
                device=device
            ) if neighbors_dict is not None and len(neighbors_dict) > 0 and t > 0 else nn.Identity()
            self.synth_atrophy.append(synth)

        # Initialize labels to image class
        self.labels_to_image = _LabelsToImage(
            IOdict=img_dict,
            same_across_tps=same_across_tps,
            in_shape=in_shape,
            noise_std=noise_std,
            blur_std=blur_std,
            deform_images=deform_images,
            bg_labels=bg_labels,
            do_CSF_augment=do_CSF_augment,
            X=self.X,
            device=device
        )

    def _seed_intensities_from_reference(self, x):
        """
        Generates dictionary of intensities for each label from reference image.
        """
        labels = [n.item() for n in x[0].unique()]
        y, ref = x

        idict = {}
        for n in labels:
            idict[n] = ref[y == n].mean()

        return idict

    def _replace_labels(self, x):
        for key, val in self.IO_labels_dict.items():
            x[x == key] = val
        return x

    def forward(self, y_in):
        X = [None] * self.T
        y = [None] * self.T

        # Generate intensities for image synthesis (if reference image is supplied)
        intensities_dict = None
        if isinstance(y_in, (list, tuple)):
            intensities_dict = (
                self._seed_intensities_from_reference(y_in) if len(y_in) > 1 else None
            )
            y_in = y_in[0]

        atrophy_dict = [{}] * (self.T - 1)

        # Simulate atrophy
        atrophy_dict = [{}] * (self.T - 1)

        for t in range(self.T):
            if self.different_lesion_means:
                labels_with_lesions, lesion_map = self.synth_atrophy[t].add_lesions(
                    self.resample_input(y_in) if t == 0 else y[t - 1])

                # remap lesions to new labels
                lesion_map = lesion_map.to(labels_with_lesions.dtype)  # convert to int
                old_max = labels_with_lesions.max()
                labels_with_lesions[lesion_map > 1] = lesion_map[lesion_map > 1] + old_max

                # store
                yt = self.synth_atrophy[t].resize_labels(labels_with_lesions)
                y[t], atrophy_dict[t - 1] = yt if isinstance(yt, tuple) else (yt, None)
            else:
                yt = self.synth_atrophy[t].resize_labels(
                    self.synth_atrophy[t].add_lesions(
                        self.resample_input(y_in) if t == 0 else y[t - 1]
                    ),
                )
                y[t], atrophy_dict[t - 1] = yt if isinstance(yt, tuple) else (yt, None)

        y = torch.stack(y, dim=-(self.X + 1))

        X, y = self.labels_to_image(y, intensities_dict)  # X is the image, y is the label map
        y = self._replace_labels(y)

        if 0:
            y1 = torch.clone(y)
            import freesurfer as fs
            fs.fv(y.squeeze().movedim(0, -1).cpu().numpy(), 
                  y0.squeeze().movedim(0, -1).cpu().numpy(), 
                  y1.squeeze().movedim(0, -1).cpu().numpy(), 
                  X.squeeze().movedim(0, -1).cpu().numpy())
            pdb.set_trace()

        return (X, y, atrophy_dict) if self.return_sdict is True else (X, y)


class _AddLesions(nn.Module):
    """
    Class to add a random number of lesions (up to max_n_lesions) within a label image. Lesions 
    will have a maximum volume of max_lesions_vol.
    """
    def __init__(self, 
                 in_shape=None,
                 lesions_label=77,      # value for lesion label
                 chance=0.5,            # prob. of inserting lesions
                 max_n_lesions=1,       # max no. of lesions
                 bg_labels=[2, 41],     # labels to insert lesions
                 bg_buffer=3,           # size of buffer (no. voxels)
                 max_lesions_vol=20.0,  # max lesion volume
                 X=3,                   # no. image dims
                 different_lesion_means=False,   # create different means for each lesion label
                 device=None):
        super(_AddLesions, self).__init__()
        self.device = 'cpu' if device is None else device

        self.lesions_label = lesions_label
        self.chance = chance
        self.max_n_lesions = max_n_lesions
        self.bg_buffer = bg_buffer
        self.bg_labels = bg_labels
        self.max_lesions_vol = max_lesions_vol
        self.X = X
        self.different_lesion_means = different_lesion_means

        if in_shape is None:
            fatal('Must provide input shape to _AddLesions')

        # Set up morphology operations
        structuring_element = torch.zeros((3,) * self.X, dtype=float)
        if self.X == 2:
            structuring_element[
                torch.tensor([0, 1, 1, 1, 2]),
                torch.tensor([1, 0, 1, 2, 1])
            ] = 1.
        elif self.X == 3:
            structuring_element[
                torch.tensor([1, 1, 2, 0, 1, 1, 1]),
                torch.tensor([1, 1, 1, 1, 1, 0, 2]),
                torch.tensor([1, 2, 1, 1, 0, 1, 1])
            ] = 1.
        else:
            print(':(')
        self.dilation_conv = init_convolution(
            in_shape=in_shape,
            out_shape=in_shape,
            conv_weight_data=structuring_element,
            kernel_size=3,
            stride=1,
            padding=1,
            dilation=1,
            bias=False,
            requires_grad=False,
            device=self.device,
            X=self.X,
        )

    def _insert_lesions(self, x):
        """
        Insert lesions within a volume
        """
        # Get mask of background labels
        M = torch.zeros(x.shape, device=x.device).to(float)
        for bg in self.bg_labels:
            M[x == bg] = 1.
        bg_idxs = torch.stack(torch.where(M > 0.))

        # Initialize up to max_n_lesions lesions (1 vox in size)        
        n_lesions = random.randint(1, self.max_n_lesions)

        M[tuple(
            bg_idxs[:, torch.randint(0, int(M.sum()), size=(n_lesions,))]
        )] = torch.arange(2, n_lesions + 2).to(float).to(M.device)

        # Create sdict and grow lesions w/ _ResizeLabels
        lesions_sdict = {}
        for n in range(n_lesions):
            lesions_sdict[n + 2] = [random.uniform(0., self.max_lesions_vol), 1]
            if debug:
                coords = torch.where(M.cpu() == n + 2)
                vol = lesions_sdict[n + 2][0]
                xl, yl, zl = coords[2][0].item(), coords[3][0].item(), coords[4][0].item()
                print(f'adding lesion {n+2} with vol {vol:.2f} at {xl}, {yl}, {zl}')

        M = _ResizeLabels(
            lesions_sdict, x.shape, device=self.device,
            apply_dropout=True, randomize=False, return_dict=False
        )(M)

        # Insert back into original volume
        x = torch.where(M > 1, self.lesions_label, x)
        if self.different_lesion_means:
            return x, M
        else:
            return x

    def forward(self, x):
        """
        Calls function w/ probability of self.chance
        """
        if random.uniform(0., 1.) <= self.chance:
            if self.different_lesion_means:
                x, lesion_map = self._insert_lesions(x)
            else:
                x = self._insert_lesions(x)
                lesion_map = torch.zeros(x.shape).to(x.device)
        else:
            lesion_map = torch.zeros(x.shape).to(x.device)

        if self.different_lesion_means:
            return x, lesion_map
        else:
            return x


class _LabelsToImage(nn.Module):
    """
    Synthesize an intensity image from a label map. Intensities are sampled either from a dictionary
    (IOdict) input at initialization or from a reference image (ref) input to the forward method. 
    In the latter case, an IOdict is created with the function _create_IOdict_from_image.
    """
    def __init__(self,
                 in_shape,              # expected image shape
                 IOdict=None,           # dict of input output label values
                 bg_labels=None,        # background label value(s)
                 blur_std=0.25,         # std dev of gaussian blur
                 do_CSF_augment=True,   # flag to turn on CSF augmentation
                 GM_labels=[3, 42],     # label values of GM (required for CSF augmentation)
                 max_intensity=255,     # max image intensity
                 noise_std=0.05,        # std dev of gaussian white noise (rel. to max intensity)
                 same_across_tps=True,  # use same random seed for all timepts
                 deform_images=True,    # flag to apply a spatial deformation
                 X=3,                   # no. image spatial dims (2 or 3)
                 device=None):
        super(_LabelsToImage, self).__init__()
        self.device = 'cpu' if device is None else device

        # Parse args
        self.IOdict = IOdict
        self.bg_labels = [bg_labels] if not isinstance(bg_labels, (list, tuple)) else bg_labels
        self.GM_labels = [GM_labels] if not isinstance(GM_labels, (list, tuple)) else GM_labels

        self.deform_images = deform_images
        self.do_CSF_augment = do_CSF_augment
        self.max_intensity = max_intensity
        self.noise_std = noise_std
        self.same_across_tps = same_across_tps
        self.X = X

        # Set up Gaussian blur
        if blur_std > 0:
            kernel = gaussian_kernel(blur_std, X=self.X)
            self.gaussian_blur = init_convolution(
                in_shape=in_shape,
                out_shape=in_shape,
                conv_weight_data=kernel,
                kernel_size=kernel.shape,
                padding=[(k - 1) // 2 for k in kernel.shape],
                stride=1,
                dilation=1,
                bias=False,
                requires_grad=False,
                device=self.device,
                X=self.X,
            )
        else:
            self.gaussian_blur = nn.Identity()

        # Set up CSF augmentation?
        if do_CSF_augment:
            structuring_element = torch.zeros((3,) * self.X, dtype=float)
            if self.X == 2:
                structuring_element[
                    torch.tensor([0, 1, 1, 1, 2]),
                    torch.tensor([1, 0, 1, 2, 1])] = 1.
            elif self.X == 3:
                structuring_element[
                    torch.tensor([1, 1, 2, 0, 1, 1, 1]),
                    torch.tensor([1, 1, 1, 1, 1, 0, 2]),
                    torch.tensor([1, 2, 1, 1, 0, 1, 1])] = 1.
            else:
                print(':(')

            self.dilation_conv = init_convolution(
                in_shape=in_shape,
                out_shape=in_shape,
                conv_weight_data=structuring_element,
                kernel_size=3,
                stride=1,
                padding=1,
                dilation=1,
                bias=False,
                requires_grad=False,
                device=self.device,
                X=self.X,
            )

        # Set up image warping
        if self.deform_images:
            self.transform_dict = {
                'elastic_factor': 0.01, 'elastic_std': 3., 'n_elastic_steps': 7, 'X': self.X,
                'apply_affine': False, 'zero_center': True, 'apply_elastic': True, 'randomize': True
            }
            self.elastic = aug.AffineElasticTransform(**self.transform_dict)
            self.onehot = aug.AssignOneHotLabels(X=X)

    def _add_noise(self, x):
        """
        Add Gaussian white noise to input image
        """
        sigma = random.uniform(0., self.noise_std)
        noise = torch.normal(
            mean=0., std=(sigma * self.max_intensity),
            size=tuple(x.shape), dtype=x.dtype, device=x.device
        )
        return x + noise

    def _augment_CSF(self, x, y, dropout_rate=0.5, noise_std=0.5):
        """
        Synthetically add noisy CSF label (helps performance in skullstripped real data)
        """
        for t in range(x.shape[2]):
            xt = x[:, :, t, ...]

            # Create masks of CSF and background
            BG_mask = torch.zeros_like(y[:, :, t, ...]).float()
            for bg in self.bg_labels:
                BG_mask[y[:, :, t, ...] == bg] = 1

            FG_mask = torch.where(y[:, :, t, ...] > 0, 1., 0.)
            CSF_mask = FG_mask.clone().float()

            # Expand GM into background to create CSF mask
            for i in range(random.randint(1, 4)):
                overlap = dilate_binary_mask(CSF_mask, self.dilation_conv) * BG_mask
                overlap *= torch.where(torch.rand_like(overlap) > random.uniform(0, 1), 1, 0)
                CSF_mask += overlap
                BG_mask -= overlap

            CSF_mask -= FG_mask

            # Add noise
            noise = torch.normal(
                mean=random.uniform(0, self.max_intensity), std=(noise_std * self.max_intensity),
                size=(CSF_mask.sum().int(),), dtype=x.dtype, device=x.device
            ).abs()

            xt[CSF_mask == 1.] = noise
            x[:, :, t, ...] = xt

        return x        

    def _convert_labels(self, x):
        """
        Given the dictionary self.IOdict, converts voxels equal to a key to
        its corresponding value
        """
        y = x.clone()
        for label in self.IOdict:
            if self.IOdict[label] != label:
                y[x == label] = self.IOdict[label]
        return y

    def _spatially_deform(self, x, y):
        """
        Calculates and applies a different, randomized elastic transform to each timepoint
        """
        # Convert y to one-hot (and save label vals)
        y_labels = y.unique()
        x, y = self.onehot([x, y])
        y = y.movedim(2, 1)

        # Apply deformation
        for t in range(x.shape[2]):
            x[:, :, t, ...], y[:, t, :, ...] = self.elastic(
                [x[:, :, t, ...].unsqueeze(2), y[:, t, :, ...].unsqueeze(1)]
            )

        # Convert label image back to int (with correct values)
        y = y_labels[y.movedim(1, 2).argmax(dim=1, keepdim=True)]

        return x, y

    def _synthesize_intensities(self, y, means_dict=None):
        """
        Create synthetic grayscale image from input label map
        """
        # Get list of foreground labels and intensity means
        labels = sorted(list(set(y.unique().tolist()) ^ set(self.bg_labels)))
        means = (
            torch.rand(len(labels)) * self.max_intensity if means_dict is None
            else [m for n, m in means_dict.items() if n not in self.bg_labels]
        )

        # Fill y
        x = torch.zeros(y.shape, dtype=torch.float, device=y.device)
        for label, m in zip(labels, means):
            x[torch.where(y == label)] = m.item()

        # Augment CSF label?
        if self.do_CSF_augment and torch.rand((1)) > 0.5:
            x = self._augment_CSF(x, y)

        return x

    def forward(self, y, img_dict=None):
        # Make sure left hemi labels match right hemi labels
        x = self._convert_labels(y)

        # Synthesize intensities
        x = self._synthesize_intensities(x, img_dict) if self.same_across_tps else torch.stack(
            [self._synthesize_intensities(xt, img_dict) for xt in x.movedim(-(self.X + 1), 0)],
            dim=-(self.X + 1)
        )

        # Other image manipulations
        x = torch.stack(
            [self.gaussian_blur(xt) for xt in x.movedim(-(self.X + 1), 0)], dim=-(self.X + 1)
        )

        if self.deform_images:
            x, y = self._spatially_deform(x, y)
        x = self._add_noise(x)
        x = min_max_norm(x, m=0., M=255.)
        return x, y


class _ResampleImage(nn.Module):
    """
    Resample input volume (tensor) to different resolution (given an input sampling factor or 
    target shape)
    """
    def __init__(self,
                 resample_factor=0.5,
                 target_shape=None,
                 X=3):
        if target_shape is None and resample_factor == 1.0:
            print('Target shape is None and resample factor is 1.0... this will do nothing..')

        self.factor = resample_factor
        self.shape = target_shape
        self.X = X

    def __call__(self, x):
        x = F.interpolate(
            x.to(torch.float), size=self.shape, scale_factor=self.factor,
            mode=('nearest' if x.dtype is torch.int
                  else 'trilinear' if self.X == 3
                  else 'bilinear' if self.X == 2
                  else 'linear')
        ).to(x.dtype)
        return x


class _ResizeLabels(nn.Module):
    """
    Class to randomly resize labels (synthetic atrophy/growth) based in an input dict of
    structures self.sdict. Has the ability to randomize the amount of atrophy and number of 
    structures w/in the input dict on the fly.
    """
    def __init__(self,
                 neighbors_dict,       # dict w/ all possible targets + neighbors (required)
                 sdict={},             # dict of structures to atrophy
                 control_bounds=[-.02, .02],
                 in_shape=None,        # dims of input volume
                 subsample=1.0,        # amt to subsample image
                 apply_dropout=False,  # flag to do dropout (dil. masks)
                 dropout_rate=0.2,     # dropout rate for dil. masks
                 randomize=True,       # flag to randomize
                 return_dict=True,     # flag to output the randomly selected labels w/ atrophy
                 X=3,                  # no. image dims
                 device=None):
        super(_ResizeLabels, self).__init__()
        self.device = 'cpu' if device is None else device

        # Parse input params
        self.neighbors_dict = neighbors_dict
        self.sdict = sdict
        self.control_bounds = control_bounds
        self.subsample = subsample
        self.n_channels = in_shape[1]
        self.dropout_rate = dropout_rate if apply_dropout else 0.
        self.randomize = randomize
        self.return_dict = return_dict
        self.X = X

        # Set up dilation convolutions
        structuring_element = torch.zeros((3,) * self.X, dtype=float)
        if self.X == 2:
            structuring_element[
                torch.tensor([0, 1, 1, 1, 2]),
                torch.tensor([1, 0, 1, 2, 1])] = 1.
        elif self.X == 3:
            structuring_element[
                torch.tensor([1, 1, 2, 0, 1, 1, 1]),
                torch.tensor([1, 1, 1, 1, 1, 0, 2]),
                torch.tensor([1, 2, 1, 1, 0, 1, 1])] = 1.
        else:
            print(':(')

        self.dilation_conv = init_convolution(
            in_shape=in_shape,
            out_shape=in_shape,
            conv_weight_data=structuring_element,
            kernel_size=3,
            stride=1,
            padding=1,
            dilation=1,
            bias=False,
            requires_grad=False,
            device=self.device,
            X=self.X,
        )

    def _apply_dropout_to_mask(self, M, dr):
        """
        Applies a dropout filter to an input mask (adds variability to the 
        label boundaries during dilation/erosion)
        """
        M[M.nonzero(as_tuple=True)] = torch.where(
            torch.rand((int(M.sum())), dtype=M.dtype, device=M.device) > dr,
            M[M.nonzero(as_tuple=True)], 0.
        )
        return M

    def _is_adjacent(self, x, label1, label2):
        """
        Determines whether or not two labels in the same volume are adjacent
        """
        mask1 = dilate_binary_mask(
            (x == label1).float(), self.dilation_conv)
        mask2 = dilate_binary_mask(
            (x == label2).float(), self.dilation_conv)
        return True if (mask1 * mask2).sum() > 0 else False

    def _shift_label_boundary(self, M_dil, M_ero, max_vol_change=None):
        """
        Moves the boundary between two labels to induce a specific amount of 
        volumetric change (M_dil will grow by M_chg, M_ero will shrink by M_chg)
        """
        M_chg = dilate_binary_mask(M_dil, self.dilation_conv) * M_ero

        # Apply dropout if necessary
        if self.dropout_rate > 0.:
            M_chg = self._apply_dropout_to_mask(M_chg, self.dropout_rate)

        # Ensure dilation mask is not inducing too much change
        if max_vol_change is not None:
            vol_change = M_chg.sum() / M_ero.sum()

            if vol_change > abs(max_vol_change):
                dr = (abs(vol_change) - abs(max_vol_change)) / vol_change
                M_chg = self._apply_dropout_to_mask(M_chg, dr)

        return M_dil + M_chg, M_ero - M_chg

    def _resize_labels(self, x, sdict):
        """
        Main function to resize the labels specified in sdict
        """
        out_dict = {}

        for label in sdict:
            # Create mask of specified adjacent neighbor labels
            nbr_list = sdict[label][1:]
            M_nbr = torch.zeros(x.shape, device=x.device).float()

            for idx, neighbor in enumerate(nbr_list):
                if not self._is_adjacent(x, label, neighbor):
                    nbr_list.pop(idx)

            # Resize target
            M_trg = (x == label).float()
            M_trg_orig = M_trg.clone()

            trg_change = sdict[label][0]
            vol_change = 0.

            it = 0
            while abs(vol_change) < abs(trg_change) and it < 100:
                change_remaining = trg_change - vol_change

                for neighbor in nbr_list:
                    M_nbr = (x == neighbor).float()
                    if trg_change < 0:
                        M_nbr, M_trg = self._shift_label_boundary(
                            M_nbr, M_trg, max_vol_change=change_remaining
                        )
                        x = torch.where(M_nbr.bool(), neighbor, x)
                    else:
                        M_trg, M_nbr = self._shift_label_boundary(
                            M_trg, M_nbr, max_vol_change=change_remaining
                        )
                        x = torch.where(M_trg.bool(), label, x)

                    vol_change = ((M_trg.sum() - M_trg_orig.sum()) / M_trg_orig.sum()).item()

                it += 1

            if debug:
                vc = vol_change
                tvc = trg_change
                print(f'label {label}: vol_change {vc:.2f}, requested {tvc:.2f} in {it} iters')

            out_dict[label] = [vol_change] + nbr_list

        return x, out_dict

    def _configure_sdict(self):
        """
        Randomize the no. structures and maximum amount of atrophy included in the sdict (allows 
        for randomization beyond class initialization)
        """
        sdict = {}

        # Get idxs of target structures to atrophy vs. others
        s_targ = set(
            random.sample(self.sdict.keys(), random.randint(1, len(self.sdict)))
        ) if len(self.sdict) > 0 else None
        s_other = random.sample(self.neighbors_dict.keys(), k=len(self.neighbors_dict))
        s_other = set(s_other) - s_targ if s_targ is not None else set(s_other)

        # Add non-targets first w/ minimal atrophy
        for label in s_other:
            amt = (
                random.uniform(self.control_bounds[0], self.control_bounds[1]) if self.randomize
                else 0.5 * sum(self.control_bounds[0] + self.control_bounds[1])
            )
            nbrs = random.sample(self.neighbors_dict[label], k=len(self.neighbors_dict[label]))
            sdict[label] = [amt] + nbrs

        # Now add target structures
        if s_targ is not None:
            for label in s_targ:
                amt = (
                    random.uniform(self.sdict[label][0], self.sdict[label][1]) if self.randomize
                    else 0.5 * sum(self.sdict[label][0] + self.sdict[label][1])
                )
                nbrs = random.sample(self.neighbors_dict[label], k=len(self.neighbors_dict[label]))
                sdict[label] = [amt] + nbrs

        return sdict

    def forward(self, x):
        sdict = self._configure_sdict()
        x, odict = self._resize_labels(x, sdict)

        return (x, odict) if self.return_dict else x


# --------------------------------------------------------------------------------------------------

def _config_synth_models(
        synth_image_lut,              # lut for labelstoimage labels
        synth_labels_lut=None,        # lut for labels to atrophy (same?)
        slist_synth_classes_config=None,  # .yaml w/ target vol. changes
        slist_neighbors_config=None,      # .yaml w/ label neighbors
        control_change_bounds=None,   # Bounds for resizing labels in control
        control_prob=0.5,             # probability to select control
        do_resample=False,            # flag to resample image res
        do_resize_labels=True,        # flag to induce atrophy
        in_shape=256,                 # expected image size
        max_perc_atrophy=None,        # max. amount of atrophy
        max_perc_lesion_growth=None,  # max amount of lesion growth
        n_image_dims=3,               # no. spatial dims
        n_timepoints=2,               # no. timepoints
        randomize=True,               # randomize params or use max values
        device=None,
        **kwargs
):
    """
    Configures all synth models given an input config dict
    """

    # Get dict of corresponding right/left label values (d = {right: left})
    if isinstance(synth_image_lut, (str, os.PathLike)):
        if not os.path.isfile(synth_image_lut):
            fatal('synth_image_lut={synth_image_lut} not a valid file')
        synth_image_lut = sf.load_label_lookup(synth_image_lut)

    lr_dict = {}
    for key, val in synth_image_lut.items():
        if 'Right' in val.name:
            lr_dict[key] = search_lut(
                synth_image_lut,
                '-'.join(['Left'] + val.name.split('-')[1:])
            )[0]
        else:
            lr_dict[key] = key

    for key in lr_dict.keys():
        if key >= 1000:
            lr_dict[key] = 3

    # Dict for I/O label correspondancy
    if synth_labels_lut is None:
        synth_labels_lut = synth_image_lut
    else:
        if isinstance(synth_labels_lut, (str, os.PathLike)):
            if not os.path.isfile(synth_labels_lut):
                fatal(f'synth_labels_lut={synth_labels_lut} not a valid file')
            synth_labels_lut = sf.load_label_lookup(synth_labels_lut)

    io_dict = {}
    for key in synth_image_lut.keys():
        if key in synth_labels_lut:
            io_dict[key] = key
        elif key >= 1000 and key <= 1035:
            io_dict[key] = 3
        elif key >= 2000 and key <= 2035:
            io_dict[key] = 42
        else:
            io_dict[key] = 0

    # Dict for atrophy induction (slists)
    slist_synth_classes_config = yaml.safe_load(open(slist_synth_classes_config))
    slist_synth_classes = {}

    for _class, sdict in slist_synth_classes_config.items():
        slist_synth_classes[_class] = {}
        if sdict is not None:
            for label, vol_change_bounds in sdict.items():
                key = search_lut(synth_image_lut, label)[0]
                slist_synth_classes[_class][key] = vol_change_bounds

    # Dict for neighboring structures
    slist_neighbors_config = yaml.safe_load(open(slist_neighbors_config))
    slist_neighbors = {}

    for label, neighbors in slist_neighbors_config.items():
        key = search_lut(synth_image_lut, label)[0]
        slist_neighbors[key] = [search_lut(synth_image_lut, neighbor)[0] for neighbor in neighbors]

    # Initialize disease classes
    synth_models = {'DiseaseClasses': {}}
    for _class, sdict in slist_synth_classes.items():
        synth_model = SynthLongitudinal(
            sdict=sdict,           # structures for atrophing (no atrophy -> set to None)
            neighbors_dict=slist_neighbors,  # all possible labels+neighbors for vol changes
            img_dict=lr_dict,      # label map seen by imagesynth (e.g., {Lwm: Lwm, Rwm: Lwm}
            seg_dict=io_dict,      # output labels to segment (e.g., combining different aparc GM)
            in_shape=in_shape,
            control_bounds=control_change_bounds,
            do_resample=do_resample,
            do_resize_labels=do_resize_labels,
            randomize=randomize,
            T=n_timepoints,
            X=n_image_dims,
            device=device,
            **kwargs
        )
        synth_models['DiseaseClasses'][_class] = synth_model

    # Add control class?
    if control_prob > 0:
        synth_models['Control'] = SynthLongitudinal(
            sdict={},
            neighbors_dict=slist_neighbors,
            img_dict=lr_dict,
            seg_dict=io_dict,
            in_shape=in_shape,
            control_bounds=control_change_bounds,
            do_resample=do_resample,
            do_resize_labels=do_resize_labels,
            randomize=randomize,
            T=n_timepoints,
            X=n_image_dims,
            device=device,
            **kwargs
        )
        synth_models['control_prob'] = control_prob

    return synth_models


def dilate_binary_mask(x, conv, n=1, dtype=None):
    dtype = x.type() if dtype is None else dtype
    if n > 0:
        for i in range(n):
            x = conv(x)
        return torch.where(x > 0., 1., 0.).type(dtype)
    else:
        return x


def erode_binary_mask(x, conv, n=1, dtype=None):
    dtype = x.type() if dtype is None else dtype
    if n > 0:
        x = torch.where(x > 0., 0., 1.).type(dtype)
        for i in range(n):
            x = conv(x)
        return torch.where(x > 0., 0., 1.)
    else:
        return x


def fatal(message):
    print(message)
    sys.exit(1)


def gaussian_kernel(sigma, X=3):
    window = np.round(sigma * 3) * 2 + 1
    center = (window - 1) / 2

    mesh = [(-0.5 * pow(torch.arange(window) - center, 2))] * X
    mesh = torch.stack(torch.meshgrid(*mesh, indexing='ij'), dim=-1)
    kernel = (
        (1 / pow(2 * torch.pi * sigma**2, 1.5))
        * torch.exp(-(pow(mesh, 2).sum(dim=-1)) / (2 * sigma ** 2))
    )

    return kernel / kernel.sum()


def init_convolution(in_shape, out_shape, conv_weight_data, kernel_size=3, stride=1, padding=1,
                     dilation=1, bias=False, X=3, device=None, requires_grad=False):
    in_channels = in_shape[1]
    out_channels = out_shape[1]
    device = 'cpu' if device is None else device

    while len(conv_weight_data.shape) < len(in_shape):
        conv_weight_data = conv_weight_data.unsqueeze(0)

    conv_fn = eval('nn.Conv%dd' % X)(
        in_channels=in_channels, out_channels=out_channels,
        kernel_size=kernel_size, stride=stride, padding=padding,
        dilation=dilation, bias=False, device=device,
    )
    conv_fn.weight.data = conv_weight_data.to(torch.float).to(device)
    conv_fn.weight.requires_grad = requires_grad
    return conv_fn


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


def search_lut(lut, string):
    """
    Returns the key corresponding to the input string=value.name
    """
    return [key for key, val in lut.items() if val.name == string]
