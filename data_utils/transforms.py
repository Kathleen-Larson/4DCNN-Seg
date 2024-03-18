import sys, math, random
import numpy as np
import numpy.random as npr

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.transforms.functional as F

import cornucopia as cc
from scipy import ndimage


class AssignOneHotLabelsND():
    def __init__(self, label_values:list=None, X:int=3):
        self.label_values = label_values
        self.X = X
        
    def __call__(self, img, seg):
        if self.label_values == None:
            self.label_values = torch.unique(torch.flatten(seg))
            
        onehot = torch.zeros(seg.shape).to(seg.device)
        if self.X == 3:
            onehot = onehot.repeat(1,len(self.label_values),1,1,1,1)
        elif self.X == 2:
            onehot = onehot.repeat(1,len(self.label_values),1,1,1)

        seg = torch.squeeze(seg)
        for i in range(len(self.label_values)):
            onehot[:,i,...] = seg==self.label_values[i]

        return img, onehot.type(torch.float32)



class BiasField():
    def __init__(self, shape:int=8, v_max:list=1, order:int=3):
        self.shape = shape if isinstance(shape, int)\
            else Exception("Invalid shape (must be int)")
        self.v_max = v_max if isinstance(v_max, int)\
            else Exception("Invalid v_max (must be int)")
        self.order = order if isinstance(order, int)\
            else Exception("Invalid order (must be int)")

        self.transform = cc.RandomMulFieldTransform(shape=shape,
                                                    vmax=v_max,
                                                    order=order,
                                                    shared=False)

    def __call__(self, img, seg):
        img = self.transform(img)
        return img, seg



class Compose(transforms.Compose):
    def __init__(self, transforms, gpuindex=1):
        super().__init__(transforms)
        self.gpuindex = gpuindex

    def __call__(self, *args, cpu=True, gpu=True, **kwargs):
        if cpu:
            for t in self.transforms[:self.gpuindex]:
                if t is not None:  args = t(*args)
        if gpu:
            for t in self.transforms[self.gpuindex:]:
                if t is not None:  args = t(*args)

        return args
    
    

class ContrastAugmentation:
    def __init__(self, gamma_std:float=0.5):
        self.stdev = gamma_std
        npr.seed()

    def _gamma_transform(self, img):
        gamma = npr.normal(0., self.stdev)
        img = img.pow(np.exp(gamma))
        return img

    def __call__(self, img, seg):
        img = self._gamma_transform(img)
        return img, seg
    


class GaussianNoise:
    def __init__(self, sigma:float=0.1):
        self.sigma = sigma
        self.transform = cc.RandomGaussianNoiseTransform(sigma=sigma)

        
    def __call__(self, img, seg):
        img = self.transform(img)
        return img, seg



class GetPatch:
    def __init__(self, patch_size:[int, list], X:int, randomize:bool=False):
        self.X = X
        self.patch_size = [patch_size] * X if isinstance(patch_size, int) else patch_size
        self.randomize = randomize
        if self.randomize:  npr.seed()

        
    def _get_center_bounds(self, in_sz, com=None):
        vol_sz = list(in_sz[-self.X:])
        patch_sz = self.patch_size
        if com is not None:
            bounds = [[com[i] - patch_sz[i]//2, com[i] + patch_sz[i]//2] for i in range(self.X)]
            for i in range(len(bounds)):
                b_start = bounds[i][0]
                if b_start < 0:  bounds[i] = [j - b_start for j in bounds[i]]
                b_end = bounds[i][1]
                if b_end > vol_sz[i]:  bounds[i] = [j - (b_end - vol_sz[i]) for j in bounds[j]]
        else:
            bounds = [((vol_sz[i] - patch_sz[i])//2, vol_sz[i] - (vol_sz[i] - patch_sz[i])//2) for i in range(self.X)]
        
        return bounds

        
    def _get_random_bounds(self, in_sz, com=None):
        vol_sz = list(in_sz[-self.X:])
        patch_sz = self.patch_size
        idx = [npr.randint(0, (vol_sz[i] - patch_sz[i])) for i in range(self.X)]
        bounds = [(idx[i], idx[i] + patch_sz[i]) for i in range(self.X)]
        return bounds


    def _get_patch(self, vols):
        crop = [None] * len(vols)
        get_bounds = self._get_random_bounds if self.randomize else self._get_center_bounds
        com = ndimage.center_of_mass(vols[1].cpu().numpy())[self.X:]
        bounds = get_bounds(vols[1].shape, [int(np.round(i)) for i in com])

        for i in range(len(vols)):
            if self.X == 2:
                h, w = bounds
                crop[i] = vols[i][..., h[0]:h[1], w[0]:w[1]]
            elif self.X >= 3:
                h, w, d = bounds
                crop[i] = vols[i][..., h[0]:h[1], w[0]:w[1], d[0]:d[1]]
        return crop


    def __call__(self, img, seg):
        img, seg = self._get_patch([img, seg])
        return img, seg



class MinMaxNorm:
    def __init__(self,
                 minim:float=0,
                 maxim:float=1,
                 **kwargs
    ):
        self.minim = minim
        self.maxim = maxim

        
    def __call__(self, img, seg):
        i_min = self.minim
        i_max = self.maxim
        o_min = torch.min(img)
        o_max = torch.max(img)

        img = (o_max - o_min) * (img - i_min) / (i_max - i_min) + o_min
        return img, seg



class RandomElasticAffineCrop:
    def __init__(self,
                 translation_bounds:[float,list]=0.0,
                 rotation_bounds:[float,list]=15,
                 shear_bounds:[float,list]=0.012,
                 scale_bounds:[float,list]=0.15,
                 max_elastic_displacement:[float,list]=0.15,
                 n_elastic_control_pts:int=5,
                 n_elastic_steps:int=0,
                 X:int=3,
                 **kwargs):

        self.X = X = X
        if isinstance(translation_bounds, list): assert len(translation_bounds) == X
        if isinstance(rotation_bounds, list): assert len(rotation_bounds) == X
        if isinstance(shear_bounds, list): assert len(shear_bounds) == X
        if isinstance(scale_bounds, list): assert len(scale_bounds) == X

        self.translations = [translation_bounds] * X \
            if isinstance(translation_bounds, float) else translation_bounds
        self.rotations = [rotation_bounds] * X \
            if isinstance(rotation_bounds, float) else rotation_bounds
        self.shears = [shear_bounds]  * X \
            if isinstance(shear_bounds, float) else shear_bounds
        self.zooms = [scale_bounds]  * X \
            if isinstance(scale_bounds, float) else scale_bounds
        self.dmax = [max_elastic_displacement] * X \
            if isinstance(max_elastic_displacement, float) else max_elastic_displacement
        self.shape = n_elastic_control_pts
        self.steps = n_elastic_steps

        self.transform = cc.RandomAffineElasticTransform(translations=self.translations,
                                                         rotations=self.rotations,
                                                         shears=self.shears,
                                                         zooms=self.zooms,
                                                         dmax=self.dmax,
                                                         shape=self.shape,
                                                         steps=self.steps,
                                                         patch=None
        )

        
    def __call__(self, img, seg):
        for b, c in zip(range(img.shape[0]), range(img.shape[1])):
            img[b, c, :], seg[b, c, :] = self.transform(img[b, c, :], seg[b, c, :])

        return img, seg



class RandomLRFlip:
    def __init__(self, chance:float=0.5):
        self.chance = chance if chance >= 0 and chance <= 1 \
            else Exception("Invalid chance (must be float between 0 and 1)")
        self.flip = cc.FlipTransform(axis=0)

        
    def __call__(self, img, seg):
        img, seg = cc.MaybeTransform(self.flip, self.chance)(img, seg)
        return img, seg



class ReplaceLabels:
    def __init__(self, labels_in:list[int], labels_out:list[int]):
        self.labels_in = labels_in
        self.labels_out = labels_out

    def __call__(self, img, seg):
        for i in range(len(self.labels_in)):
            seg[seg==self.labels_in[i]] = self.labels_out[i]
        return img, seg


    
class SubSampleND:
    def __init__(self, factors:[int,tuple]=[2,2,2], X:int=3):
        self.X = X
        self.factors = [factors] * X if isinstance(factors, int) else factors
        assert len(self.factors)==X, "Invalid factors argument (length should match number of image dims)"

    
    def __call__(self, img, seg):
        if self.X == 3:
            img = img[...,::self.factors[0],::self.factors[1],::self.factors[2]]
            seg = seg[...,::self.factors[0],::self.factors[1],::self.factors[2]]

        elif self.X == 2:
            img = img[...,::self.factors[0],::self.factors[1]]
            seg = seg[...,::self.factors[0],::self.factors[1]]
            

        return img, seg
