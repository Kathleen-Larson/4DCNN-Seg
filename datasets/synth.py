import os
import sys
import glob
import random
import torch
import torch.nn as nn
import numpy as np
import nibabel as nib
from . import transforms
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from torchvision.datasets.vision import VisionDataset



### Full synth dataset class ###
class Synth(VisionDataset):
    def __init__(
            self,
            #root: str = '/space/kale/1/users/siy0/atrophy/2d',
            range = range(0,5000),
            #out_shape: list = (256,256),
            #numinput: int = 1,
            #numclass: int = 1,
            #multiply: int = 1,
            transforms: Optional[Callable] = None,
            **kwargs
    ):
        super().__init__(root, transforms)
        self.image_file = 'subject%06d.time%d.intensity.mgz'
        self.label_file = 'subject%06d.time%d.labels.mgz'
        self.range = range
        self.out_shape = out_shape
        self.numinput = numinput
        self.numclass = numclass
        self.multiply = multiply

        self.images1 = [os.path.join(self.root, self.image_file) % (p, 1) for p in self.range]
        self.images2 = [os.path.join(self.root, self.image_file) % (p, 2) for p in self.range]
        self.labels1 = [os.path.join(self.root, self.label_file) % (p, 1) for p in self.range]
        self.labels2 = [os.path.join(self.root, self.label_file) % (p, 2) for p in self.range]

    def __getitem__(self, index: int, cpu=True, gpu=False) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is the image segmentation.
        """
        image1 = self.images1[index % len(self.images1)]
        image2 = self.images2[index % len(self.images2)]
        label1 = self.labels1[index % len(self.labels1)]
        label2 = self.labels2[index % len(self.labels2)]

        img = np.asarray(np.stack([nib.load(image1).dataobj, nib.load(image2).dataobj], 0), dtype=np.float32)
        tar = np.asarray(np.stack([nib.load(label1).dataobj, nib.load(label2).dataobj], 0), dtype=np.uint8)

        img = img[:self.numinput]
        tar = tar[:self.numinput]

        if self.transforms is not None:
            img, tar = self.transforms(img, tar, cpu=cpu, gpu=gpu)

        return img, tar, index

    def __len__(self) -> int:
        return int(len(self.images1) * self.multiply)

    def __outshape__(self) -> list:
        return self.out_shape

    def __numinput__(self) -> int:
        return self.numinput

    def __weights__(self):
        return 1

    def __numclass__(self) -> int:
        return self.numclass


    
### 3D synth data ###
def synth_3d(root='/space/kale/1/users/siy0/atrophy/3d_adni', **kwargs):
    newlabels = [0,0,1,2,3,4,0,5,6,0,7,8,9,10,11,12,13,14,15,0,0,0,0,0,0,0,16,0,17]
    numclass = 1 + max(newlabels)
    numinput = 1

    transformer = transforms.Compose([transforms.ToTensor3d(), transforms.Subsample3d(factor=[8,8,8]), transforms.ReplaceLabels(newlabels), transforms.ScaleZeroOne(), transforms.RandAffine3dPair(translations=0, rotations=0, bulk_translations=0.1, bulk_rotations=0, zooms=(0, 0)), transforms.OneHotLabels(numclass)])

    train_range=range(0,100)
    valid_range=range(100,120)
    test_range=range(120,140)
    
    train = Synth(root, range=range(0,2000), transforms=transformer,\
                  numinput=numinput, numclass=numclass, **kwargs)
    valid = Synth(root, range=range(2000,2100), transforms=transformer,\
                  numinput=numinput, numclass=numclass, **kwargs)
    tests = Synth(root, range=range(2000,2100), transforms=transformer,\
                  numinput=numinput, numclass=numclass, **kwargs)

    return train, valid, tests



### 2D Synth data ###
def synth_2d(root='/space/kale/1/users/siy0/atrophy/2d', **kwargs):
    newlabels = [0,0,1,2,3,4,0,5,6,0,7,8,9,10,11,12,13,14,15,0,0,0,0,0,0,0,16,0,17]
    numclass = 1 + max(newlabels)
    numinput = 1

      transformer = transforms.Compose([transforms.ToTensor(), transforms.Subsample2d(), transforms.ReplaceLabels(newlabels), transforms.ScaleZeroOne(), transforms.RandAffine2dPair(translations=0, rotations=0, bulk_translations=0.1, bulk_rotations=0, zooms=(0, 0)), transforms.OneHotLabels(numclass)])

    train_range=range(0,100)
    valid_range=range(100,120)
    test_range=range(120,140)
    
    train = Synth(root, range=train_range, transforms=transformer, numinput=numinput, numclass=numclass, **kwargs)
    valid = Synth(root, range=valid_range, transforms=transformer, numinput=numinput, numclass=numclass, **kwargs)
    tests = Synth(root, range=test_range, transforms=transformer, numinput=numinput, numclass=numclass, **kwargs)



    return train, valid, tests
