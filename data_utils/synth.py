import pandas as pd
import os, sys, glob, random
import torch
import torch.nn as nn
import numpy as np
import nibabel as nib

from . import transforms as t
from torchvision import transforms as tvt
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from torch.utils.data import Dataset



### Full synth dataset class ###
class Synth(Dataset):
    def __init__(self,
                 X:int,
                 data_inds:list,
                 n_input:int=1,
                 n_class:int=1,
                 data_config=None,
                 data_dir=None,
                 label_list=None,
                 convert_to_tensor=True,
                 base_augmentation=None,
                 full_augmentation=None,
                 affine_transform=None,
                 **kwargs
    ):
        self.data_inds = data_inds
        self.n_input = n_input
        self.n_class = n_class
        self.X = X
        
        self.data_config = data_config
        self.data_labels = label_list

        self.convert_to_tensor = convert_to_tensor
        self.base_augmentation = base_augmentation
        self.full_augmentation = full_augmentation
        self.aff = affine_transform
        
        file = pd.read_csv(data_config, header=None)
        image_files = [None] * len(data_inds)
        label_files = [None] * len(data_inds)
        
        for i in range(len(data_inds)):
            data_list = file.iloc[data_inds[i], :].values.tolist()
            label_files[i] = [filename for filename in data_list if filename.split('.')[-2]=='labels']
            #label_files[i] = label_files[i][0::2]

            data_list = [filename for filename in data_list if filename not in label_files[i]]
            image_files[i] = [[filename for filename in data_list if filename.split('.')[-3]==tp] \
                              for tp in sorted(list(set([filename.split('.')[-3] for filename in data_list])))]
            #image_files[i] = image_files[i][0::2]

        
        self.image_files = image_files
        self.label_files = label_files

        if len(image_files) != len(label_files):
            raise ValueError('Mismatch between number of images and number of labels')
        

    def _load_volume(self, path, data_type):
        data = nib.load(path)
        vol = data.get_fdata().astype(data_type)
        return vol

    
    def _save_output(self, img, path, dtype, is_onehot:bool=False):
        aff = np.array([[-1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1]])
        header = nib.Nifti1Header()
        nib.save(nib.Nifti1Image(img.cpu().numpy().astype(dtype), aff if self.X==3 else None, header), path)

    
    def __getitem__(self, idx, gpu=True, cpu=False):
        image_paths = self.image_files[idx]
        label_paths = self.label_files[idx]

        images_list = [[np.squeeze(self._load_volume(Path(mod_path), np.float32)) for mod_path in timepoint_paths] \
                  for timepoint_paths in image_paths]
        labels_list = [np.squeeze(self._load_volume(Path(tp_path), np.int32)) for tp_path in label_paths]

        if self.convert_to_tensor:
            images = torch.stack([torch.stack([torch.tensor(mod) for mod in tp], dim=0) for tp in images_list], dim=1)
            labels = torch.unsqueeze(torch.stack([torch.tensor(label) for label in labels_list], dim=0), dim=0)
        else:
            images = np.stack([np.stack([mod for mod in tp], axis=0) for tp in images_list], axis=1)
            labels = np.expand_dims(np.stack([label for label in labels_list], axis=0), axis=0)

        
        images = images[:,:-1,...] # only gonna deal w/ even numbers for now
        labels = labels[:,:-1,...]
        return images, labels, idx


    def __len__(self) -> int:
        return int(len(self.image_files)) #* self.multiply)

    def __n_input__(self) -> int:
        return self.n_input

    def __n_class__(self) -> int:
        return self.n_class



def augmentation_setup(aug_config:str=None,
                       X:int=3,
                       subsample_factors:int=None,
                       crop_patch_size:list[int]=224,
                       labels_in=None,
                       labels_out=None,
                       **kwargs
):
    subsample = t.SubSampleND(factors=subsample_factors, X=X) if subsample_factors is not None else None
    flip = t.RandomLRFlip(chance=0.5)
    norm = t.MinMaxNorm()
    replace_labels = t.ReplaceLabels(labels_in=labels_in, labels_out=labels_out) if labels_in is not None else None
    onehot = t.AssignOneHotLabelsND(label_values=labels_out, X=X)

    if crop_patch_size is not None:
        crop_patch_size = [crop_patch_size] * X if isinstance(crop_patch_size, int) else crop_patch_size
        center_patch = t.GetPatch(patch_size=crop_patch_size, X=X, randomize=False)
        rand_patch = t.GetPatch(patch_size=crop_patch_size, X=X, randomize=True)
    else:
        center_patch = None
        rand_patch = None
    
    if aug_config is not None:
        df = pd.read_table(aug_config, delimiter='=', header=None)

        translation_bounds = df.loc[df.iloc[:,0]=="translation_bounds",1].item()
        rotation_bounds = df.loc[df.iloc[:,0]=="rotation_bounds",1].item()
        shear_bounds = df.loc[df.iloc[:,0]=="shear_bounds",1].item()
        scale_bounds = df.loc[df.iloc[:,0]=="scale_bounds",1].item()
        max_disp = df.loc[df.iloc[:,0]=="max_elastic_displacement",1].item()
        n_cont_pts = df.loc[df.iloc[:,0]=="n_elastic_control_pts",1].item()
        n_steps = df.loc[df.iloc[:,0]=="n_elastic_steps",1].item()
        gamma_lower = df.loc[df.iloc[:,0]=="gamma_lower",1].item()
        gamma_upper = df.loc[df.iloc[:,0]=="gamma_upper",1].item()
        shape = df.loc[df.iloc[:,0]=="shape",1].item()
        v_max = df.loc[df.iloc[:,0]=="v_max",1].item()
        order = df.loc[df.iloc[:,0]=="order",1].item()
        sigma = df.loc[df.iloc[:,0]=="sigma",1].item()

        spatial = t.RandomElasticAffineCrop(translation_bounds=translation_bounds,
                                            rotation_bounds=rotation_bounds,
                                            shear_bounds=shear_bounds,
                                            scale_bounds=scale_bounds,
                                            max_elastic_displacement=max_disp,
                                            n_elastic_control_pts=int(n_cont_pts),
                                            n_elastic_steps=int(n_steps),
                                            patch_size=None,
                                            n_dims=X
        )
        contrast = t.ContrastAugmentation(gamma_std=0.5)
        bias = t.BiasField(shape=int(shape),
                           v_max=v_max,
                           order=int(order)
        )
        noise = t.GaussianNoise(sigma=sigma)

        full_augmentation = t.Compose([replace_labels, rand_patch, spatial, norm, flip, contrast, bias, noise, subsample, onehot])
        
    else:
        full_augmentation = t.Compose([replace_labels, center_patch, subsample, norm, onehot])

    base_augmentation = t.Compose([replace_labels, center_patch, subsample, norm, onehot])
    
    return full_augmentation, base_augmentation



def get_inds(data_config:str, n_samples:int=None):
    with open(data_config, 'r') as f:
        lines = f.readlines()

    n_subjects = len(lines) if n_samples is None else n_samples
    x = int(0.2*n_subjects)

    all_inds = list(range(0,n_subjects))
    random.shuffle(all_inds)
    test_inds = all_inds[:x]
    valid_inds = all_inds[x:2*x]
    train_inds = all_inds[2*x:]

    return train_inds, valid_inds, test_inds



### Call specific datasets ###
def synth_3d(data_config:str='data_utils/Synth_OASIS-2_3d.csv',
             aug_config:str=None,
             subsample:int=None,
             n_samples:int=None,
             **kwargs
):
    X = 3
    labels_in = [0,2,3,4,5,7,8,10,11,12,13,14,15,16,17,18,24,26,28,30,31,41,\
              42,43,44,46,47,49,50,51,52,53,54,58,60,62,63,77,80,85,165,258,259]
    labels_out = [0,2,3,4,4,0,0,10,11,12,13,14,15,16,17,18,0,0,0,0,0,41,42,43,43,0,0,49,50,51,52,53,54,58,0,0,0,77,0,0,0,0,0]
    labels_dict = {}
    for i in range(len(labels_in)):
        labels_dict[labels_in[i]] = labels_out[i]

    n_class = len(np.unique(labels_out))
    n_input = 1
    factor = subsample if subsample is not None else None
    aff = None

    full_augmentation, base_augmentation = augmentation_setup(aug_config=aug_config,
                                                              X=X,
                                                              data_labels=labels_out,
                                                              subsample_factors=factor,
                                                              crop_patch_size=192,
                                                              labels_in=labels_in,
                                                              labels_out=labels_out
    )
    train_inds, valid_inds, test_inds = get_inds(data_config, n_samples=n_samples)
    train = Synth(data_config=data_config,
                  data_inds=train_inds,
                  convert_to_tensor=True,
                  base_augmentation=base_augmentation,
                  full_augmentation=full_augmentation,
                  label_list=labels_out,
                  n_input=n_input,
                  n_class=n_class,
                  X=X
    )
    valid = Synth(data_config=data_config,
                  data_inds=valid_inds,
                  convert_to_tensor=True,
                  base_augmentation=base_augmentation,
                  full_augmentation=full_augmentation,
                  label_list=labels_out,
                  n_input=n_input,
                  n_class=n_class,
                  X=X
    )
    test = Synth(data_config=data_config,
                 data_inds=test_inds,
                 convert_to_tensor=True,
                 base_augmentation=base_augmentation,
                 full_augmentation=full_augmentation,
                 label_list=labels_out,
                 n_input=n_input,
                 n_class=n_class,
                 X=X
    )
    
    return train, valid, test



def synth_2d(data_config:str='data_utils/Synth_OASIS-2_2d.csv',
             aug_config:str=None,
             subsample:int=None,
             n_samples:int=None,
             **kwargs
):
    X = 2
    labels = [0,2,3,4,5,7,8,10,11,12,13,14,15,16,17,18,24,26,28,30,31,41,\
              42,43,44,46,47,49,50,51,52,53,54,58,60,62,63,77,80,85,165,258,259]
    n_class = len(np.unique(labels))
    n_input = 1
    factor = subsample if subsample is not None else None
    aff = None
    
    full_augmentation, base_augmentation = augmentation_setup(aug_config=None, #aug_config,
                                                              X=X,
                                                              data_labels=labels,
                                                              subsample_factors=factor,
                                                              crop_patch_size=192
    )
    train_inds, valid_inds, test_inds = get_inds(data_config, n_samples=n_samples)
    train = Synth(data_config=data_config,
                  data_inds=train_inds,
                  convert_to_tensor=True,
                  base_augmentation=base_augmentation,
                  full_augmentation=full_augmentation,
                  label_list=labels,
                  n_input=n_input,
                  n_class=n_class,
                  X=X
    )
    valid = Synth(data_config=data_config,
                  data_inds=valid_inds,
                  convert_to_tensor=True,
                  base_augmentation=base_augmentation,
                  full_augmentation=full_augmentation,
                  label_list=labels,
                  n_input=n_input,
                  n_class=n_class,
                  X=X
    )
    test = Synth(data_config=data_config,
                 data_inds=test_inds,
                 convert_to_tensor=True,
                 base_augmentation=base_augmentation,
                 full_augmentation=full_augmentation,
                 label_list=labels,
                 n_input=n_input,
                 n_class=n_class,
                 X=X
    )
    
    return train, valid, test
