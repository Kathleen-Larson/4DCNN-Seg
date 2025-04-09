import os
from glob import glob
from pathlib import Path
import random
import yaml

import logging
import pandas as pd
import numpy as np
import surfa as sf
import torch

from scipy import ndimage

from torch.utils.data import Dataset
from torchvision import transforms

import utils
import augmentations as aug



class SynthDataset(Dataset):
    def __init__(self,
                 image_files:list,
                 aug_config:dict=None,
                 n_input:int=1,
                 n_class:int=None,
                 n_timepoint:int=2,
                 lut=None,
                 X:int=3,
                 device=None,
    ):
        # Parse input params
        if lut is None and n_class is None:
            utils.fatal('Error in SynthDataset: must either input a label '
                        'lut or n_class.')

        self.n_input = n_input
        self.n_class = n_class if n_class is not None else len(lut)
        self.lut = lut
        self.device = 'cpu' if device is None else device
        self.X = X        

        # Parse I/O filenames
        self.input_files = image_files
        self.outbases = [os.path.splitext(os.path.basename(fname))[0]
                         for fname in self.input_files]

        # Set up data augmentations
        aug_list = [
            None if lut is None
            else aug.AssignOneHotLabels(label_values=[x for x in lut])
        ]
        aug_list += [getattr(aug, func)(**aug_config[func])
                     for func in aug_config['_transform_order']
                     if func in aug_config
        ]
        self.augmentations = aug.ComposeTransforms(aug_list)

        
    def __len__(self) -> int:
        return int(len(self.input_files))

    
    def __n_input__(self) -> int:
        return self.n_input

    
    def __n_class__(self) -> int:
        return self.n_class

    
    def _save_volume(self, img, outdir, outbase, idx,
                     is_labels=False, is_onehot=False, rescale=True,
    ):
        img = img.softmax(dim=1) if is_onehot else img
        inbase, ext = os.path.splitext(os.path.basename(self.input_files[idx]))
        path = os.path.join(outdir, inbase, outbase + ext)
        utils.save_volume(
            img, path, label_lut=self.lut,
            is_labels=is_labels, rescale=rescale
        )

        
    def __getitem__(self, idx, gpu=True, cpu=False):
        inpath = self.input_files[idx]
        invol = utils.load_volume(inpath, is_int=True).unsqueeze(0)
        return invol, idx


#------------------------------------------------------------------------------

def _config_datasets(data_config:dict,
                     aug_config:dict,
                     device=None,
                     infer_only:bool=False,
                     n_splits:int=3,
                     randomize:bool=False,
                     split_ratio:float=0.2,
):
    # Load label lut
    if not 'lut' in data_config:
        utils.fatal('Error: must include path to lut in data_config')
    lut = sf.load_label_lookup(data_config['lut'])

    # Read list of input images
    if not os.path.isfile(data_config['input_data_config']):
        utils.fatal(f'{data_config["input_data_config"]} does not exist')

    with open(data_config['input_data_config'], 'r') as f:
        image_files = [x.strip() for x in f.readlines()]
    n_images = len(image_files)
        
    # Split images into separate cohorts (e.g. train/valid/test)
    n_splits = n_splits if infer_only else data_config['n_splits']
    data_split_names = (
        ['test'] if infer_only else ['train', 'test', 'valid'][:n_splits]
    )
    if infer_only:
        idxs_lists = [np.arange(0, image_files.shape[0]).tolist()]
    else:
        random.shuffle(image_files)
        split_ratio = (
            data_config['split_ratio']
            if utils.check_config(data_config, 'split_ratio') else split_ratio
        )
        x = int(split_ratio * n_images)
        split_idxs = [0] + [
            n_images - (j+1) * x for j in reversed(range(n_splits-1))
        ] + [n_images]
        idxs_lists = [
            np.arange(split_idxs[n], split_idxs[n+1]).tolist()
            for n in range(n_splits)
        ]

    # Initialize torch dataset for each cohort
    datasets_dict = {}
    for n, (idxs, split_name) in enumerate(zip(idxs_lists, data_split_names)):
        datasets_dict[split_name] = SynthDataset(
            image_files=np.array(image_files)[idxs].tolist(),
            aug_config=aug_config, n_input=1, lut=lut, device=device
        )

    return datasets_dict



#------------------------------------------------------------------------------

