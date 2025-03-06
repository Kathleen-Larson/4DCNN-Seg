import os
from glob import glob
from pathlib import Path
import random

import logging
import pandas as pd
import nibabel as nib
import numpy as np
import surfa as sf
import torch

from scipy import ndimage

from torch.utils.data import Dataset
from torchvision import transforms
import freesurfer as fs

from models import augmentations as aug
from models.generate_atrophy import SynthLongitudinal


### Full synth dataset class ###

class Synth(Dataset):
    def __init__(self,
                 data_inds:list,
                 n_input:int=1,
                 n_class:int=1,
                 n_timepoint:int=2,
                 data_file=None,
                 labels_lr_dict:dict=None,
                 labels_seg:list[int]=None,
                 synth_model_dict=None,
                 base_augmentations=None,
                 full_augmentations=None,
                 lut_path:str=None,
                 X:int=3,
                 device=None,
    ):
        # Initialize
        self.n_input = n_input
        self.n_class = len(labels_seg) if labels_seg is not None else len(labels_lr_dict)
        self.n_timepoint = n_timepoint
        self.device = 'cpu' if device is None else device

        self.X = X
        self.T = 2
        
        self.data_config = data_file
        self.data_inds = data_inds
        self.lut = sf.load_label_lookup(lut_path) if lut_path is not None else None
        
        # Set up synth models
        self.labels_lr_dict = labels_lr_dict
        self.labels_seg = labels_seg
        self.synth_models_dict = synth_model_dict
        self.n_synth_classes = len(self.synth_models_dict)
        
        # Initialize augmentations
        self.base_augmentations = base_augmentations
        self.full_augmentations = full_augmentations

        # Load data
        if not os.path.isfile(data_file):
            raise ValueError(f'File {data_file} does not exist')
        if data_file is None:
            raise Exception('Must input data_file')

        df = pd.read_csv(data_file, header=None)
        self.input_files = df.iloc[self.data_inds, :].values.tolist()
        

    def __len__(self) -> int:
        return int(len(self.input_files))


    def __n_input__(self) -> int:
        return self.n_input


    def __n_class__(self) -> int:
        return self.n_class
        

    def _load_volume(self, path, dtype, shape=(256,256,256), res=1.0, orientation='RAS', is_labels=False):
        raw = sf.load_volume(str(path))
        vol = raw.conform(shape=shape, voxsize=res, orientation=orientation, dtype=dtype,
                          method='nearest' if is_labels else 'linear'
        )
        return vol


    def _convert_labels(self, img, labels):
        img_convert = img.copy()
        for i in range(len(labels)):
            img_convert[img==i] = labels[i]

        return img_convert



    def _save_output(self, img, path, dtype, is_labels=False, is_onehot=False, convert_labels=False):
        output = torch.argmax(img, dim=1).squeeze().cpu().numpy().astype(dtype) if is_onehot \
            else img.squeeze().cpu().numpy().astype(dtype)

        output = self._convert_labels(output, self.labels_seg) if is_labels and convert_labels \
            else output

        output = sf.Volume(output)
        if is_labels and self.lut is not None:  output.labels = self.lut
        output.save(path)
        

    def __getitem__(self, idx, gpu=True, cpu=False):
        inpath = self.input_files[idx][0]
        invol = self._load_volume(inpath, dtype=np.int32)
        invol = torch.tensor(invol.data).unsqueeze(0)
        
        return invol, idx




def labels_setup(lut_path:str=None,
                 perc_atrophy:float=0.,
                 perc_lesion_growth=5.,
                 include_lesions:bool=False,
                 device=None,
):
    device = 'cpu' if device is None else device

    # Label set-up
    labels_df = pd.read_csv(lut_path, sep='\s+', index_col=1, header=None)
    label_values = labels_df.iloc[:,0].values.tolist()
    label_names = labels_df.index.to_list()
    
    labels_dict = {}
    labels_lr_dict = {}
    for i in range(len(label_values)):
        labels_dict[label_names[i]] = label_values[i]
        if label_names[i].split('-')[0] == 'Right':
            label_str_lh = '-'.join(['Left'] + label_names[i].split('-')[1:])
            labels_lr_dict[label_values[i]] = label_values[label_names.index(label_str_lh)]
        else:
            labels_lr_dict[label_values[i]] = label_values[i]

    lesion = 498
    labels_dict['Lesion'] = lesion
    if include_lesions: labels_lr_dict[lesion] = lesion
            

    # Initialize possible slists (move this stuff to a separate file eventually)
    slistL = {labels_dict['Left-Hippocampus']:           [perc_atrophy,
                                                          labels_dict['Left-Lateral-Ventricle'],
                                                          labels_dict['Left-Inf-Lat-Vent']],
              labels_dict['Left-Amygdala']:              [perc_atrophy,
                                                          labels_dict['Left-Lateral-Ventricle'],
                                                          labels_dict['Left-Inf-Lat-Vent']],
              labels_dict['Left-Caudate']:               [perc_atrophy,
                                                          labels_dict['Left-Lateral-Ventricle']],
              labels_dict['Left-Thalamus']:              [perc_atrophy,
                                                          labels_dict['Left-Lateral-Ventricle']],
              labels_dict['Left-Accumbens-area']:         [perc_atrophy,
                                                           labels_dict['Left-Lateral-Ventricle']],
              labels_dict['Left-Cerebral-White-Matter']: [perc_atrophy,
                                                          labels_dict['Left-Lateral-Ventricle'],
                                                          labels_dict['Left-Inf-Lat-Vent']]
    }
    slistR = {labels_dict['Right-Hippocampus']:           [perc_atrophy,
                                                           labels_dict['Right-Lateral-Ventricle'],
                                                           labels_dict['Right-Inf-Lat-Vent']],
              labels_dict['Right-Amygdala']:              [perc_atrophy,
                                                           labels_dict['Right-Lateral-Ventricle'],
                                                           labels_dict['Right-Inf-Lat-Vent']],
              labels_dict['Right-Caudate']:               [perc_atrophy,
                                                           labels_dict['Right-Lateral-Ventricle']],
              labels_dict['Right-Thalamus']:              [perc_atrophy,
                                                           labels_dict['Right-Lateral-Ventricle']],
              labels_dict['Right-Accumbens-area']:         [perc_atrophy,
                                                            labels_dict['Right-Lateral-Ventricle']],
              labels_dict['Right-Cerebral-White-Matter']: [perc_atrophy,
                                                           labels_dict['Right-Lateral-Ventricle'],
                                                           labels_dict['Right-Inf-Lat-Vent']]
    }
    
    if include_lesions:
        slistL[labels_dict['Lesion']] = [perc_lesion_growth, labels_dict['Left-Cerebral-Cortex'],
                                         labels_dict['Left-Cerebral-White-Matter']]
        slistR[labels_dict['Lesion']] = [perc_lesion_growth, labels_dict['Right-Cerebral-Cortex'],
                                         labels_dict['Right-Cerebral-White-Matter']]
            
    slists = [slistL, slistR]
    slist_classes = ['Left', 'Right']


    # Initialize the synth models
    synth_models = {}
    
    for n, synth_class in enumerate(slist_classes):
        synth_models[synth_class] = SynthLongitudinal(labels_in=labels_lr_dict,
                                                      slist=slists[n],
                                                      input_shape=(1, 1, 256, 256, 256),
                                                      include_lesions=True if include_lesions else False,
                                                      device=device
        )
            

    # Create list of labels to segment
    labels_seg = [labels_dict['Unknown'],
                  labels_dict['Left-Cerebral-White-Matter'],
                  labels_dict['Right-Cerebral-White-Matter'],
                  labels_dict['Left-Cerebral-Cortex'],
                  labels_dict['Right-Cerebral-Cortex'],
                  labels_dict['Left-Lateral-Ventricle'],
                  labels_dict['Right-Lateral-Ventricle'],
                  labels_dict['Left-Inf-Lat-Vent'],
                  labels_dict['Right-Inf-Lat-Vent'],
                  labels_dict['Left-Thalamus'],
                  labels_dict['Right-Thalamus'],
                  labels_dict['Left-Caudate'],
                  labels_dict['Right-Caudate'],
                  labels_dict['Left-Putamen'],
                  labels_dict['Right-Putamen'],
                  labels_dict['Left-Pallidum'],
                  labels_dict['Right-Pallidum'],
                  labels_dict['Left-Hippocampus'],
                  labels_dict['Right-Hippocampus'],
                  labels_dict['Left-Amygdala'],
                  labels_dict['Right-Amygdala'],
                  labels_dict['Left-Accumbens-area'],
                  labels_dict['Right-Accumbens-area'],
                  labels_dict['WM-hypointensities'],
    ]

    return labels_lr_dict, labels_seg, synth_models



def augmentation_setup(aug_config:str=None,
                       data_types:list[str]=None,
                       label_values:list[int]=None,
                       X:int=3,
                       crop_patch_size:list[int]=160,
                       random_crop:bool=False,
                       random_aug:bool=True,
                       apply_robust_normalization:bool=True,
):
    X = 3

    ## Base transforms
    onehot = aug.AssignOneHotLabels(label_values=label_values, X=X)

    center_crop = aug.CropPatch(patch_size=crop_patch_size,
                                randomize=False,
                                X=X
    )
    norm = aug.MinMaxNorm(minim=0.,
                          maxim=1.,
                          norm_perc=[0., 0.99],
                          use_robust=True if apply_robust_normalization else False
    )
    base_augmentations = [onehot, center_crop, norm]

    
    ## Full augmentations
    if aug_config is None:
        full_augmentations = [onehot, center_crop, norm]

    else:
        df = pd.read_table(aug_config, delimiter='=', header=None)

        # Left-right flipping
        flip = aug.FlipTransform(flip_axis=X, chance=0.5)

        # Spatial deformation
        translation_bounds = float(df.loc[df.iloc[:,0]=="translation_bounds",1].item())
        rotation_bounds = float(df.loc[df.iloc[:,0]=="rotation_bounds",1].item())
        shear_bounds = float(df.loc[df.iloc[:,0]=="shear_bounds",1].item())
        scale_bounds = float(df.loc[df.iloc[:,0]=="scale_bounds",1].item())

        elastic_shape_factor = float(df.loc[df.iloc[:,0]=="elastic_shape_factor",1].item())
        elastic_std = float(df.loc[df.iloc[:,0]=="elastic_std",1].item())
        n_elastic_steps = int(df.loc[df.iloc[:,0]=="n_elastic_integration_steps",1].item())

        spatial = aug.AffineElasticTransform(translation_bounds=translation_bounds,
                                             rotation_bounds=rotation_bounds,
                                             shear_bounds=shear_bounds,
                                             scale_bounds=scale_bounds,
                                             elastic_shape_factor=elastic_shape_factor,
                                             elastic_std=elastic_std,
                                             n_elastic_steps=n_elastic_steps,
                                             apply_affine=True,
                                             apply_elastic=True,
                                             randomize=True if random_aug else False,
                                             X=X,
        )

        # Random patch
        rand_crop = aug.CropPatch(patch_size=crop_patch_size,
                                  randomize=True if random_aug else False,
                                  X=X
        )

        # Bias field
        bias_shape_factor = float(df.loc[df.iloc[:,0]=="bias_shape_factor",1].item())
        bias_std = float(df.loc[df.iloc[:,0]=="bias_std",1].item())
        bias_max = float(df.loc[df.iloc[:,0]=="bias_max",1].item())

        bias = aug.BiasField(shape_factor=bias_shape_factor,
                             max_value=bias_max,
                             std=bias_std,
                             randomize=True if random_aug else False,
                             X=X
        )

        # Gaussian noise
        noise_std = float(df.loc[df.iloc[:,0]=="noise_std",1].item())
        noise = aug.GaussianNoise(std=noise_std,
                                  randomize=True if random_aug else False
        )

        # Gamma transform
        gamma_std = float(df.loc[df.iloc[:,0]=="gamma_std",1].item())
        gamma = aug.GammaTransform(std=gamma_std,
                                   randomize=True if random_aug else False
        )
        
        full_augmentations = [flip, onehot, spatial, rand_crop, bias, noise, norm, gamma]

    base_augmentations = aug.ComposeTransforms(base_augmentations)
    full_augmentations = aug.ComposeTransforms(full_augmentations)
    return base_augmentations, full_augmentations



def get_inds(data_config:str,
             n_samples:int=None,
             n_data_splits:int=1
):
    with open(data_config, 'r') as f:
        lines = f.readlines()

    n_subjects = len(lines) if n_samples is None else n_samples
    x = int(0.2*n_subjects)
    
    all_inds = list(range(0,n_subjects))
    random.shuffle(all_inds)

    if n_data_splits == 3:
        test_inds = all_inds[:x]
        valid_inds = all_inds[x:2*x]
        train_inds = all_inds[2*x:]
        inds_lists = [train_inds, valid_inds, test_inds]

    elif n_data_splits == 2:
        test_inds = all_inds[:x]
        train_inds = all_inds[x:]
        inds_lists = [train_inds, test_inds]

    else:
        inds_lists = [[all_inds]]
        
    return inds_lists



### Call specific datasets ###
def call_dataset(input_data_files:str,
                 aug_config:str=None,
                 n_data_splits:int=1,
                 crop_patch_size=None,
                 n_subjects:int=None,
                 randomize=True,
                 include_lesions:bool=False,
                 apply_robust_normalization:bool=False,
                 device=None,
):
    # Initialize synthesis/augmentations
    lut_path = 'data_utils/samseg44_labels.ctab'
    perc_atrophy = -0.3
    perc_lesion_growth = 5.
    labels_lr_dict, labels_seg, synth_models = labels_setup(lut_path=lut_path,
                                                            perc_atrophy=perc_atrophy,
                                                            perc_lesion_growth=perc_lesion_growth,
                                                            include_lesions=include_lesions,
                                                            device=device,
    )

    base_augs, full_augs = augmentation_setup(aug_config=aug_config,
                                              label_values=labels_seg if labels_seg is not None \
                                              else [x for x in labels_lr_dict.values()],
                                              crop_patch_size=crop_patch_size,
                                              random_crop=True if aug_config is not None else False,
                                              apply_robust_normalization=apply_robust_normalization,
    )

    # Compile to generate datasets
    inds_lists = get_inds(input_data_files, n_samples=n_subjects, n_data_splits=n_data_splits)

    datasets = [None] * len(inds_lists)
    for n in range(n_data_splits):
        datasets[n] = Synth(data_inds=inds_lists[n],
                            n_input=1,
                            data_file=input_data_files,
                            labels_lr_dict=labels_lr_dict,
                            labels_seg=labels_seg,
                            synth_model_dict=synth_models,
                            base_augmentations=base_augs,
                            full_augmentations=full_augs,
                            lut_path=lut_path,
                            X=3,
                            device=device
        )

    return datasets
