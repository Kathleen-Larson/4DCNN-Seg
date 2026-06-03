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
from collections import OrderedDict

from torch.utils.data import Dataset
from torchvision import transforms

import utils
import augmentations_longitudinal as aug


# TO-DO: update if using multimodal inputs
valid_data_config_headers = ['intensity', 'labels']


# --------------------------------------------------------------------------------------------------

class BaseDataset(Dataset):
    """
    Base class for initialization all datasets
    """
    def __init__(
            self,
            aug_config=None,
            n_input=1,
            n_class=None,
            in_lut=None,
            out_lut=None,
            merge_LR_labels=True,
            X=3,
            device=None
    ):
        # Labels lookup tables
        if in_lut is None and n_class is None:
            utils.fatal('Error in SynthDataset: must either input a label lut or n_class.')

        self.in_lut, self.lr_dict = (
            self._merge_LR_lut(in_lut) if merge_LR_labels
            else (in_lut, None)
        )
        self.out_lut = out_lut if out_lut is not None else self.in_lut

        # Parse other input params   
        self.n_input = n_input
        self.n_class = n_class if n_class is not None else len(self.out_lut)
        self.device = 'cpu' if device is None else device
        self.X = X

        # Set up data augmentations
        aug_list_full = [
            None if self.out_lut is None
            else aug.AssignOneHotLabels(label_values=[x for x in self.out_lut], X=self.X)
        ]
        aug_list_partial = aug_list_full.copy()

        if aug_config.get('_train') is not None:
            for func, config in aug_config['_train'].items():
                config['X'] = self.X

            aug_list_full = [
                None if self.out_lut is None
                else aug.AssignOneHotLabels(label_values=[x for x in self.out_lut], X=self.X)
            ] + [
                getattr(aug, func)(**aug_config['_train'][func])
                for func in aug_config['_transform_order'] if func in aug_config['_train']
            ]

        if aug_config.get('_infer') is not None:
            for func, config in aug_config['_infer'].items():
                config['X'] = self.X

            aug_list_partial = [
                None if self.out_lut is None
                else aug.AssignOneHotLabels(label_values=[x for x in self.out_lut], X=self.X)
            ] + [
                getattr(aug, func)(**aug_config['_infer'][func])
                for func in aug_config['_transform_order'] if func in aug_config['_infer']
            ]

        if aug_list_full is not None or aug_list_partial is not None:
            aug_list_full = aug_list_partial if aug_list_full is None else aug_list_full
            aug_list_partial = aug_list_full if aug_list_partial is None else aug_list_partial

            self.full_augmentations = aug.ComposeTransforms(aug_list_full)
            self.partial_augmentations = aug.ComposeTransforms(aug_list_partial)
            self.data_shape = [
                fn.patch_sz for fn in aug_list_partial if hasattr(fn, 'patch_sz')
            ][0]

        else:
            self.full_augmentations = None
            self.partial_augmentations = aug.ComposeTransforms(None)
            self.data_shape = [256] * self.X

    def __len__(self) -> int:
        if self.input_label_files is not None:
            return int(len(self.input_label_files))
        else:
            return int(len(self.input_image_files))

    def __n_input__(self) -> int:
        return self.n_input

    def __n_class__(self) -> int:
        return self.n_class

    def _merge_LR_lut(self, lut):
        def search_lut(lut, string):
            return [[key, val] for key, val in lut.items() if val.name == string][0]

        lut_dict = {}
        lr_dict = {}

        for label in list(lut.keys()):
            ref_name = lut.get(label).name
            needs_merging = ('Left' in ref_name or 'Right' in ref_name)

            name = '-'.join(ref_name.split('-')[1:]) if needs_merging else ref_name
            ref_label, ref_elem = (
                search_lut(lut, f'Left-{name}') if 'Right' in ref_name
                else [label, lut.get(label)]
            ) if needs_merging else [label, lut.get(label)]

            lr_dict[label] = ref_label
            lut_dict[ref_label] = sf.labels.LabelElement(name=name, color=ref_elem.color)

        return sf.labels.LabelLookup(lut_dict), lr_dict

    def _save_volume(self, img, outdir, outbase, idx, is_labels=False, is_onehot=False,
                     rescale=True, make_subdir=False):

        basename = '.'.join([self.outbases[idx], outbase, 'mgz'])
        outdir = os.path.join(outdir, self.outbases[idx]) if make_subdir else outdir
        os.makedirs(outdir, exist_ok=True)

        utils.save_volume(
            x=img.softmax(dim=1) if is_onehot else img,
            path=os.path.join(outdir, basename),
            label_lut=self.out_lut,
            is_labels=is_labels,
            rescale=rescale
        )


# --------------------------------------------------------------------------------------------------

class SynthDataset(BaseDataset):
    """
    Dataset for synthetic atrophy generation
    """
    def __init__(self, image_files, label_files, outbase, **kwargs):
        super().__init__(**kwargs)

        # Parse I/O filenames
        self.input_label_files = label_files
        self.synth_reference_image_files = image_files
        self.outbases = [
            '_'.join([outbase, os.path.basename(x).split('.')[0]])
            for x in self.input_label_files
        ]

    def __getitem__(self, idx):
        # Input labels (seeds for synthetic atrophy)
        invol = utils.load_volume(
            self.input_label_files[idx], is_int=True, is_slice=(True if self.X == 2 else False)
        ).unsqueeze(0)

        if self.lr_dict is not None:
            invol = utils.replace_labels(invol, self.lr_dict.values(), self.lr_dict.keys())

        # Reference image for intensity image synthesis
        refvol = (
            utils.load_volume(
                self.synth_reference_image_files[idx], is_slice=(True if self.X == 2 else False)
            ).unsqueeze(0)
            if hasattr(self, 'synth_reference_image_files')
            and self.synth_reference_image_files is not None
            else None
        )

        return ((invol, refvol), idx) if refvol is not None else (invol, idx)


class StaticDataset(BaseDataset):
    """
    Regular dataset with no synthetic atrophy (probably just debugging)
    """
    def __init__(self, image_files, label_files, outbase, **kwargs):
        super().__init__(**kwargs)

        # Parse I/O filenames
        self.input_image_files = image_files
        self.input_label_files = label_files
        self.outbases = [
            '_'.join([outbase, os.path.basename(x).split('.')[0]]) for x in self.input_image_files
        ]

    def __getitem__(self, idx):
        inputs = torch.stack([
            utils.load_volume(
                x, is_int=False, shape=192, is_slice=(True if self.X == 2 else False)
            ) for inpaths in self.input_files for x in [inpaths]
        ], dim=0).unsqueeze(0)

        labels = torch.stack([
            utils.load_volume(
                inpath, is_int=True, shape=192, is_slice=(True if self.X == 2 else False)
            ) for inpaths in self.label_files for x in [inpaths]
        ], dim=0).unsqueeze(0) if self.label_files is not None else None

        if self.lr_dict is not None:
            labels = utils.replace_labels(labels, self.lr_dict.values(), self.lr_dict.keys())

        return (inputs, idx) if labels is None else ((inputs, labels), idx)


class MultiTimepointTestDataset(BaseDataset):
    """
    Regular dataset with no ground truth labels (inference only)
    """
    def __init__(self, image_files, label_files, outbase, **kwargs):
        super().__init__(**kwargs)

        # Parse I/O
        self.input_image_files = image_files
        self.input_label_files = None

        self.outbases = [x for x in map(list, zip(*(self.input_image_files)))]
        self.outbases = [
            '_'.join([outbase, os.path.basename(x).split('.')[0]])
            for x in [y for y in map(list, zip(*(self.input_image_files)))][0]
        ]
        self.n_timepoints = len(self.input_image_files[0])

    def __getitem__(self, idx):
        inputs = torch.stack([
            utils.load_volume(
                x, is_int=False, shape=192, is_slice=(True if self.X == 2 else False)
            ) for inpaths in self.input_image_files[idx] for x in [inpaths]
        ], dim=0).unsqueeze(0)

        return (inputs, idx)


# --------------------------------------------------------------------------------------------------

def _config_datasets(
        data_config_path,      # path to csv file w/ filenames for dataloader
        input_lut_path,        # path to label lookup table for input data
        aug_config=None,       # dict w/ augmentation parameters
        do_synth=False,        # flag to configure for synth atrophy dataset or use static tps
        infer_only=False,      # flag for infer only (no training/validation cohorts)
        lut=None,              # label lookup table (lut)
        merge_LR_labels=True,  # flag to combine labels from left/right hemis into a single lut
        n_data_splits=3,       # no. cohorts (train/valid/test)
        n_image_dims=3,        # no. spatial dims
        outbase=None,          # string to add to basename of output files
        output_lut_path=None,  # path to label lookup table for output data
        randomize=False,       # flag to randomize data order
        split_ratio=0.2,       # ratio of no. valid/test to no. train
        df_has_header=False,   # flag to specify if data_config_path has header line
        device=None
):
    # Load label luts
    if input_lut_path is None:
        utils.fatal('Error: lut for input label segmentations required')

    in_lut = sf.load_label_lookup(input_lut_path)
    out_lut = in_lut if output_lut_path is None else sf.load_label_lookup(output_lut_path)

    # Read list of input images
    if not os.path.isfile(data_config_path):
        utils.fatal(f'{data_config_path} does not exist')

    input_files_df = pd.read_csv(data_config_path, header=(0 if df_has_header else None))

    if not df_has_header:
        # Check if it actually does have headers
        headers = [None] * len(input_files_df.columns)
        for y in valid_data_config_headers:
            if y in [x.lower() for x in input_files_df.loc[0]]:
                headers[[x.lower() for x in input_files_df.loc[0]].index(y)] = y

        headers = [x for x in headers if x is not None]
        if not headers:
            # None found.. exit if multiple columns, otherwise infer from do_synth or infer_only
            input_files_df.columns = (
                ['labels' if do_synth else 'intensity'] if len(input_files_df.columns) == 1
                else ['intensity'] * len(input_files_df.columns) if infer_only
                else utils.fatal('Image files DataFrame has multiple columns but no headers... '
                                 'unable to infer data types (intensity vs. labels). Please add '
                                 'headers (Intensity/Labels) to input data config file.')
            )
        else:
            # headers found in df, so add to df remove first row
            input_files_df.columns = headers
            input_files_df = input_files_df.loc[1:]

    else:
        # Check that headers are correct
        headers = [x.lower() for x in input_files_df.columns if x in valid_data_config_headers]
        if not headers:
            # No valid columns
            utils.fatals('Image files DataFrame contains no valid headers (must be "intensity" '
                         'and/or "labels")')

        if len(headers) != len(input_files_df.columns):
            # Retain only columns with valid headers
            print('Warning: ignoring image files DataFrame columns with invalid headers '
                  '{[y for y in input_files_df.columns if y not in valid_data_config_headers]}')
            input_files_df = input_files_df[[x for x in headers]]

    # Split images into separate cohorts (e.g. train/valid/test)
    n_samples = len(input_files_df)
    n_data_splits = 1 if infer_only else n_data_splits
    data_split_names = (['test'] if infer_only else ['train', 'test', 'valid'][:n_data_splits])

    if n_samples == 1:
        idxs_lists = [np.arange(0, n_samples)] * n_data_splits
    else:
        idxs = np.arange(0, n_samples)
        random.shuffle(idxs)

        x = int(np.ceil(split_ratio * n_samples))
        split_idxs = [0] + [
            n_samples - (j + 1) * x for j in reversed(range(n_data_splits - 1))
        ] + [n_samples]
        idxs_lists = [
            idxs[split_idxs[n]:split_idxs[n + 1]] for n in range(n_data_splits)
        ]

    # Create torch dataset for each cohort
    _class = (
        SynthDataset if do_synth
        else MultiTimepointTestDataset if infer_only
        else StaticDataset
    )
    datasets_dict = {}
    for n, (idxs, split_name) in enumerate(zip(idxs_lists, data_split_names)):
        datasets_dict[split_name] = _class(
            image_files=(
                np.array(input_files_df.intensity)[idxs].tolist()
                if 'intensity' in input_files_df.columns and split_name != 'train' else None
            ),
            label_files=(
                np.array(input_files_df.labels)[idxs].tolist()
                if 'labels' in input_files_df.columns else None
            ),
            aug_config=aug_config,
            n_input=1,
            in_lut=in_lut,
            out_lut=out_lut,
            outbase=outbase,
            device=device,
            X=n_image_dims,
            merge_LR_labels=merge_LR_labels,
        )

    return datasets_dict
