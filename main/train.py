import os
from os import system
import sys
import warnings
import argparse
import logging
import random
import time
import yaml
import numpy as np
import surfa as sf
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim
from torch.utils.data import DataLoader, RandomSampler

from synth_dataset import _config_datasets
from generate_atrophy import _config_synth_models
import utils
from unet4d_classifier import UCNetLong
from segmenter_synth_ucnet import SynthUCNetSegmenter

#------------------------------------------------------------------------------

def main(pargs):
    # Parse commandline args
    infer_only = pargs.infer_only
    output_dir = pargs.output_dir
    print_time = pargs.print_time
    resume_training = pargs.resume
    use_cuda = pargs.use_cuda
    
    if print_time:
        print('Start time:',  datetime.now())

    if not '--config' in sys.argv:
        print('No .yaml config file supplied, using default '
              'configs/train_base.yaml')
    config = yaml.safe_load(open(pargs.config))

    # Set up device
    device = 'cpu'
    if use_cuda:
        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            print('CUDA is unavailable, running w/ cpu')
    utils.set_seeds(config['seed'])
    torch.set_float32_matmul_precision('medium')
    torch.cuda.memory._set_allocator_settings("max_split_size_mb:128")
    
    # Create datasets
    datasets = _config_datasets(
        data_config=config['dataset'],
        aug_config=config['augmentations'],
        infer_only=False,
        device=device,
    )
    if 'train' in datasets:
        if utils.check_config(config['training'], 'steps_per_epoch'):
            n_train_samples = config['training']['steps_per_epoch']
        else:
            n_train_samples = len(datasets['train'])
            config['training']['steps_per_epoch'] = n_train_samples
    else:
        n_train_samples = None
        
    # Generate data loaders
    loaders = {}
    for key in datasets:
        def _random_DL(dataset, config, num_samples=n_train_samples):
            sampler = RandomSampler(
                dataset, replacement=True, num_samples=num_samples,
            )
            return DataLoader(dataset, sampler=sampler, **config)

        loaders[key] = (
            _random_DL(datasets[key], config['dataloader']) if key == 'train'
            else DataLoader(datasets[key], **config['dataloader'])
        )

    # Print out dataset info
    n_test = len(datasets['test']) if 'test' in datasets else 0
    n_train = len(datasets['train']) if 'train' in datasets else 0
    n_valid = len(datasets['valid']) if 'valid' in datasets else 0

    fstr = f'Train: {n_train} | Valid: {n_valid} | Test: {n_test}'
    bffr = '-' * (len(fstr) + 2)
    print(f'{bffr}\n {fstr}\n{bffr}')

    # Initialize synth model
    synth_models = _config_synth_models(
        device=device, synth_labels_lut=datasets['test'].lut,
        **config['synth']
    )
    
    # Initialize UCNet model
    do_class = (
        config['training']['do_class']
        if utils.check_config(config['training'], 'do_class')
        else True
    )
    
    n_inputs = datasets['test'].__n_input__()
    n_labels = datasets['test'].__n_class__()
    n_classes = len(synth_models) if do_class else None

    model = UCNetLong(
        in_channels=n_inputs, out_channels_unet=n_labels,
        out_channels_cnet=n_classes, do_class=do_class,
        **config['model']
    ).to(device)

    # Initialize optimizer
    freeze_cnet = config['training']['freeze_cnet'] if (
        utils.check_config(config['training'], 'freeze_cnet')
    ) else False
    freeze_unet = config['training']['freeze_unet'] if (
        utils.check_config(config['training'], 'freeze_unet')
    ) else False
    
    model_params = []
    if do_class and not freeze_cnet:
        model_params += list(model.cnet.parameters())
    if not freeze_unet:
        model_params += list(model.unet.parameters())

    if len(model_params) == 0 and not infer_only:
        utils.fatal(
            'Model has no parameters... check if both freeze_cnet and '
            'freeze_unet are True (?)'
        ) 

    optimizer = _config_optimizer(model_params, **config['optimizer'])

    # Configure segmenter
    config['training']['output_dir'] = output_dir if (
        not utils.check_config(config['training'], 'output_dir')
    ) else config['training']['output_dir']

    segmenter = SynthUCNetSegmenter(
        synth=synth_models,
        model=model,
        optimizer=optimizer,
        resume=resume_training,
        infer_only=False,
        device=device,
        n_train_samples=n_train_samples,
        **config['training']
    )

    # Run training
    for epoch in range(segmenter.current_epoch, segmenter.max_n_epochs):
        if not 'train' in loaders:
            utils.fatal('No train loader exists... something went wrong')
        segmenter._train(loader=loaders['train'])

        if 'valid' in loaders:
            segmenter._predict(loader=loaders['valid'], loss_type='valid')

        segmenter._epoch_end()
    
    # Run inference
    if 'test' in loaders:
        segmenter._predict(
            loader=loaders['test'],
            save_outputs=True,
            write_posteriors=False,
            write_targets=True,
            write_inputs=True,
        )

    if print_time:
        print('End time:',  datetime.now())

        
#------------------------------------------------------------------------------

def _config_optimizer(network_params, **config):
    optimizerID = config['_class']
    
    if 'Adam' in config['_class']:
        optimizer = eval(config['_class'])(
            params=network_params,
            betas=tuple(config['betas']),
            lr=config['lr_start'],
            weight_decay=config['weight_decay'],
        )
    elif 'SGD' in optimizerID:
        optimizer = eval(optimizerID)(
            params=network_params,
            dampening=config['dampening'],
            lr=config['lr_start'],
            momentum=config['momentum'],
            weight_decay=config['weight_decay'],
        )
    else:
        raise Exception('invalid optimizer')

    return optimizer


#------------------------------------------------------------------------------

if __name__ == "__main__":
    main(utils.parse_args())

    
