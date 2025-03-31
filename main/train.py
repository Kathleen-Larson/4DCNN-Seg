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

import torch
import torch.nn as nn
import torch.optim
from torch.utils.data import DataLoader, RandomSampler

#import segmenter
from synth_dataset import _config_datasets
from generate_atrophy import _config_synth_models
import utils
from unet4d_classifier import UClassNetXD_Longitudinal
from segmenter import SynthAtrophySegmenter

#------------------------------------------------------------------------------

def main(pargs):
    # Parse commandline args
    use_cuda = pargs.use_cuda
    resume_training = pargs.resume

    if not 'config' in sys.argv:
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

    # Build synth models
    synth_models = _config_synth_models(config['synth'], device=device)
    
    # Create datasetls
    n_train_samples = None if 'steps_per_epoch' not in config['training'] \
        else config['training']['steps_per_epoch']
    datasets = _config_datasets(data_config=config['dataset'],
                                aug_config=config['augmentations'],
                                infer_only=False,
                                device=device,
    )
    if 'train' in datasets:
        n_train_samples = config['training']['steps_per_epoch'] \
            if 'steps_per_epoch' in config['training'] \
               else len(datasets['train'])
    else:
        n_train_samples = None

    # Generate data loaders
    loaders = {}
    for key in datasets:
        def _random_DL(dataset, config):
            sampler = RandomSampler(dataset, replacement=True,
                                    num_samples=n_train_samples)
            return DataLoader(dataset, sampler=sampler, **config)

        loaders[key] = _random_DL(datasets[key], config['dataloader']) \
            if key == 'train' and n_train_samples > len(datasets[key]) \
               else DataLoader(datasets[key], **config['dataloader'])

    # Initialize segmentation model + optimizer
    network = UClassNetXD_Longitudinal(
        in_channels=datasets['test'].__n_input__(),
        out_channels_unet=datasets['test'].__n_class__(),
        out_channels_cnet=len(synth_models),
        **config['network']
    ).to(device)
    optimizer = _config_optimizer(network.parameters(), **config['optimizer'])

    # Configure segmenter
    segmenter = SynthAtrophySegmenter(
        synth=synth_models,
        model=network,
        optimizer=optimizer,
        resume=resume_training,
        infer_only=False,
        device=device,
        n_train_samples=n_train_samples,
        **config['training']
    )

    # Print cohort #s info
    n_test = len(datasets['test']) if 'test' in datasets else 0
    n_train = len(datasets['train']) if 'train' in datasets else 0
    n_valid = len(datasets['valid']) if 'valid' in datasets else 0

    fstr = f'Train: {n_train} | Valid: {n_valid} | Test: {n_test}'
    bffr = '-' * (len(fstr) + 2)
    print(f'{bffr}\n {fstr}\n{bffr}')


    # Run training
    for epoch in range(segmenter.start_epoch, segmenter.max_n_epochs):
        if not 'train' in loaders:
            utils.fatal('No train loader exists... something went wrong')
        segmenter._train(loader=loaders['train'])

        if 'valid' in loaders:
            segmenter._predict(loader=loaders['valid'], loss_type='valid')

    segmenter._epoch_end()

    # Run inference
    if 'test' in loaders:
        segmenter._predict(
            loader=loaders['test'], outbase='pglands',
            save_outputs=True, write_posteriors=True,
        )

#------------------------------------------------------------------------------

def _config_optimizer(network_params, **config):
    optimizerID = config['_class']
    lr_start = config['lr_start']
    weight_decay = config['weight_decay']

    if 'Adam' in optimizerID:
        betas = tuple(config['betas'])
        optimizer = eval(optimizerID)(params=network_params,
                                      betas=betas,
                                      lr=lr_start,
                                      weight_decay=weight_decay,
        )
    elif 'SGD' in optimizerID:
        dampening = config['dampening']
        momentum = config['momentum']
        optimizer = eval(optimizerID)(params=network_params,
                                      lr=lr_start,
                                      momentum=momentum,
                                      weight_decay=weight_decay,
                                      dampening=dampening,
        )
    else:
        raise Exception('invalid optimizer')

    return optimizer


#------------------------------------------------------------------------------

if __name__ == "__main__":
    main(utils.parse_args())

    
