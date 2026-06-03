import os
from os import system
import sys
import time
from datetime import datetime
import sys
import warnings
import argparse
import time
import gc
import random
import yaml
import pathlib as Path
import numpy as np
import surfa as sf

import torch
import torch.optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler

import pandas as pd
import matplotlib.pyplot as plt

from datasets import _config_datasets
from atrophy_simulator import _config_synth_models
from models import UNetLong, CNetLong, FineTuneLayers
import loss_functions
import utils


# --------------------------------------------------------------------------------------------------

def main(pargs):
    # Parse commandline args
    infer_only = pargs.infer_only
    num_workers = pargs.num_workers
    output_dir = pargs.output_dir
    print_time = pargs.print_time
    resume_training = pargs.resume
    synth_off = pargs.synth_off
    use_cuda = pargs.use_cuda
    use_multiple_gpus = pargs.use_multiple_gpus

    if print_time:
        print('Start time:', datetime.now())

    # Load config
    if '--config' not in sys.argv:
        print('No .yaml config file supplied, using default configs/train_base.yaml')

    config = yaml.safe_load(open(pargs.config))

    debug_mode = config['training'].get('debug_mode')
    debug_mode = False if debug_mode is None else debug_mode

    # Set up device
    device = device2 = 'cpu'
    if use_cuda:
        if torch.cuda.is_available():
            device = torch.device('cuda', 0)
            device2 = torch.device('cuda', 1) if use_multiple_gpus else device
        else:
            print('CUDA is unavailable, running everything w/ cpu')

    print(f'using devices {device} and {device2}' if use_multiple_gpus else f'using {device} only')

    utils.set_seeds(config['seed'])
    torch.cuda.memory._set_allocator_settings("max_split_size_mb:128")
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

    # Create datasets
    do_synth = False if config.get('synth') is None or synth_off else True
    datasets = _config_datasets(
        aug_config=config['augmentations'],
        infer_only=infer_only,
        do_synth=do_synth,
        **config['dataset']
    )
    if 'train' in datasets:
        if config['training'].get('steps_per_epoch') is not None:
            n_train_samples = config['training']['steps_per_epoch']
        else:
            n_train_samples = len(datasets['train'])
            config['training']['steps_per_epoch'] = n_train_samples
    else:
        n_train_samples = None

    # Generate data loaders
    def _random_DL(dataset, config, num_samples=n_train_samples):
        sampler = RandomSampler(dataset, replacement=True, num_samples=num_samples)
        return DataLoader(dataset, sampler=sampler, **config)

    config['dataloader']['num_workers'] = (
        num_workers if num_workers is not None
        else config['dataloader'].get('num_workers')
        if config['dataloader'].get('num_workers') is not None
        else 1
    )

    loaders = {}
    for key in datasets:
        loaders[key] = (
            _random_DL(datasets[key], config['dataloader']) if key == 'train'
            else DataLoader(datasets[key], **config['dataloader']) if not debug_mode
            else _random_DL(datasets[key], config['dataloader'], num_samples=2)
        )

    # Print out dataset info
    n_test = len(datasets['test']) if 'test' in datasets else 0
    n_train = len(datasets['train']) if 'train' in datasets else 0
    n_valid = len(datasets['valid']) if 'valid' in datasets else 0

    ref_dataset = 'train' if 'train' in datasets else 'test'

    fstr = f'Train: {n_train} | Valid: {n_valid} | Test: {n_test}'
    bffr = '-' * (len(fstr) + 2)
    print(f'{bffr}\n {fstr}\n{bffr}')

    # Initialize synth models
    n_image_dims = datasets.get(ref_dataset).X

    synth_classes_config = yaml.safe_load(
        open(config.get('synth').get('slist_synth_classes_config'))
    )
    control_prob = config.get('synth').get('control_prob')
    synth_diseases = (
        ['Control'] + [x for x in synth_classes_config.keys()] if control_prob > 0
        else [x for x in synth_classes_config.keys()]
    )
    n_synth_classes = len(synth_diseases)

    synth_models = _config_synth_models(
        n_image_dims=datasets.get(ref_dataset).X,
        synth_image_lut=datasets.get(ref_dataset).in_lut,
        synth_labels_lut=datasets.get(ref_dataset).out_lut,
        device=(device2 if use_multiple_gpus else device),
        **config['synth']
    ) if config.get('synth') is not None and not synth_off else None

    # Initialize model(s)
    do_cnet = config.get('do_cnet') if config.get('do_cnet') is not None else False
    freeze_cnet = config.get('freeze_cnet') if config.get('freeze_cnet') is not None else False

    do_unet = config.get('do_unet') if config.get('do_unet') is not None else False
    freeze_unet = config.get('freeze_unet') if config.get('freeze_unet') is not None else False

    do_fine_tuning = (
        config.get('do_fine_tuning') if config.get('do_fine_tuning') is not None else False
    )

    n_inputs = datasets[ref_dataset].__n_input__()
    n_labels = datasets[ref_dataset].__n_class__()
    n_classes = n_synth_classes
    n_timepoints = (
        datasets[ref_dataset].n_timepoints if synth_off
        else config.get('synth').get('n_timepoints') if config.get('synth') is not None
        else 2
    )

    transfer_index = (
        0 if config.get('ucnet_transfer_index') is None
        else config.get('ucnet_transfer_index')
    )

    unet = UNetLong(
        in_channels=n_inputs,
        out_channels=n_labels,
        T=n_timepoints,
        X=n_image_dims,
        L=transfer_index,
        return_multiple=(True if do_cnet and not do_fine_tuning else False),
        **config['unet']
    ).to(device2 if do_cnet else device) if do_unet else None

    cnet = CNetLong(
        in_channels=unet.n_transfer_features,
        out_channels=n_labels,
        out_classes=n_classes,
        in_shape=datasets.get(ref_dataset).data_shape,
        T=n_timepoints,
        X=n_image_dims,
        L=transfer_index,
        **config['cnet']
    ).to(device) if do_cnet else None

    fine_tune_layers = FineTuneLayers(
        in_channels=unet.n_transfer_features,
        out_channels=n_labels,
        n_skip_channels=unet.n_starting_features,
        T=n_timepoints,
        X=n_image_dims,
        return_multiple=(True if do_cnet else False),
        **config['fine_tune_layers']
    ).to(device) if do_fine_tuning else None

    # Load checkpoints?
    load_unet = False
    resume_unet_path = os.path.join(output_dir, 'model_last_unet.pth')
    unet_state_path = (
        config.get('unet_state_path') if config.get('unet_state_path') is not None and do_unet
        else resume_unet_path if (resume_training or infer_only) and do_unet
        else None
    )

    load_cnet = False
    resume_cnet_path = os.path.join(output_dir, 'model_last_cnet.pth')
    cnet_state_path = (
        config.get('cnet_state_path') if config.get('cnet_state_path') is not None and do_cnet
        else resume_cnet_path if (resume_training or infer_only) and do_cnet
        else None
    )

    load_fine_tune = False
    resume_fine_tune_path = os.path.join(output_dir, 'model_last_fine_tune.pth')
    fine_tune_state_path = (
        config.get('fine_tune_state_path')
        if config.get('fine_tune_state_path') is not None and do_fine_tuning
        else resume_fine_tune_path if (resume_training or infer_only) and do_fine_tuning
        else None
    )

    if unet_state_path is not None:
        if unet_state_path != resume_unet_path:
            unet_symlink_path = os.path.join(output_dir, 'pretrained_unet.pth')
            if os.path.islink(unet_symlink_path):
                os.remove(unet_symlink_path)
            os.symlink(os.path.abspath(unet_state_path), unet_symlink_path)
            unet_checkpoint_path = unet_symlink_path
        elif do_unet and not freeze_unet and infer_only:
            unet_checkpoint_path = os.path.join(output_dir, 'model_last_unet.pth')
        else:
            unet_checkpoint_path = resume_unet_path

        load_unet = True
        unet_checkpoint = torch.load(unet_checkpoint_path)
        unet.load_state_dict(unet_checkpoint['model_state'])
        unet_epoch = unet_checkpoint['epoch']
    else:
        unet_epoch = 0

    if cnet_state_path is not None:
        if cnet_state_path != resume_cnet_path:
            cnet_symlink_path = os.path.join(output_dir, 'pretrained_cnet.pth')
            if os.path.exists(cnet_symlink_path):
                os.remove(cnet_symlink_path)
            os.symlink(os.path.abspath(cnet_state_path), cnet_symlink_path)
            cnet_checkpoint_path = cnet_symlink_path
        elif do_cnet and not freeze_cnet and infer_only:
            cnet_checkpoint_path = os.path.join(output_dir, 'model_last_cnet.pth')
        else:
            cnet_checkpoint_path = resume_cnet_path

        load_cnet = True
        cnet_checkpoint = torch.load(cnet_checkpoint_path)
        cnet.load_state_dict(cnet_checkpoint['model_state'])
        cnet_epoch = cnet_checkpoint['epoch']
    else:
        cnet_epoch = 0

    if fine_tune_state_path is not None:
        if fine_tune_state_path != resume_fine_tune_path:
            fine_tune_symlink_path = os.path.join(output_dir, 'pretrained_fine_tune.pth')
            if os.path.exists(fine_tune_symlink_path):
                os.remove(fine_tune_symlink_path)
            os.symlink(os.path.abspath(fine_tune_state_path), fine_tune_symlink_path)
            fine_tune_checkpoint_path = fine_tune_symlink_path
        elif do_fine_tuning and infer_only:
            fine_tune_checkpoint_path = os.path.join(output_dir, 'model_last_fine_tune.pth')
        else:
            fine_tune_checkpoint_path = resume_fine_tune_path

        load_fine_tune = True
        fine_tune_checkpoint = torch.load(fine_tune_checkpoint_path)
        fine_tune_layers.load_state_dict(fine_tune_checkpoint['model_state'])
        fine_tune_epoch = fine_tune_checkpoint['epoch']
    else:
        fine_tune_epoch = 0    

    # Configure optimizer(s)
    unet_optimizer = (
        _config_optimizer(unet.parameters(), **config['optimizer']) if do_unet and not freeze_unet
        else None
    )

    cnet_optimizer = (
        _config_optimizer(cnet.parameters(), **config['optimizer']) if do_cnet and not freeze_cnet
        else None
    )

    fine_tune_optimizer = (
        _config_optimizer(fine_tune_layers.parameters(), **config['optimizer']) if do_fine_tuning
        else None
    )

    # Configure wrapper for segmentation/classification
    output_dir = config.get('output_dir') if output_dir is None else output_dir

    start_epoch = (
        0 if not resume_training and not infer_only
        else fine_tune_epoch if load_fine_tune
        else max(unet_epoch, cnet_epoch) if load_unet and load_cnet
        else unet_epoch if do_unet and not do_cnet and load_unet
        else cnet_epoch if do_cnet and not do_unet and load_cnet
        else 0
    )
    start_step = (
        0 if not resume_training and not infer_only
        else fine_tune_checkpoint['step'] if start_epoch == fine_tune_epoch and load_fine_tune
        else unet_checkpoint['step'] if start_epoch == unet_epoch and load_unet
        else cnet_checkpoint['step'] if start_epoch == cnet_epoch and load_cnet
        else 0
    )
    start_train_loss = (
        None if not resume_training and not infer_only
        else fine_tune_checkpoint['train_loss'] if start_epoch == fine_tune_epoch and load_fine_tune
        else unet_checkpoint['train_loss'] if start_epoch == unet_epoch and load_unet
        else cnet_checkpoint['train_loss'] if start_epoch == cnet_epoch and load_cnet
        else None
    )
    start_valid_loss = (
        None if not resume_training and not infer_only
        else fine_tune_checkpoint['valid_loss'] if start_epoch == fine_tune_epoch and load_fine_tune
        else unet_checkpoint['valid_loss'] if start_epoch == unet_epoch and load_unet
        else unet_checkpoint['valid_loss'] if start_epoch == cnet_epoch and load_cnet
        else None
    )

    trainer = SynthUCNet(
        unet=unet,
        cnet=cnet,
        fine_tune_layers=fine_tune_layers,
        freeze_unet=freeze_unet,
        freeze_cnet=freeze_cnet,
        unet_optimizer=unet_optimizer,
        cnet_optimizer=cnet_optimizer,
        fine_tune_optimizer=fine_tune_optimizer,
        device=device,
        device2=device2,
        synthesizer=synth_models,
        data_splits=[x for x in loaders.keys()],
        infer_only=infer_only,
        n_train_samples=n_train_samples,
        output_dir=output_dir,
        resume=resume_training,
        start_epoch=start_epoch,
        start_step=start_step,
        start_train_loss=start_train_loss,
        start_valid_loss=start_valid_loss,
        synth_classes=synth_diseases,
        **config['training']
    )

    # Run training
    if not infer_only:
        for epoch in range(trainer.current_epoch, trainer.max_n_epochs):
            if 'train' not in loaders:
                utils.fatal('No train loader exists... something went wrong')
            trainer._train(loader=loaders['train'])

            if 'valid' in loaders:
                trainer._predict(loader=loaders['valid'], loss_type='valid')

            trainer._epoch_end()

        # Generate loss figures
        if trainer.train_log is not None and loaders.get('train') is not None:
            trainer._plot_loss(loss_type='train')

        if trainer.valid_log is not None and loaders.get('valid') is not None:
            trainer._plot_loss(loss_type='valid')

    # Run inference
    if 'test' in loaders:
        trainer._predict(
            loader=loaders['test'],
            save_outputs=True,
            write_posteriors=False,
            write_targets=True,
            write_inputs=True
        )

    if print_time:
        print('End time:', datetime.now())


# --------------------------------------------------------------------------------------------------

def _config_optimizer(model_params, **config):
    optimizerID = config.get('_class')

    if 'Adam' in config['_class']:
        optimizer = eval(f'torch.optim.{optimizerID}')(
            params=model_params,
            betas=tuple(config['betas']),
            lr=config['lr_start'],
            weight_decay=config['weight_decay'],
        )
    elif 'SGD' in optimizerID:
        optimizer = eval(f'torch.optim.{optimizerID}')(
            params=model_params,
            dampening=config['dampening'],
            lr=config['lr_start'],
            momentum=config['momentum'],
            weight_decay=config['weight_decay'],
        )
    else:
        raise Exception('invalid optimizer')

    return optimizer


# --------------------------------------------------------------------------------------------------

class SynthUCNet:
    def __init__(
            self,
            synthesizer,
            loss_funcs,
            synth_classes,
            unet=None,
            cnet=None,
            fine_tune_layers=None,
            unet_optimizer=None,
            cnet_optimizer=None,
            fine_tune_optimizer=None,
            data_splits=None,
            debug_mode=False,
            freeze_cnet=False,
            freeze_unet=False,
            infer_only=False,
            loss_weights=None,
            max_n_epochs=None,
            max_n_steps=None,
            n_train_samples=None,
            optimizer_config=None,
            output_dir=None,
            print_loss_every=None,
            resume=False,
            save_model_every=None,
            save_outputs_every=None,
            start_epoch=0,
            start_step=-1,
            start_train_loss=None,
            start_valid_loss=None,
            steps_per_epoch=None,
            switch_loss_weights_after=None,
            take_abs_change=False,
            device2=None,
            device=None,
            **kwargs
    ):
        # Parse config args
        self.device2 = 'cpu' if device2 is None else device2
        self.device = 'cpu' if device is None else device

        self.unet = None if unet is None else unet.to(self.device)
        self.cnet = None if cnet is None else cnet.to(self.device2)

        if self.unet is None and self.cnet is None:
            utils.arg_error('must provide either the segmenter (unet) or classifier (cnet) (or '
                            'both) to SynthUCNet')

        self.fine_tune_layers = (
            None if fine_tune_layers is None else fine_tune_layers.to(self.device)
        )

        self.unet_optimizer = unet_optimizer
        self.cnet_optimizer = cnet_optimizer
        self.fine_tune_optimizer = fine_tune_optimizer

        opts = [self.unet_optimizer, self.cnet_optimizer, self.fine_tune_optimizer]

        if not infer_only and all(opt is None for opt in opts):
            utils.arg_error('must provide at least one optimizer to SynthUCNet if infer_only is '
                            'False')

        self.synthesizer = synthesizer
        self.synth_classes = synth_classes

        self.freeze_cnet = freeze_cnet
        self.freeze_unet = freeze_unet
        self.infer_only = infer_only
        self.max_n_epochs = max_n_epochs
        self.max_n_steps = max_n_steps
        self.output_dir = output_dir
        self.save_outputs_every = save_outputs_every
        self.steps_per_epoch = steps_per_epoch
        self.take_abs_change = take_abs_change

        self.current_epoch = start_epoch
        self.current_step = start_step
        self.current_epoch_step = -1

        # Losses
        self.seg_loss = None
        self.class_loss = None
        self.unet_change_loss = None

        self.seg_loss_kwargs = loss_funcs.get('seg')
        if self.seg_loss_kwargs is not None:
            self.seg_loss = eval(f'loss_functions.{self.seg_loss_kwargs.get("name")}')
            self.seg_loss_kwargs.pop('name')

        self.class_loss_kwargs = loss_funcs.get('class')
        if self.class_loss_kwargs is not None:
            self.class_loss = eval(f'loss_functions.{self.class_loss_kwargs.get("name")}')
            self.class_loss_kwargs.pop('name')

        self.unet_change_loss_kwargs = loss_funcs.get('unet_change')
        if self.unet_change_loss_kwargs is not None:
            self.unet_change_loss = eval(
                f'loss_functions.{self.unet_change_loss_kwargs.get("name")}'
            )
            self.unet_change_loss_kwargs.pop('name')

        self.multiple_losses = (
            (self.seg_loss is not None)
            + (self.class_loss is not None)
            + (self.unet_change_loss is not None)
        ) > 1

        if loss_weights is not None and len(loss_weights) > 1:
            self.loss_weights = loss_weights
            self.seg_loss_kwargs['weight'] = self.loss_weights[0][0]
            self.class_loss_kwargs['weight'] = self.loss_weights[0][1]
            self.unet_change_loss_kwargs['weight'] = self.loss_weights[0][2]

            if switch_loss_weights_after is None:
                utils.arg_error('Must specify switch_loss_weights_after if using multiple loss '
                                'weights')
            else:
                self.switch_loss_weights_after = switch_loss_weights_after
        else:
            self.switch_loss_weights_after = None

        self.train_loss = (
            start_train_loss if start_train_loss is not None
            else {'last': None, 'best': None}
        ) if data_splits is not None and 'train' in data_splits or data_splits is None else None

        self.valid_loss = (
            start_valid_loss if start_valid_loss is not None
            else {'last': None, 'best': None}
        ) if data_splits is not None and 'valid' in data_splits or data_splits is None else None

        # Configure training specific args
        if not infer_only:
            # Number of steps per epoch
            if self.steps_per_epoch is None and n_train_samples is None:
                utils.arg_error('Must specify either steps_per_epoch and/or n_train_samples.')

            if self.steps_per_epoch is None:
                self.steps_per_epoch = n_train_samples
            elif n_train_samples is not None and self.steps_per_epoch != n_train_samples:
                print(f'Mismatch between steps_per_epoch={self.steps_per_epoch} '
                      f'n_train_samples={n_train_samples}, using n_train_samples')
                self.steps_per_epoch = n_train_samples

            # Maximum number of steps
            if self.max_n_epochs is None and self.max_n_steps is None:
                utils.fatal('Error initializing SynthUCNetSegmenter: must specify either '
                            'max_n_steps or max_n_epochs')

            if self.max_n_epochs is None:
                self.max_n_epochs = self.max_n_steps // self.steps_per_epoch
            elif self.max_n_steps is None:
                self.max_n_steps = self.max_n_epochs * self.steps_per_epoch

            if self.max_n_epochs * self.steps_per_epoch < self.max_n_steps:
                print(f'Warning: max_n_steps set to {self.max_n_steps}, but training will exit '
                      f'after max_n_epochs={max_n_epochs}, '
                      f'({self.max_n_epochs * self.steps_per_epoch} steps).')

            elif self.max_n_epochs * self.steps_per_epoch > self.max_n_steps:
                print(f'Warning: max_n_epochs set to {self.max_n_epochs}, but training will exit '
                      f'after max_n_steps={max_n_steps} '
                      f'({self.max_n_steps // self.steps_per_epoch} epochs).')

            # Frequency of model outputs
            if self.steps_per_epoch < 10 and self.max_n_epochs < 10:
                self.print_loss_every = 1
                self.save_model_every = None
            else:
                self.print_loss_every = (
                    print_loss_every if print_loss_every is not None
                    else self.steps_per_epoch // 10
                )
                self.save_model_every = (
                    save_model_every if save_model_every is not None
                    else self.max_n_epochs // 10
                )

        # Set up output logs
        if self.output_dir is not None and not debug_mode:
            if not os.path.isdir(self.output_dir):
                os.makedirs(self.output_dir)

            self.model_output = os.path.join(self.output_dir, 'model')
            self.test_log = (
                os.path.join(self.output_dir, 'test_log.txt')
                if data_splits is not None and 'test' in data_splits or data_splits is None
                else None
            )
            self.train_log = (
                os.path.join(self.output_dir, 'train_log.txt')
                if ((data_splits is not None and 'train' in data_splits) or data_splits is None)
                and not infer_only else None
            )
            self.valid_log = (
                os.path.join(self.output_dir, 'valid_log.txt')
                if ((data_splits is not None and 'valid' in data_splits) or data_splits is None)
                and not infer_only else None
            )

            print(f'train_log: {self.train_log}')
            print(f'valid_log: {self.valid_log}')
            print(f'test_log: {self.test_log}')

            log_str = ''
            log_str += 'seg_loss ' if self.seg_loss is not None else ''
            log_str += 'unet_change_loss ' if self.unet_change_loss is not None else ''
            log_str += 'class_loss ' if self.class_loss is not None else ''
            log_str += 'total_loss ' if self.multiple_losses else ''

            utils.init_text_file(
                self.test_log, f'FileId {log_str}',
                check_if_exists=(True if resume else False)
            )
            utils.init_text_file(
                self.train_log, f'Epoch Step {log_str}',
                check_if_exists=(True if resume else False)
            )
            utils.init_text_file(
                self.valid_log, f'Epoch {log_str}',
                check_if_exists=(True if resume else False)
            )
            if not infer_only:
                sf.system.run(f'head -n1 {self.train_log}')

        else:
            self.model_output = None
            self.test_log = None
            self.train_log = None
            self.valid_log = None

    def _train(self, loader):
        N = len(loader)
        loss_avg = 0.

        if self.unet is not None:
            self.unet.train() if not self.freeze_unet else self.unet.eval()
            self.unet.zero_grad()

        if self.cnet is not None:
            self.cnet.train() if not self.freeze_cnet else self.cnet.eval()
            self.cnet.zero_grad()

        augmentations = loader.dataset.partial_augmentations

        for y, idx in loader:
            self.current_step += 1
            self.current_epoch_step += 1

            # Synthesize images from input volume (?)
            if self.synthesizer is not None:
                control_prob = self.synthesizer.get('control_prob')

                synth_class, synth_model = (
                    ('Control', self.synthesizer.get('Control'))
                    if control_prob is not None and random.uniform(0., 1.) < control_prob
                    else random.choice(list(self.synthesizer['DiseaseClasses'].items()))
                )
                X, y = synth_model(
                    [yi.to(synth_model.device) for yi in y] if isinstance(y, (list, tuple))
                    else [y.to(synth_model.device)]
                )

            X, y = augmentations([X, y]) if augmentations is not None else y

            """
            self._save_change_map(
                utils.replace_labels(
                    torch.argmax(y.diff(dim=2), dim=1, keepdims=True),
                    labels_out=[x for x in loader.dataset.out_lut.keys()],
                    labels_in=[x for x in np.arange(0, len(loader.dataset.out_lut.keys()))]
                ),
                loader.dataset, 'target', idx, save_dir='examples'
            )
            """
            """
            self._save_model_outputs(
                dataset=loader.dataset, idx=idx, save_dir='examples_OASIS1_CSFaugments_v3',
                data_dict={
                    'Input': X,
                    'Target': utils.replace_labels(
                        torch.argmax(y, dim=1, keepdims=True),
                        labels_out=[x for x in loader.dataset.out_lut.keys()],
                        labels_in=[x for x in np.arange(0, len(loader.dataset.out_lut.keys()))]
                    )
                }
            )
            print(synth_class, loader.dataset.outbases[idx])
            if self.current_step == 5:
                exit()
            """
            # Run model
            if self.unet_optimizer is not None:
                self.unet_optimizer.zero_grad()
            if self.cnet_optimizer is not None:
                self.cnet_optimizer.zero_grad()
            if self.fine_tune_optimizer is not None:
                self.fine_tune_optimizer.zero_grad()

            if self.fine_tune_layers is not None:
                unet_output_initial, skip_conn = self.unet(X.to(self.device))
                unet_logits, _ = self.fine_tune_layers(unet_output_initial, skip_conn)

            else:
                unet_output = (
                    self.unet(X.to(self.device)) if self.unet is not None
                    else y.to(self.device2)
                )
                if isinstance(unet_output, (list, tuple)) and len(unet_output) > 1:
                    unet_logits, cnet_input = unet_output
                else:
                    unet_logits = cnet_input = unet_output

            cnet_logits, _ = (
                self.cnet(unet_output_initial.to(self.device2)) if self.cnet is not None
                else (None, None)
            )

            # Compute losses
            loss = 0.
            if self.seg_loss is not None:
                loss_seg = self.seg_loss(unet_logits, y, **self.seg_loss_kwargs)
                loss += loss_seg.to(self.device2)
            if self.unet_change_loss is not None:
                loss_unet_change = self.unet_change_loss(
                    unet_logits, y, **self.unet_change_loss_kwargs
                )
                loss += loss_unet_change.to(self.device2)
            if self.class_loss is not None:
                loss_class = self.class_loss(
                    cnet_logits, synth_class, self.synth_classes, **self.class_loss_kwargs
                )
                loss += loss_class
                loss_avg += loss.item()
            """
            # Write data?
            y = utils.replace_labels(
                torch.argmax(y, dim=1, keepdims=True),
                labels_out=[x for x in loader.dataset.out_lut.keys()],
                labels_in=[x for x in np.arange(0, len(loader.dataset.out_lut.keys()))]
            )
            seg = utils.replace_labels(
                torch.argmax(torch.softmax(unet_logits, dim=1), dim=1, keepdims=True),
                labels_out=[x for x in loader.dataset.out_lut.keys()],
                labels_in=[x for x in np.arange(0, len(loader.dataset.out_lut.keys()))]
            )
            print(f'Writing data for {loader.dataset.outbases[idx]}')
            self._save_model_outputs(
                dataset=loader.dataset, idx=idx, save_dir='testing_include_cerebellum',
                data_dict={'Input': X, 'Target': y, 'Posteriors': None, 'Output': seg}
            )
            breakpoint()
            """
            """
            fstr = f'{(self.current_epoch + 1):>5}, {(self.current_epoch_step + 1):>4},'
            if self.multiple_losses:
                fstr += f' {loss_seg.item():>.4f},' if self.seg_loss is not None else ''
                fstr += f' {loss_class.item():>.4f},' if self.class_loss is not None else ''
            fstr += f' {loss.item():>.4f}, ({synth_class}),'
            print(fstr, cnet_logits.data)

            if (self.current_epoch_step + 1) % 10 == 0:
                breakpoint()
            """
            if (self.current_epoch_step + 1) % self.print_loss_every == 0:
                fstr = f'{(self.current_epoch + 1):>5} {(self.current_epoch_step + 1):>4}'
                if self.multiple_losses:
                    fstr += f' {loss_seg.item():>.4f}' if self.seg_loss is not None else ''
                    fstr += f' {loss_unet_change.item():>.4f}' \
                        if self.unet_change_loss is not None else ''
                    fstr += f' {loss_class.item():>.4f}' if self.class_loss is not None else ''
                fstr += f' {loss.item():>.4f}'
                print(fstr)

                if self.train_log is not None:
                    with open(self.train_log, 'a') as f:
                        f.write(f'{fstr}\n')

            loss.backward()

            if self.unet_optimizer is not None:
                self.unet_optimizer.step()
            if self.cnet_optimizer is not None:
                self.cnet_optimizer.step()
            if self.fine_tune_optimizer is not None:
                self.fine_tune_optimizer.step()

        # Store loss at end of epoch
        loss_avg /= N
        self.train_loss['last'] = loss_avg

        if self.train_loss['best'] is None or self.train_loss['best'] > self.train_loss['last']:
            self.train_loss['best'] = self.train_loss['last']

    def _predict(self,
                 loader,
                 loss_type='test',
                 save_dir=None,
                 save_outputs=False,
                 write_inputs=True,
                 write_targets=True,
                 write_posteriors=False):
        N = len(loader)
        loss_avg = 0.
        accuracy = 0
        avg_seg_loss = 0. if self.seg_loss is not None else None

        if loss_type not in ['valid', 'test']:
            utils.fatal(
                f'Error in segmenter._predict: {loss_type} not a valid input '
                'for loss type!'
            )
        log_output = eval(f'self.{loss_type}_log')

        if self.unet is not None:
            self.unet.eval()
            self.unet.zero_grad()

        if self.cnet is not None:
            self.cnet.eval()
            self.cnet.zero_grad()

        if self.fine_tune_layers is not None:
            self.fine_tune_layers.eval()
            self.cnet.zero_grad()

        augmentations = loader.dataset.partial_augmentations

        for y, idx in loader:
            if self.synthesizer is not None:
                control_prob = self.synthesizer.get('control_prob')
                synth_class, synth_model = (
                    ('Control', self.synthesizer.get('Control'))
                    if control_prob is not None and random.uniform(0., 1.) < control_prob
                    else random.choice(list(self.synthesizer['DiseaseClasses'].items()))
                )

                X, y = synth_model(
                    [yi.to(synth_model.device) for yi in y] if isinstance(y, (list, tuple))
                    else [y.to(synth_model.device)]
                )

            X, y = augmentations([X, y]) if augmentations is not None else y

            # Run model
            with torch.no_grad():
                if self.fine_tune_layers is not None:
                    unet_output_initial, skip_conn = self.unet(X.to(self.device))
                    unet_logits, _ = self.fine_tune_layers(unet_output_initial, skip_conn)
                    # fine_tune_input = self.unet(X.to(self.device))
                    # unet_logits, cnet_input = self.fine_tune_layers(fine_tune_input)

                else:                    
                    unet_output = (
                        self.unet(X.to(self.device)) if self.unet is not None
                        else y.to(self.device2)
                    )
                    if isinstance(unet_output, (list, tuple)) and len(unet_output) > 1:
                        unet_logits, cnet_input = unet_output
                    else:
                        unet_logits = cnet_input = unet_output

                cnet_logits, change_map = (
                    self.cnet(unet_output_initial.to(self.device2)) if self.cnet is not None
                    else (None, None)
                )

            # Compute losses
            loss = 0.

            if self.seg_loss is not None:
                loss_seg = self.seg_loss(unet_logits, y, **self.seg_loss_kwargs)
                loss += loss_seg.to(self.device2)
                avg_seg_loss += loss_seg.item()
            if self.unet_change_loss is not None:
                loss_unet_change = self.unet_change_loss(
                    unet_logits, y, **self.unet_change_loss_kwargs
                )
                loss += loss_unet_change.to(self.device2)
            if self.class_loss is not None:
                loss_class = self.class_loss(
                    cnet_logits, synth_class, self.synth_classes, **self.class_loss_kwargs
                )
                loss += loss_class
            loss_avg += loss.item()

            # Log if classifier is correct
            if self.class_loss is not None:
                correct_idx = [self.synth_classes.index(x) for x in [synth_class]]
                out_idx = torch.argmax(torch.softmax(cnet_logits, dim=1), dim=1)
                accuracy += torch.tensor(
                    [x == y for x, y in zip(out_idx, correct_idx)]
                ).sum().item()

            # Write outputs?
            if loss_type == 'test':
                if save_outputs:
                    y = utils.replace_labels(
                        torch.argmax(y, dim=1, keepdims=True),
                        labels_out=[x for x in loader.dataset.out_lut.keys()],
                        labels_in=[x for x in np.arange(0, len(loader.dataset.out_lut.keys()))]
                    )

                    posts = torch.softmax(unet_logits, dim=1)

                    seg = utils.replace_labels(
                        torch.argmax(posts, dim=1, keepdims=True),
                        labels_out=[x for x in loader.dataset.out_lut.keys()],
                        labels_in=[x for x in np.arange(0, len(loader.dataset.out_lut.keys()))]
                    )
                    self._save_model_outputs(
                        dataset=loader.dataset, idx=idx,
                        save_dir=f'model_outputs_epoch{self.current_epoch}',
                        make_subdir=True,
                        data_dict={
                            'Input': X if write_inputs else None,
                            'Target': y if write_targets else None,
                            'Posteriors': posts if write_posteriors else None,
                            'Output': seg
                        },
                    )

                if log_output is not None:
                    fstr = f'{loader.dataset.outbases[idx]} {synth_class} '
                    if self.cnet is not None:
                        choice = self.synth_classes[F.softmax(cnet_logits, dim=1).argmax()]
                        fstr += f'{choice} '
                    if self.multiple_losses:
                        fstr += (
                            f'{loss_seg.item():>.4f} ' if self.seg_loss is not None else ''
                        )
                        fstr += (
                            f'{loss_unet_change.item():>.4f} ' if self.unet_change_loss is not None
                            else ''
                        )
                        fstr += (
                            f'{loss_class.item():>.4f} ' if self.class_loss is not None else ''
                        )
                    fstr += f'{loss.item():>.4f}'
                    with open(log_output, 'a') as f:
                        f.write(f'{fstr}\n')
                    print(fstr)

        # Store loss at end of epoch
        loss_avg /= N
        accuracy /= N

        if self.seg_loss is not None:
            avg_seg_loss /= N

        fstr = ''
        if self.multiple_losses:
            fstr += f'{loss_seg.item():>.4f} ' if self.seg_loss is not None else ''
            fstr += f'{loss_unet_change.item():>.4f} ' if self.unet_change_loss is not None else ''
            fstr += f'{loss_class.item():>.4f} ' if self.class_loss is not None else ''

        fstr += f'{loss.item():>.4f}'

        if loss_type == 'valid':
            self.valid_loss['last'] = loss_avg
            fstr = f'{(self.current_epoch + 1):>5} {fstr}'
            print(fstr)

            if self.valid_loss['best'] is None or self.valid_loss['best'] > self.valid_loss['last']:
                self.valid_loss['best'] = self.valid_loss['last']
        elif loss_type == 'test':
            fstr = ''
            if self.seg_loss is not None:
                fstr += f'Average seg_loss: {avg_seg_loss:>.4f}\n'
            if self.class_loss is not None:
                fstr += f'Accuracy: {accuracy:>.4f}%'

        print(fstr)
        if log_output is not None:
            with open(log_output, 'a') as f:
                f.write(f'{fstr}\n')

    def _epoch_end(self):
        self.current_epoch_step = -1
        self.current_epoch += 1

        save_model = (
            self.save_model_every is None or self.save_model_every == 0
        ) or (
            self.current_epoch % self.save_model_every == 0
        )
        if save_model and self.model_output is not None:
            if self.unet is not None and not self.freeze_unet:
                model_dict = {
                    'epoch': self.current_epoch,
                    'step': self.current_step,
                    'model_state': self.unet.state_dict(),
                    'optimizer_state': self.unet_optimizer.state_dict(),
                    'train_loss': self.train_loss,
                    'valid_loss': self.valid_loss,
                }
                torch.save(model_dict, '_'.join([self.model_output, 'last_unet.pth']))

            if self.cnet is not None and not self.freeze_cnet:
                model_dict = {
                    'epoch': self.current_epoch,
                    'step': self.current_step,
                    'model_state': self.cnet.state_dict(),
                    'optimizer_state': self.cnet_optimizer.state_dict(),
                    'train_loss': self.train_loss,
                    'valid_loss': self.valid_loss,
                }
                torch.save(model_dict, '_'.join([self.model_output, 'last_cnet.pth']))

            if self.fine_tune_layers is not None:
                model_dict = {
                    'epoch': self.current_epoch,
                    'step': self.current_step,
                    'model_state': self.fine_tune_layers.state_dict(),
                    'optimizer_state': self.fine_tune_optimizer.state_dict(),
                    'train_loss': self.train_loss,
                    'valid_loss': self.valid_loss,
                }
                torch.save(model_dict, '_'.join([self.model_output, 'last_fine_tune.pth']))

        if self.switch_loss_weights_after is not None:
            if self.current_epoch == self.switch_loss_weights_after:
                self.seg_loss_kwargs['weight'] = self.loss_weights[1][0]
                self.class_loss_kwargs['weight'] = self.loss_weights[1][1]
                self.unet_change_loss_kwargs['weight'] = self.loss_weights[1][2]

    def _save_model_outputs(self, data_dict, dataset, idx, save_dir=None, make_subdir=False):
        nB, _, nT = (
            data_dict.get('Output').shape[:3] if data_dict.get('Output') is not None
            else data_dict.get('Target').shape[:3] if data_dict.get('Target') is not None
            else data_dict.get('Input').shape[:3] if data_dict.get('Input') is not None
            else data_dict.get('Posteriors').shape[:3] if data_dict.get('Posteriors') is not None
            else utils.fatal('Must provide at least one input to _save_model_outputs')
        )
        save_dir = os.path.join(
            self.output_dir if self.output_dir is not None else os.getcwd(),
            '_'.join(['model_outputs_epoch', f'{self.current_epoch}'])
            if save_dir is None else save_dir
        )

        for t in range(nT):
            outbase = f'timepoint{t}'
            if data_dict.get('Input') is not None:
                nC = data_dict['Input'].shape[1]
                for c in range(nC):
                    dataset._save_volume(
                        img=data_dict['Input'][:, c, t, ...],
                        outbase=(outbase + f'.input{c}'),
                        outdir=save_dir, idx=idx, rescale=True, make_subdir=make_subdir
                    )
            if data_dict.get('Target') is not None:
                dataset._save_volume(
                    img=data_dict['Target'][:, :, t, ...],
                    outbase=(outbase + f'.target'),
                    outdir=save_dir, idx=idx, is_labels=True, make_subdir=make_subdir
                )
            if data_dict.get('Posteriors') is not None:
                for n, val in enumerate(dataset.lut):
                    label_name = dataset.lut[val].name
                    dataset._save_volume(
                        img=data_dict['Posteriors'][:, n, t, ...],
                        outbase=(outbase + f'.post.{label_name}'),
                        outdir=save_dir, idx=idx, make_subdir=make_subdir
                    )
            if data_dict.get('Output') is not None:
                dataset._save_volume(
                    img=data_dict['Output'][:, :, t, ...],
                    outbase=(outbase + f'.prediction'),
                    outdir=save_dir, idx=idx, is_labels=True, make_subdir=make_subdir
                )

    def _save_change_map(self, img, dataset, outbase_str, idx, save_dir=None):
        save_dir = os.path.join(
            self.output_dir if self.output_dir is not None else os.getcwd(),
            'examples' if save_dir is None else save_dir
        )
        dataset._save_volume(
            img=img, outbase=f'change_map.{outbase_str}', outdir=save_dir, idx=idx, is_labels=True
        )

    def _plot_loss(self,
                   loss_type=None,
                   loss_fname=None,
                   loss_function_name=None,
                   title=None,
                   figname=None):
        """
        Output loss curve
        """

        # Determine file to load
        if loss_type is None and loss_fname is None:
            utils.fatal('Error: must input either loss_type or loss_fname to '
                        'SynthUNetSegmenter._plot_loss()')
        if loss_fname is not None and loss_type is not None:
            print('Warning: both loss_type and loss_fname provided to '
                  'SynthUNetSegmenter._plot_loss()... loss_type input will be ignored')

        loss_fname = (
            loss_fname if loss_fname is not None
            else self.train_log if 'train' in loss_type.casefold()
            else self.valid_log if 'valid' in loss_type.casefold()
            else utils.fatal('Error: input loss_type to SynthUNetSegmenter._plot_loss() must '
                             f'contain either "train" or "valid" (case ignored) if loss_fname not '
                             'provided.')
        )
        if not os.path.isfile(loss_fname):
            utils.fatal(f'Error: {loss_fname} is not a valid file')

        # Load loss output file
        df = pd.read_csv(loss_fname, delim_whitespace=True)
        epoch = df['Epoch'].values
        steps = df['Step'].values if 'Step' in df else None
        epoch = (
            (epoch - 1) + (steps - self.print_loss_every) / (self.steps_per_epoch)
            if steps is not None
            else (epoch - 1)
        )
        loss = df[[x for x in df.columns.tolist() if 'loss' in x]]
        cnvg = np.nanmean(loss.values[(int(loss.shape[0] * 0.9)):, :], axis=0)
        n_losses = loss.shape[1]

        # Plot
        fig, ax = plt.subplots()
        for n in range(n_losses):
            line = ax.plot(epoch, loss.values[:, n], label=loss.columns[n])
            asmpt = ax.plot(epoch, cnvg[n] * np.ones_like(epoch), 'k--',
                            label=(r'${{{\rightarrow\approx}}}$' + f'{cnvg[n]:>.3f}'))

        ax.set_xlim(0, self.max_n_epochs)
        ax.set_ylim(0, np.nanmax(loss.values))
        ax.tick_params(axis='both', which='major', labelsize=10)
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Loss' if loss_type is None else loss_type, fontsize=12)
        ax.set_title(title, fontsize=14)
        plt.legend(loc='upper right', fontsize=10)

        fig.savefig(
            figname if figname is not None
            else
            os.path.join(
                self.output_dir, 'loss.png' if loss_type is None else f'{loss_type}_loss.png'
            )
        )


# --------------------------------------------------------------------------------------------------

def _config_optimizer(model_params, **config):
    optimizerID = config.get('_class')

    if 'Adam' in config['_class']:
        optimizer = eval(f'torch.optim.{optimizerID}')(
            params=model_params,
            betas=tuple(config['betas']),
            lr=config['lr_start'],
            weight_decay=config['weight_decay'],
        )
    elif 'SGD' in optimizerID:
        optimizer = eval(f'torch.optim.{optimizerID}')(
            params=model_params,
            dampening=config['dampening'],
            lr=config['lr_start'],
            momentum=config['momentum'],
            weight_decay=config['weight_decay'],
        )
    else:
        raise Exception('invalid optimizer')

    return optimizer


# --------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    main(utils.parse_args())
