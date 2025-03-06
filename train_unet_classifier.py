import os, warnings, argparse, logging, random, time
from os import system
import numpy as np
import nibabel as nib
import freesurfer as fs

import torch
import torch.nn as nn
import torch.optim
from torch.utils.data import DataLoader, RandomSampler

from data_utils.dataloader_synth import call_dataset
from models import augmentations as aug

import options
import models
from models import unet4d_classifier as model
from models.segment import Segment as segment
import models.loss_functions as loss_functions



def _set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.autograd.set_detect_anomaly(True)
    os.environ["PYTHONHASHSEED"] = str(seed)



def _setup_log(filename, string):
    f = open(filename, 'w')
    f.write(string + '\n')
    f.close()


def _write_data_ids_log(filename, dataset):
    id_list = set([dataset.input_files[i][0].split('/')[-1].split('.')[0] \
                   for i in range(len(dataset.input_files))])
    f = open(filename, 'w')
    for subject_id in sorted(id_list): f.write(subject_id + '\n')
    f.close()
    


def _config_optimizer(config, network_params):
    optimizerID = config.optimizer

    if optimizerID=="Adam" or optimizerID=="AdamW":
        betas = (config.beta1, config.beta2)
        optimizer = torch.optim.__dict__[optimizerID](params=network_params,
                                                      betas=(config.beta1, config.beta2),
                                                      lr=config.lr_start,
                                                      weight_decay=config.weight_decay
        )
    elif optimizerID=="SGD":
        optimizer = torch.optim.__dict__[optimizerID](params=network_params,
                                                      lr=config.lr_start,
                                                      momentum=config.momentum,
                                                      weight_decay=config.weight_decay,
                                                      dampening=config.dampening
        )
    else:
        raise Exception('invalid optimizer')

    return optimizer



def _config_lr_scheduler(config, optimizer):
    schedulerID = config.lr_scheduler
    
    if schedulerID=="ConstantLR":
        scheduler = torch.optim.lr_scheduler.__dict__[schedulerID](optimizer,
                                                                   factor=config.lr_param,
                                                                   total_iters=config.max_n_epochs,
        )
    elif schedulerID=="StepLR":
        scheduler = torch.optim.lr_scheduler.__dict__[schedulerID](optimizer,
                                                                   step_size=config.max_n_epochs,
        )
    elif schedulerID=="PolynomialLR":
        scheduler = torch.optim.lr_scheduler.__dict__[schedulerID](optimizer,
                                                                   total_iters=config.max_n_epochs,
                                                                   power=config.lr_param
        )
    elif schedulerID=="ExponentialLR":
        scheduler = torch.optim.lr_scheduler.__dict__[schedulerID](optimizer,
                                                                   gamma=config.lr_param
        )
    else:
        raise Exception("invalid lr_scheduler")

    return scheduler



def _config():
    parser = argparse.ArgumentParser()
    parser = options.add_argparse(parser)
    pargs = parser.parse_args()

    seed = pargs.seed
    _set_seed(seed)

    return pargs



def _dataloader_random_sample(pargs, dataset):
    batch_size = pargs.batch_size
    n_workers = pargs.n_workers
    n_samples = pargs.steps_per_epoch

    sampler = RandomSampler(dataset, replacement=True, num_samples=n_samples)
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=batch_size, num_workers=n_workers, pin_memory=True)
    return dataloader

    

def main(pargs):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.set_float32_matmul_precision('medium')

    ### Data set-up ###
    input_data_files = pargs.input_data_files
    data_inds_config = pargs.data_inds_config
    n_data_splits = pargs.n_data_splits

    aug_config = pargs.aug_config
    crop_patch_size = pargs.crop_patch_size
    n_samples = pargs.n_dataset_samples
    apply_robust_normalization = pargs.apply_robust_normalization

    datasets = call_dataset(input_data_files=input_data_files,
                            aug_config=aug_config,
                            n_data_splits=n_data_splits,
                            n_subjects=n_samples,
                            crop_patch_size=crop_patch_size,
                            apply_robust_normalization=apply_robust_normalization,
                            device=device
    )
    train_data = datasets[0]
    test_data = datasets[-1] if n_data_splits > 1 else None
    valid_data = datasets[-2] if n_data_splits > 2 else None

    batch_size = pargs.batch_size
    n_workers = pargs.n_workers
    train_loader = _dataloader_random_sample(pargs, train_data)
    test_loader = None if test_data is None else \
        DataLoader(test_data, batch_size=batch_size, num_workers=n_workers, shuffle=True)
    valid_loader = None if valid_data is None else \
        DataLoader(valid_data, batch_size=batch_size, num_workers=n_workers, shuffle=True)
    
    
    ### Model set-up ###
    activation_function = pargs.activation_function
    conv_window_size = pargs.conv_window_size
    pool_window_size = pargs.pool_window_size
    use_residuals = bool(pargs.use_residuals)

    network = model.__dict__[pargs.network](in_channels=train_data.__n_input__(),
                                            out_channels_unet=train_data.__n_class__(),
                                            out_channels_cnet=train_data.n_synth_classes,
    ).to(device)
    loss_fns = [loss_functions.__dict__[loss_fn] for loss_fn in pargs.loss_fns]
    optimizers = [_config_optimizer(pargs, network.parameters()) for i in range(len(loss_fns))]
    lr_schedulers = [_config_lr_scheduler(pargs, optim) for optim in optimizers]

    
    ### Load pre-trained model parameters ###
    train_checkpoint_path = pargs.model_state_path
    if train_checkpoint_path is not None and os.path.isfile(train_checkpoint_path):
        train_checkpoint = torch.load(train_checkpoint_path)
        network.load_state_dict(train_checkpoint["model_state"])
        optimizers[0].load_state_dict(train_checkpoint["optimizer_state"])
    else:
        train_checkpoint = None
        

    ### Parse everything into trainer ###
    output_folder = pargs.output_dir
    start_epoch = 0 if train_checkpoint is None else train_checkpoint["epoch"]
    steps_per_epoch = pargs.steps_per_epoch
    max_n_epochs = pargs.max_n_epochs
    start_aug_on = pargs.start_aug_on if pargs.start_aug_on is not None else max_n_epochs
    switch_loss_on = pargs.switch_loss_on
    
    trainer = segment(model=network,
                      optimizer=optimizers[0],
                      scheduler=lr_schedulers[0],
                      loss_functions=loss_fns,
                      start_epoch=start_epoch,
                      max_n_epochs=max_n_epochs,
                      start_full_aug_on=0,
                      steps_per_epoch=steps_per_epoch,
                      switch_loss_on=switch_loss_on,
                      output_folder=output_folder,
                      device=device,
    )

    
    ### Logging set-up ###
    if output_folder is not None and train_checkpoint_path is None:
        if not os.path.exists(output_folder):  os.mkdir(output_folder)

        _write_data_ids_log(os.path.join(output_folder, "train_data_ids_log.txt"), train_data)
        _setup_log(os.path.join(output_folder, "train_log.txt"), "Epoch TrainLoss")

        if valid_data is not None:
            _write_data_ids_log(os.path.join(output_folder, "valid_data_ids_log.txt"), valid_data)
            _setup_log(os.path.join(output_folder, "valid_log.txt"), "Epoch ValidLoss")

        if test_data is not None:
            _write_data_ids_log(os.path.join(output_folder, "test_data_ids_log.txt"), test_data)
            _setup_log(os.path.join(output_folder, "test_log.txt"), "SubjectId TestLoss")
            
        param_output_file=os.path.join(output_folder, "training_params.txt")
        f = open(param_output_file, 'w')
        if aug_config is not None:
            f.write('--------------------------------------\n')
            f.write('Full data augmentation set:\n')        
            f.write('--------------------------------------\n')
            
            f.write(f'Augmentations:\n')
            for T in train_data.full_augmentations.transforms:
                f.write(f'{T.__class__.__name__}:')
                for k in T.__dict__:
                    if k == '_get_bounds': f.write(f'{k}={T.__dict__[k].__name__}\n')
                    elif k != 'X': f.write(f'{k}={T.__dict__[k]}\n')
                f.write('\n')
            f.write('\n')
            
        f.write('--------------------------------------\n')
        f.write('Training parameters:\n')
        f.write('--------------------------------------\n')
        for k in pargs.__dict__:
            f.write(f'{k}={pargs.__dict__[k]}\n')
        f.write('\n')        
        f.close()
        
        with open(param_output_file, 'r') as f:
            print(f.read())
        f.close()
            

    ### Run ###
    print('--------------------------------------')
    if valid_data is not None and test_data is not None:
        print('Train: %d | Valid: %d | Tests: %d' % (len(train_data), len(valid_data), len(test_data)))
    elif test_data is not None and valid_data is None:
         print('Train: %d | Tests: %d' % (len(train_data), len(test_data)))
    else:
        print('Train: %d' % (len(train_data)))
    print('--------------------------------------')

    for epoch in range(start_epoch, max_n_epochs):
        trainer.train(loader=train_loader)
        if valid_loader is not None:
            trainer.validate(loader=valid_loader, save_output=False)
        trainer._epoch_end()
        

    save_dir = '_'.join(['model_outputs__epoch', str(max_n_epochs)])
    if test_loader is not None:
        trainer.test(loader=test_loader, save_dir=save_dir, save_output=False,
                     write_inputs=False, write_targets=False)
    

##########################
if __name__ == "__main__":
    main(_config())
