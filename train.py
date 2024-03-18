import os, warnings, argparse, logging, random
import numpy as np
import nibabel as nib
import freesurfer as fs
import torch
import torch.nn as nn
import torch.optim
from torch.utils.data import DataLoader

from data_utils import synth
from data_utils import transforms as t

import options
import models
from models import unet4d, metrics
from models.segment import Segment as segment
import models.loss_functions as loss_fns



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



def _setup_log(file, str):
    f = open(file, 'w')
    f.write(str + '\n')
    f.close()




def _config_optimizer(config, network_params):
    optimizerID = config.optimizer
    
    if optimizerID=="Adam" or optimizerID=="AdamW":
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
                                                                   step_size=5,
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


def main(pargs):
    ### Torch set-up ###
    torch.cuda.empty_cache() # not sure if this line is necessary
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.set_float32_matmul_precision('medium')

    ### Data set-up ###
    aug_config = pargs.aug_config
    batch_size = pargs.batch_size
    data_config = pargs.data_config
    call_dataset = synth.__dict__[pargs.data_loader] # calls synth3d or synth2d
    n_samples = pargs.n_samples if pargs.n_samples != 0 else None
    n_workers = pargs.n_workers

    train_data, valid_data, test_data = call_dataset(data_config=data_config,
                                                     aug_config=aug_config,
                                                     #subsample=2,
                                                     n_samples=n_samples
    )
    train_loader = DataLoader(train_data,
                              batch_size=batch_size,
                              shuffle=False,
                              num_workers=n_workers,
                              pin_memory=True
    )

    valid_loader = DataLoader(valid_data,
                              batch_size=batch_size,
                              shuffle=False,
                              num_workers=n_workers,
                              pin_memory=True
    )
    test_loader = DataLoader(test_data,
                             batch_size=batch_size,
                             shuffle=False,
                             num_workers=n_workers,
                             pin_memory=True
    )
    

    ### Model set-up ###
    network = models.unet4d.__dict__[pargs.network](in_channels=train_data.__n_input__(),
                                                    out_channels=train_data.__n_class__(),
    ).to(device)
    loss_function =  loss_fns.__dict__[pargs.loss_fn]

    load_model_state = pargs.load_model_state
    output_folder = pargs.output_dir
    if load_model_state is not None and output_folder is not None:
        checkpoint = torch.load(load_model_state)
        network.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint['optimizer_state'])
        current_epoch = checkpoint['epoch'] + 1
    else:
        checkpoint = None
        optimizer = _config_optimizer(pargs, network.parameters())

    lr_scheduler = _config_lr_scheduler(pargs, optimizer) # this should also be saved w/in chckpnt
    

    ## Metrics set-up ###
    metrics_train = [models.metrics.__dict__[pargs.metrics_train[i]] \
                     for i in range(len(pargs.metrics_train))]
    metrics_test = [models.metrics.__dict__[pargs.metrics_test[i]] \
                    for i in range(len(pargs.metrics_test))]
    metrics_valid = [models.metrics.__dict__[pargs.metrics_valid[i]] \
                     for i in range(len(pargs.metrics_valid))]


    ### Logging set-up ###
    start_epoch = 0 if load_model_state is None else current_epoch
    max_n_epochs = pargs.max_n_epochs
    if output_folder is not None:
        if not os.path.exists(output_folder):  os.mkdir(output_folder)
        if checkpoint is None:
            _setup_log(os.path.join(output_folder, "training_log.txt"), "Epoch Loss " + \
                       " ".join(pargs.metrics_train[i] for i in range(len(pargs.metrics_train))))
            _setup_log(os.path.join(output_folder, "validation_log.txt"), "Epoch Loss " + \
                       " ".join(pargs.metrics_valid[i] for i in range(len(pargs.metrics_valid))))
            _setup_log(os.path.join(output_folder, "testing_log.txt"), "ImgID Loss " + \
                       " ".join(pargs.metrics_test[i] for i in range(len(pargs.metrics_test))))

            param_output_file=os.path.join(output_folder, "training_params.txt")
            f = open(param_output_file, 'w')
            
            f.write('--------------------------------------\n')
            f.write('Full data augmentation set:\n')
            f.write('--------------------------------------\n')
            for t in train_data.full_augmentation.transforms:
                f.write(f'{t.__class__.__name__}:\n')
                if t is not None:
                    for k in t.__dict__:
                        if k != 'X':  f.write(f'    {k}={t.__dict__[k]}\n')
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


    ### Parse everything into trainer ###
    trainer = segment(model=network,
                      optimizer=optimizer,
                      scheduler=lr_scheduler,
                      loss_function=loss_function,
                      max_n_epochs=max_n_epochs,
                      start_full_aug_on=None,
                      output_folder=output_folder,
                      device=device,
    )
    

    ### Run ###
    print('--------------------------------------')
    print("Train: %d | Valid: %d | Tests: %d" % \
          (len(train_loader.dataset), len(valid_loader.dataset), len(test_loader.dataset)))
    print('--------------------------------------')
    
    save_output_every=100
    for epoch in range(start_epoch, max_n_epochs):
        trainer.train(loader=train_loader, save_output=True if (epoch+1) % save_output_every == 0 else False)
        trainer.validate(loader=valid_loader, save_output=False)
        trainer._epoch_end()
    trainer.test(loader=test_loader, metrics_list=metrics_test, save_output=True)



##########################
if __name__ == "__main__":
    main(_config())
