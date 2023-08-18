import os
import sys
import torch
import torch.nn as nn
import argparse
import warnings

import models
import models.losses
import models.optimizers

import options
import datasets
import logging

from torch.utils.data import DataLoader
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import loggers as pl_loggers



if __name__ == "__main__":

    # Argument parsing
    parser = argparse.ArgumentParser()
    parser = Trainer.add_argparse_args(parser)
    parser = options.set_argparse_defs(parser)
    parser = options.add_argparse_args(parser)

    args = parser.parse_args()
    args.default_root_dir = os.path.join('./checkpoints/', args.remarks) #change this to your taste

    warnings.filterwarnings('ignore',\
                            "The \\`srun\\` command is available on your system but is not used.")
    warnings.filterwarnings('ignore',\
                            "torch.meshgrid: in an upcoming release, it will be required to pass the indexing argument")
    warnings.filterwarnings('ignore',\
                            "Detected call of \\`lr_scheduler.step\\(\\)\\` before \\`optimizer.step\\(\\)\\`")
    warnings.filterwarnings('ignore',\
                            "Checkpoint directory .* exists and is not empty")

    # Torch set-up
    seed_everything(args.seed, workers=True)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    torch.autograd.set_detect_anomaly(True)

    # Data loaders
    train_data, valid_data, tests_data = datasets.__dict__[args.dataset]\
        (seed=args.seed, fraction=args.fraction, augment=args.augment) #, positional=args.positional)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=8, pin_memory=True)
    valid_loader = DataLoader(valid_data, batch_size=args.valid_batch_size, shuffle=False, num_workers=8, pin_memory=True)
    tests_loader = DataLoader(tests_data, batch_size=args.valid_batch_size, shuffle=False, num_workers=8, pin_memory=True)


    # Model set-up
    loss = models.losses.__dict__[args.loss]
    network = models.__dict__[args.network]\
        (train_data.__numinput__(), train_data.__numclass__(), pretrained=args.pretrained, drop=args.drop_rate)
    optim = models.optimizers.__dict__[args.optim]\
        (network.parameters(), lr=args.lr_start, momentum=args.momentum, weight_decay=args.decay, nesterov=args.nesterov)

    # Metrics for optimizing
    train_metrics = [models.losses.__dict__[args.train_metrics[i]]() for i in range(len(args.train_metrics))]
    valid_metrics = [models.losses.__dict__[args.valid_metrics[i]]() for i in range(len(args.valid_metrics))]
    tests_metrics = [models.losses.__dict__[args.tests_metrics[i]]() for i in range(len(args.tests_metrics))]

    
    # This is how we visualize and store progress and stuff
    callbacks = [ModelCheckpoint(monitor=args.monitor, mode=args.monitor_mode,\
                    dirpath=args.default_root_dir, filename='best', save_last=True), models.ProgressBar(refresh_rate=5)]
    logger = pl_loggers.TensorBoardLogger('logs/', name=args.remarks, default_hp_metric=False, version='all')
    loader = models.__dict__[args.trainee].load_from_checkpoint if args.load != '' else models.__dict__[args.trainee]
    checkpt = os.path.join(args.default_root_dir, args.load) if args.load != '' else args.load

    
    # 
    trainee = loader(checkpoint_path=checkpt, model=network, optimizer=optim, loss=loss\
                    train_data=train_data, valid_data=valid_data, tests_data=tests_data,\
                    train_metrics=train_metrics, valid_metrics=valid_metrics, tests_metrics=tests_metrics,\
                    schedule=args.schedule, monitor=args.monitor, strict=False)
    trainer = Trainer.from_argparse_args(args, callbacks=callbacks, logger=logger,\
                    gradient_clip_val=0.5, gradient_clip_algorithm='value', precision=16)

    print("Train: %d | Valid: %d | Tests: %d" % \
          (len(train_loader.dataset), len(valid_loader.dataset), len(tests_loader.dataset)), file=sys.stderr)
    if args.train:
        trainer.fit(trainee, train_loader, valid_loader)
    if args.validate:
        trainer.validate(trainee, val_dataloaders=valid_loader, verbose=False)
    if args.test:
        trainer.test(trainee, test_dataloaders=tests_loader, verbose=False)
