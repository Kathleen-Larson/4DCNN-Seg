import os
import sys
import argparse
import warnings
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint as pl_ModelCheckpoint
from pytorch_lightning import loggers as pl_loggers

from torchvision import transforms
from torchsummary import summary
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader

import models
import models.losses
import models.optimizers
from models.segment import Segment as segment
from models.unet4d import UNet3D_long, UNet2D_long
from models.progress import ProgressBar as ProgressBar

from datasets.synth import synth_3d, synth_2d



### Torch set-up ###
pl.seed_everything(0, workers=True) #####
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True
torch.autograd.set_detect_anomaly(True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_float32_matmul_precision('medium')

warnings.filterwarnings('ignore',
                        "Your \\`val_dataloader\\`\\'s sampler has shuffling enabled, it is strongly recommended that you turn shuffling off for val/test/predict dataloaders.")


# Data set up
data_config = 'data_config_2d.csv'
aug_config = 'datasets/augmentation_parameters.txt'
train_data, valid_data, test_data  = synth_2d(data_config=data_config, augmentation_config=aug_config)

train_loader = DataLoader(train_data, batch_size=1, shuffle=True, num_workers=8, pin_memory=True)
valid_loader = DataLoader(valid_data, batch_size=1, shuffle=True, num_workers=8, pin_memory=True)
test_loader = DataLoader(test_data, batch_size=1, shuffle=False, num_workers=8, pin_memory=True)


### Model set-up ###
lr_start = 0.0001
lr_param = 0.1
decay = 0.0000


# Network set up
model = UNet2D_long(in_channels=1, out_channels=train_data.__n_class__()).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr_start, weight_decay=decay)
loss_fn = nn.BCEWithLogitsLoss()
metrics=[models.losses.MeanDice()]


output_folder = '2d_adni_results'
if not os.path.exists(output_folder):  os.mkdir(output_folder)

trainee = segment(model=model, optimizer=optimizer, loss=loss_fn, \
                  train_data=train_data, valid_data=valid_data, test_data=test_data, output_folder=output_folder,
                  seed=0, lr_start=lr_start, lr_param=lr_param,
                  train_metrics=metrics, valid_metrics=metrics, test_metrics=metrics,
                  save_train_output_every=1, save_valid_output_every=0, schedule='poly',
)



### Run ? ###
print("Train: %d | Valid: %d | Tests: %d" % \
      (len(train_loader.dataset), len(valid_loader.dataset), len(test_loader.dataset)))

callbacks = [pl_ModelCheckpoint(monitor='val_metric0', mode='max'),
             ProgressBar(refresh_rate=1)]
#logger = pl_loggers.TensorBoardLogger('logs/', name=output_base, default_hp_metric=False)

trainer = pl.Trainer(accelerator='gpu', devices=1, max_epochs=1,
                     gradient_clip_val=0.5, gradient_clip_algorithm='value', precision=16)

trainer.fit(trainee, train_loader, valid_loader)
trainer.validate(trainee, valid_loader, verbose=False)
trainer.test(trainee, test_loader, verbose=False)
