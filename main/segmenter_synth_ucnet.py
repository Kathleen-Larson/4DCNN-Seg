import os
import time
import random
import pathlib as Path
import numpy as np
import surfa as sf

import torch
import torch.optim
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast

import loss_functions
import utils

from synth_dataset import SynthDataset as SD

class SynthUCNetSegmenter:
    def __init__(self,
                 loss_funcs,
                 model,
                 optimizer,
                 synth,
                 device:str=None,
                 infer_only:bool=False,
                 max_n_epochs:int=None,
                 max_n_steps:int=None,
                 model_state_path:str=None,
                 n_train_samples:int=None,
                 output_dir:str=None,
                 print_loss_every:int=None,
                 resume:bool=False,
                 save_model_every:int=None,
                 save_outputs_every:int=None,
                 steps_per_epoch:int=None,
                 switch_loss_on:int=None,
                 T:int=2,
                 **kwargs
    ):
        # Parse config args
        self.checkpoint_path = model_state_path
        self.device = 'cpu' if device is None else device
        self.infer_only = infer_only
        self.max_n_epochs = max_n_epochs
        self.max_n_steps = max_n_steps
        self.output_dir = output_dir
        self.save_outputs_every = save_outputs_every
        self.steps_per_epoch = steps_per_epoch
        self.T = 2
        
        # Configure training specific args
        if not self.infer_only:
            # Number of steps per epoch
            if self.steps_per_epoch is None and n_train_samples is None:
                utils.fatal(
                    'Error initializing PGlandsSegmenter: must specify either '
                    'steps_per_epoch and/or n_train_samples.'
                )
            if self.steps_per_epoch is None:
                self.steps_per_epoch = n_train_samples
            elif (n_train_samples is not None and
                  self.steps_per_epoch != n_train_samples):
                print(
                    f'Mismatch between steps_per_epoch={self.steps_per_epoch} '
                    f'n_train_samples={n_train_samples}, using '
                    f'n_train_samples'
                )
                self.steps_per_epoch = n_train_samples

            # Maximum number of steps
            if self.max_n_epochs is None and self.max_n_steps is None:
                utils.fatal(
                    'Error initializing PGlandsSegmenter: must specify either '
                    'max_n_steps or max_n_epochs'
                )
                
            if self.max_n_epochs is None:
                self.max_n_epochs = self.max_n_steps // self.steps_per_epoch
            elif self.max_n_steps is None:
                self.max_n_steps = self.max_n_epochs * self.steps_per_epoch
                
            if self.max_n_epochs * self.steps_per_epoch < self.max_n_steps:
                print(
                    f'Warning: max_n_steps set to {self.max_n_steps}, but '
                    f'training will exit after max_n_epochs={max_n_epochs}, '
                    f'({self.max_n_epochs * self.steps_per_epoch} steps).'
                )
            elif self.max_n_epochs * self.steps_per_epoch > self.max_n_steps:
                print(
                    f'Warning: max_n_epochs set to {self.max_n_epochs}, but '
                    f'training will exit after max_n_steps={max_n_steps} '
                    f'({self.max_n_steps // self.steps_per_epoch} epochs).'
                )
                
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
            
        # Initialize synth models
        self.synth = synth
        self.synth_classes_all = [cls for cls in self.synth]
        
        # Initialize training
        self.model = model
        self.optimizer = optimizer

        if self.checkpoint_path is not None:
            checkpoint = torch.load(self.checkpoint_path)
            self.model.load_state_dict(checkpoint['model_state'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state'])

            self.current_epoch = checkpoint['epoch']
            self.current_step = checkpoint['step']
            self.train_loss = checkpoint['train_loss']
            self.valid_loss = checkpoint['valid_loss']
        else:
            self.current_epoch = 0
            self.current_step = 0
            self.train_loss = {'last': None, 'best': None}
            self.valid_loss = {'last': None, 'best': None}
            
        self.current_epoch_step = 0

        # Loss functions
        self.loss_logits = {
            'func': eval(loss_funcs['logits']['func']),
            'weight': loss_funcs['logits']['weight']
        }
        self.loss_change = {
            'func': eval(loss_funcs['change']['func']),
            'weight': loss_funcs['change']['weight']
        }
        self.loss_class = {
            'func': eval(loss_funcs['class']['func']),
            'weight': loss_funcs['class']['weight']
        }
            
        # Set up output logs
        if self.output_dir is not None:
            if not os.path.isdir(self.output_dir):
                os.makedirs(self.output_dir)
                
            self.model_output = os.path.join(self.output_dir, 'model')
            self.test_log = os.path.join(self.output_dir, 'test_log.txt')
            utils.init_text_file(
                self.test_log, 'FileId LogitsLoss ChangeLoss ClassLoss'
            )
            
            if not infer_only:
                self.train_log = os.path.join(self.output_dir, 'train_log.txt')
                self.valid_log = os.path.join(self.output_dir, 'valid_log.txt')
                utils.init_text_file(
                    self.train_log,
                    'Epoch Step LogitsLoss ChangeLoss ClassLoss TotalLoss'
                )
                utils.init_text_file(
                    self.valid_log,
                    'Epoch LogitsLoss ChangeLoss ClassLoss TotalLoss'
                )
            else:
                self.train_log = None
                self.valid_log = None
        else:
            self.model_output = None
            self.test_log = None
            self.train_log = None
            self.valid_log = None

            
    #--------------------------------------------------------------------------
    
    def _train(self, loader):
        """
        Training loop
        """
        N = self.steps_per_epoch

        loss_logits_avg = 0.
        loss_change_avg = 0.
        loss_class_avg = 0.
        loss_avg = 0.

        self.model.train()
        self.model.zero_grad()

        t = time.time()

        for vol, idx in loader:
            # Synthesize images from input volume
            synth_class, synth_model = random.choice(list(self.synth.items()))
            X, y = loader.dataset.augmentations(
                synth_model(vol.to(self.device))
            )
            
            # Run model
            logits, pclass = self.model(X)

            # Compute losses
            loss_logits = self.loss_logits['func'](
                logits, y, weight=self.loss_logits['weight'],
                compute_softmax=True
            )
            loss_change = self.loss_change['func'](
                logits, y, weight=self.loss_change['weight'],
                compute_softmax=True
            )
            loss_class = self.loss_class['func'](
                pclass, synth_class, self.synth_classes_all,
                weight=self.loss_class['weight']
            )
            loss = loss_logits + loss_change + loss_class
            
            loss_logits_avg += loss_logits.item() / N
            loss_change_avg += loss_change.item() / N
            loss_class_avg += loss_class.item() / N
            loss_avg += loss.item() / N

            # Update end of step
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            self.current_step += 1
            self.current_epoch_step += 1

            if self.current_epoch_step % (self.print_loss_every) == 0:
                with open(self.train_log, 'a') as f:
                    f.write(f'{self.current_epoch:>5}')
                    f.write(f'{self.current_epoch_step:>5}')
                    f.write(f'{loss_logits.item():>11.4f}')
                    f.write(f'{loss_change.item():>11.4f}')
                    f.write(f'{loss_class.item():>10.4f}')
                    f.write(f'{loss.item():>10.4f}')
                    f.write(f'\n')
                    
                print(f'Epoch={self.current_epoch},',
                      f'step={self.current_epoch_step}:',
                      f'loss_logits={loss_logits.item():>.4f}',
                      f'loss_change={loss_change.item():>.4f}',
                      f'loss_class={loss_class.item():>.4f}',
                      f'loss={loss.item():>4f}')
                
        # Update end of epoch train loss
        self.train_loss['last'] = loss_avg
        if (self.train_loss['best'] is None
            or self.train_loss['best'] > self.train_loss['last']):
            self.train_loss['best'] = self.train_loss['last']

              
                
    def _predict(self, loader,
                 loss_type:str='test',
                 save_dir:str=None,
                 save_outputs:bool=False,
                 write_inputs:bool=False,
                 write_targets:bool=False,
                 write_posteriors:bool=False,
    ):
        """
        Inference (validation/testing) loop
        """
        N = len(loader.dataset)

        loss_logits_avg = 0.
        loss_change_avg = 0.
        loss_class_avg = 0.
        loss_avg = 0.

        if loss_type not in ['valid', 'test']:
            utils.fatal(
                f'Error in segmenter._predict: {loss_type} not a valid input '
                'for loss type!'
            )
        
        self.model.eval()
        self.model.zero_grad()
        
        for vol, idx in loader:
            # Synthesize images from input volume
            synth_class, synth_model = random.choice(list(self.synth.items()))
            X, y = loader.dataset.augmentations(
                synth_model(vol.to(self.device))
            )

            # Run model
            with torch.no_grad():
                logits, pclass = self.model(X)

            # Compute losses
            loss_logits = self.loss_logits['func'](
                logits, y, weight=self.loss_logits['weight'],
                compute_softmax=True
            )
            loss_change = self.loss_change['func'](
                logits, y, weight=self.loss_change['weight'],
                compute_softmax=True
            )
            loss_class = self.loss_class['func'](
                pclass, synth_class, self.synth_classes_all,
                weight=self.loss_class['weight']
            )
            loss = loss_logits + loss_change + loss_class

            loss_logits_avg += loss_logits.item() / N
            loss_change_avg += loss_change.item() / N
            loss_class_avg += loss_class.item() / N
            loss_avg += loss.item() / N
            
            if loss_type == 'test':
                with open(self.test_log, 'a') as f:
                    f.write(f'{loader.dataset.outbases[idx]}')
                    f.write(f'{loss_logits.item():>11.4f}')
                    f.write(f'{loss_change.item():>11.4f}')
                    f.write(f'{loss_class.item():>11.4f}')
                    f.write(f'{loss.item():>11.4f}')
                    
            # Write outputs?
            if save_outputs:
                posts = F.softmax(logits, dim=1)
                seg = torch.argmax(posts, dim=1, keepdims=True)
                out_dict = {'Output': seg}

                if write_inputs:
                    out_dict['Input'] = X
                if write_targets:
                    out_dict['Target'] = torch.argmax(y, dim=1, keepdims=True)
                if write_posteriors:
                    out_dict['Posteriors'] = posts

                self._save_model_outputs(
                    data_dict=out_dict, dataset=loader.dataset,
                    idx=idx, save_dir=save_dir
                )

        # Store loss at end of loop
        if loss_type == 'test':
            with open(self.test_log, 'a') as f:
                f.write(f'Average {loss_avg:>.4f}\n')
        else:
            self.valid_loss['last'] = loss_avg
            if (self.valid_loss['best'] is None
                or self.valid_loss['best'] > self.valid_loss['last']):
                self.valid_loss['best'] = self.valid_loss['last']
                
            if self.valid_log is not None:
                with open(self.valid_log, 'a') as f:
                    f.write(f'{self.current_epoch:>5}')
                    f.write(f'{loss_logits_avg:>11.4f}')
                    f.write(f'{loss_change_avg:>11.4f}')
                    f.write(f'{loss_class_avg:>10.4f}')
                    f.write(f'{loss_avg:>10.4f}')
                    f.write(f'\n')

            
        
                
    #--------------------------------------------------------------------------

    def _epoch_end(self):
        """
        Runs at the end of each training epoch (after validation steps)
        """
        self.current_epoch_step = 0
        self.current_epoch += 1

        # Save model
        if self.model_output is not None and not (
                self.save_model_every is None or self.save_model_every == 0
        ) and (self.current_epoch + 1) % self.save_model_every == 0:
            torch.save(
                {'epoch': self.current_epoch,
                 'step': self.current_step,
                 'model_state': self.model.state_dict(),
                 'optimizer_state': self.optimizer.state_dict()},
                '_'.join([self.model_output, 'last.pth'])
            )
            

    def _save_model_outputs(self, data_dict:dict, dataset, idx:int,
                            save_dir:str=None
    ):
        nB, _, nT = data_dict['Output'].shape[:3]
        save_dir = os.path.join(
            self.output_dir,
            '_'.join(['model_outputs_epoch', f'{self.current_epoch}'])
            if save_dir is None else save_dir
        )

        for t in range(nT):
            outbase = f'timepoint{t}'

            if 'Input' in data_dict:
                nC = data_dict['Input'].shape[1]
                for c in range(nC):
                    dataset._save_volume(
                        img=data_dict['Input'][:, c, t, ...],
                        outbase=(outbase + f'.input{c}'),
                        outdir=save_dir, idx=idx, rescale=True
                    )
            if 'Target' in data_dict:
                dataset._save_volume(
                    img=data_dict['Target'][:, :, t, ...],
                    outbase=(outbase + f'.target'),
                    outdir=save_dir, idx=idx, is_labels=True
                )

            if 'Posteriors' in data_dict:
                for n, val in enumerate(dataset.lut):
                    label_name = dataset.lut[val].name
                    dataset._save_volume(
                        img=data_dict['Posteriors'][:, n, t, ...],
                        outbase=(outbase + f'.post.{label_name}'),
                        outdir=save_dir, idx=idx
                    )
                    
            if 'Output' in data_dict:
                dataset._save_volume(
                    img=data_dict['Output'][:, :, t, ...],
                    outbase=(outbase + f'.prediction'),
                    outdir=save_dir, idx=idx, is_labels=True
            )

            

def write_volume(x, pathbase:str, lut=None, is_logits:bool=False,
                 is_labels:bool=False, is_onehot:bool=False
):
    is_onehot = True if is_logits else is_onehot
    is_labels = True if is_onehot else is_labels
    
    x = (torch.argmax(F.softmax(x, dim=1), dim=1) if is_logits else x
    ).cpu().numpy().astype(np.int32 if is_labels else np.float32)
    
    for t in range(x.shape[2]):
        img = sf.Volume(x[:, :, t, ...])
        img.labels = lut if lut is not None and is_labels else None
        img.save(f'{pathbase}.{t}.mgz')



#------------------------------------------------------------------------------



