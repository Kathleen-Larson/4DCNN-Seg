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



###################################################################################################

class Segment:
    def __init__(self,
                 model,
                 optimizer,
                 scheduler,
                 loss_functions,
                 steps_per_epoch,
                 switch_loss_on,
                 max_n_epochs,
                 output_folder,
                 device,
                 start_full_aug_on:int=0,
                 start_epoch:int=0,
    ):
        super().__init__()
        self.device = device
        self.model = model                

        self.T = 2 # temporal dimension index
        self.loss_functions = loss_functions
        self.switch_loss_on = 0 if not isinstance(switch_loss_on, int) \
            else switch_loss_on
        self.loss_fn = self.loss_functions[0] if start_epoch < self.switch_loss_on \
            or len(self.loss_functions) == 1 else self.loss_functions[1]
        self.optimizer = optimizer
        self.scheduler = scheduler

        self.valid_loss = [None, None]

        self.max_n_epochs = max_n_epochs
        self.steps_per_epoch = steps_per_epoch
        self.start_full_aug_on = start_full_aug_on if start_full_aug_on is not None \
            else self.max_n_epochs

        self.current_epoch = start_epoch
        self.current_step = 0
        self.current_train_loss_avg = 0

        self.output_folder = output_folder
        if output_folder is not None:
            self.train_output = os.path.join(self.output_folder, "train_log.txt")
            self.valid_output = os.path.join(self.output_folder, "valid_log.txt")
            self.test_output = os.path.join(self.output_folder, "test_log.txt")
            self.model_output = os.path.join(self.output_folder, "model")
        else:
            self.train_output = None
            self.valid_output = None
            self.test_output = None
            self.model_output = None

        self.epoch_st = None
        self.epoch_et = 0
        self.train_step_time = 0


    def _output_vol(self, x, loader, path:str, convert_labels:bool=False,
                    is_labels:bool=False, is_onehot:bool=False
    ):
        x = torch.argmax(x, dim=1).numpy() if is_onehot and is_labels else x.numpy()        
        while x.shape[0] == 1:  x = x.squeeze()

        if is_labels and convert_labels:
            x = x.astype(np.int32)
            x_copy = x.copy()
            for i in range(len(loader.dataset.labels_seg)):
                x_copy[x==i] = loader.dataset.labels_seg[i]
            x = x_copy.copy()

        if len(x.shape) > 3:
            n_timepoints = x.shape[0]
            for t in range(n_timepoints):
                vol = sf.Volume(x[t,...])
                if is_labels: vol.labels = loader.dataset.lut                
                outpath = '.'.join(['_'.join([path.split('.')[0], str(t)]),
                                 path.split('.')[1]])
                print('saving as', outpath)
                vol.save(outpath)
        else:
            vol = sf.Volume(x)
            if is_labels: vol.labels = loader.dataset.lut
            print('saving as', path)
            vol.save(path)
        
            
            
    ### Training loop
    def train(self, loader, save_output:bool=False):
        N = len(loader.dataset)
        nT = loader.dataset.n_timepoint
        nD = loader.dataset.X
        
        loss_avg = 0.0

        self.model.train()
        self.model.zero_grad()
        
        if self.current_epoch < self.start_full_aug_on:
            augmentations = loader.dataset.base_augmentations
        else:
            augmentations = loader.dataset.full_augmentations

        for vol, idx in loader:
            # Synthesize images from input volume
            synth_class, synth_model = \
                random.choice(list(loader.dataset.synth_models_dict.items()))
            
            X, y = synth_model(vol.to(self.device))
            """
            self._output_vol(X.cpu(), loader, 'data/test/X.mgz')
            self._output_vol(y.cpu(), loader, 'data/test/y.mgz', is_labels=True)
            """
            X, y = augmentations([X, y])
            """
            self._output_vol(X.cpu(), loader, 'data/test/X_aug.mgz')
            self._output_vol(y.cpu(), loader, 'data/test/y_aug.mgz', is_labels=True,
                             is_onehot=True, convert_labels=True)
            """
            
            # Run model
            logits = self.model(X)
            loss = self.loss_fn(logits, y)
            loss_avg += loss.item()
            loss.backward()

            self.optimizer.step()
            self.optimizer.zero_grad()

            # Update
            self.current_step += 1
            self.current_train_loss_avg = loss_avg / self.current_step

            if self.steps_per_epoch > 10:
                if self.current_step % (self.steps_per_epoch // 10) == 0:
                    print(f'Epoch={self.current_epoch},', \
                          f'step={self.current_step}:', \
                          f'loss={loss.item():>.4f}')
            else:
                print(f'Epoch={self.current_epoch},'\
                      f'step={self.current_step}:,'\
                      f'loss={loss.item():>.4f}')

        self.current_train_loss_avg = loss_avg / self.steps_per_epoch
        if self.train_output is not None:
            f = open(self.train_output, 'a')
            f.write(f'{self.current_epoch:5} {self.current_train_loss_avg:>9.4f}')
            f.write(f'\n')
            f.close()


            
    ### Validation loop
    def validate(self,
                 loader,
                 save_dir:str=None,
                 save_output:bool=False,
                 write_inputs=True,
                 write_targets=False
    ):
        N = len(loader.dataset)
        nT = loader.dataset.n_timepoint
        nD = loader.dataset.X
        
        loss_avg = 0.

        self.model.eval()
        self.model.zero_grad()

        for vol, idx in loader:
            # Synthesize images from input volume
            synth_class, synth_model = \
                random.choice(list(loader.dataset.synth_models_dict.items()))
            X, y = synth_model(vol.to(self.device))
            X, y = loader.dataset.base_augmentations([X, y])
            
            # Run
            with torch.no_grad():
                logits = self.model(X)
            loss = self.loss_fn(logits, y, gpu=True)

            loss_avg += loss.item()
            y_target = torch.argmax(y, dim=1)
            y_pred = torch.argmax(F.softmax(logits, dim=1), dim=1)

            if save_output and self.output_folder is not None:
                out_str = "epoch_" + \
                    str(self.current_epoch + 1).zfill(len(str(self.max_n_epochs)))
                output_folder_valid = os.path.join(self.output_folder, "valid_data",
                                                   epoch_str) if save_dir is None else save_dir
                
                self._save_output(output_folder_valid, loader.dataset,
                                  X, y_target, y_pred, idx,
                                  write_targets=write_targets, write_inputs=write_inputs)
        loss_avg = loss_avg / N

        if self.valid_output is not None:
            f = open(self.valid_output, 'a')
            f.write(f'{self.current_epoch} {loss_avg:>9.4f}')
            f.write(f'\n')
            f.close()
        
            
    
    ### Testing loop
    def test(self,
             loader,
             save_output:bool=False,
             save_dir:str=None,
             mode:str='train',
             write_inputs:bool=True,
             write_targets:bool=False,
             write_posteriors:bool=False,
             output_basename=None,
    ):
        save_dir = self.output_folder if save_dir is None \
            else os.path.join(self.output_folder, save_dir)

        dataset = loader.dataset
        N = len(dataset)
        nT = dataset.n_timepoint
        nD = dataset.X
        
        self.model.eval()

        for vol, idx in loader:
             # Synthesize images from input volume
            synth_class, synth_model = \
                random.choice(list(dataset.synth_models_dict.items()))
            X, y = synth_model(vol.to(self.device))
            X, y = dataset.base_augmentations([X, y])
                
            # Run
            with torch.no_grad():
                logits = self.model(X)
            posteriors = F.softmax(logits, dim=1)

            # Save data?
            if save_output:
                self._save_output(save_dir, loader.dataset,
                                  X, y, posteriors, idx,
                                  output_basename=output_basename,
                                  write_inputs=write_inputs,
                                  write_targets=write_targets,
                                  write_posteriors=write_posteriors)

            # Output loss
            if self.test_output is not None:
                loss = self.loss_fn(logits, y, gpu=True)
                idx_id = dataset.input_files[idx][0].split('/')[-1].split('.')[0]

                f = open(self.test_output, 'a')
                f.write(f'{idx_id} {loss.item():>.4f}')
                f.write(f'\n')
                f.close()

                


    # Run at the end of each epoch
    def _epoch_end(self):
        # Save model
        save_model_every_epoch = 20
        save_model = (self.current_epoch + 1) % 20 == 0
        if self.model_output is not None and save_model:
            torch.save({'epoch': self.current_epoch,
                        'model_state': self.model.state_dict(),
                        'optimizer_state': self.optimizer.state_dict(),
            }, self.model_output + "_last.tar")

        # Update/reset
        self.current_step = 0
        self.current_epoch += 1
        self.current_train_loss_avg = 0

        if self.current_epoch == self.switch_loss_on:
            self.loss_fn = self.loss_functions[1]
            print('Switching from', self.loss_functions[0].__name__, 'to', \
                  self.loss_functions[1].__name__, 'after ',
                  self.current_epoch, ' epochs')




    ### Function to write image data
    def _save_output(self, folder, dataset, inputs, target, output, idx,
                     output_basename=None, conform2orig=False,
                     write_inputs=False, write_targets=False, write_posteriors=False):

        if folder is not None:
            basename = output_basename if output_basename is not None else \
		dataset.input_files[idx][0].split("/")[-1].split(".")[0]

            folder = os.path.join(folder, basename)
            if not os.path.exists(folder): os.makedirs(folder, exist_ok=True)

            target_segmentation = torch.argmax(target, dim=1) if target is not None \
                else None
            output_segmentation = torch.argmax(output, dim=1)

            for t in range(inputs.shape[self.T]):
                # Posteriors
                if write_posteriors:
                    posterior = sf.Volume(np.squeeze(output[:, :, t, ...].cpu().numpy()))
                    posterior_path = \
                        os.path.join(folder, "_".join([basename, str(t), "posterior.mgz"]))
                    posterior.save(posterior_path)

                # Output segmentation
                output_path = \
                    os.path.join(folder, "_".join([basename, str(t), "prediction.mgz"]))
                dataset._save_output(output_segmentation[:, t, ...], output_path,
                                     dtype=np.int32, is_labels=True, convert_labels=True)

                # Target segmentation
                if write_targets and target is not None:
                    target_path = \
                        os.path.join(folder, "_".join([basename, str(t), "target.mgz"]))
                    dataset._save_output(target_segmentation[:, t, ...], target_path,
                                         dtype=np.int32, is_labels=True, convert_labels=True)

                # Input image
                if write_inputs:
                    for i in range(dataset.__n_input__()):
                        input_path = \
                            os.path.join(folder, "_".join([basename, str(t), "input.mgz"]))
                        dataset._save_output(inputs[:, i, t, ...], input_path,
                                             dtype=np.float32, is_labels=False)
