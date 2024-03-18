import os
import time
import pathlib as Path
import numpy as np

import torch
import torch.optim
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast


class Segment:
    def __init__(self, model, optimizer, scheduler, loss_function,
                 max_n_epochs, output_folder, device, start_full_aug_on=None,
                 metrics_train=[], metrics_valid=[], metrics_test=[],
    ):
        super().__init__()

        self.T = 2 # temporal dimension index
        self.device = device
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_fn = loss_function

        self.valid_loss = [None, None]
        self.current_epoch = 0
        self.current_train_loss_avg = 0

        self.max_n_epochs = max_n_epochs
        self.start_full_aug_on = start_full_aug_on if start_full_aug_on is not None else self.max_n_epochs
        self.output_folder = output_folder
        self.print_training_metrics_on_epoch = 1
        
        self.metrics_train = metrics_train
        self.metrics_valid = metrics_valid
        self.metrics_test = metrics_test
        
        if output_folder is not None:
            self.train_output = os.path.join(output_folder, "training_log.txt")
            self.valid_output = os.path.join(output_folder, "validation_log.txt")
            self.test_output = os.path.join(output_folder, "testing_log.txt")
            self.model_output = os.path.join(output_folder, "model")
        else:
            self.train_output = None
            self.valid_output = None
            self.test_output = None
            self.model_output = None

        self.epoch_st = None
        self.epoch_et = 0
        self.train_step_time = 0
            
            
            
    ### Training loop
    def train(self, loader, save_output:bool=False):
        self.epoch_st = time.time() if self.epoch_st is None else time.time()
        N = len(loader.dataset)
        M = len(self.metrics_train)
        metrics = np.zeros((M))
        self.current_train_loss_avg = 0.0
        self.train_step_time = 0.0
        
        if self.current_epoch < self.start_full_aug_on:
            augmentation = loader.dataset.base_augmentation
        else:
            augmentation = loader.dataset.full_augmentation
        
        self.model.zero_grad()
        self.model.train()

        for X, y, idx in loader:
            X, y = augmentation(X.to(self.device), y.to(self.device))
            torch.cuda.empty_cache()
            
            self.optimizer.zero_grad()
            logits = self.model(X)
            loss = self.loss_fn(logits, y)
            
            loss.backward()
            self.current_train_loss_avg += loss.item()
            self.optimizer.step()

            y_target = torch.argmax(y, dim=1)
            y_pred = torch.argmax(F.softmax(logits, dim=1), dim=1)
            
            if self.metrics_train is not None:
                metrics = [metrics[m] + self.metrics_train[m](y_pred, y_target) for m in range(M)]
                
            et = time.time()
            self.train_step_time += (et - st)
            
        if save_output and self.output_folder is not None:
            output_folder_train = os.path.join(self.output_folder, "train_data",
                                               "epoch_" + str(self.current_epoch).zfill(len(str(self.max_n_epochs))))
            self._save_output(output_folder_train, loader.dataset, X, y_target, y_pred, idx)

        self.current_train_loss_avg /= N
        self.train_step_time /= N
        metrics = [metrics[m] / N for m in range(M)]

        if self.train_output is not None:
            f = open(self.train_output, 'a')
            f.write(f'{self.current_epoch} {self.current_train_loss_avg:>.4f}')
            for m in range(M):
                f.write(f' {metrics[m]:>.5f}')
            f.write(f'\n')
            f.close()            


            
    ### Validation loop
    def validate(self, loader, save_output:bool=False):
        N = len(loader.dataset)
        M = len(self.metrics_valid)
        loss_avg = 0
        metrics = np.zeros((M))
        
        if self.current_epoch < self.start_full_aug_on:
            augmentation = loader.dataset.base_augmentation
        else:
            augmentation = loader.dataset.full_augmentation
            
        self.model.eval()

        for X, y, idx in loader:
            X, y = augmentation(X.to(self.device), y.to(self.device))

            with torch.no_grad():
                logits = self.model(X)
                loss = self.loss_fn(logits, y)
                loss_avg += loss.item()

            y_target = torch.argmax(y, dim=1)
            y_pred = torch.argmax(F.softmax(logits, dim=1), dim=1)

            if self.metrics_valid is not None:
                metrics = [metrics[m] + self.metrics_valid[m](y_pred, y_target) for m in range(M)]

        if save_output and self.output_folder is not None:
            output_folder_valid = os.path.join(self.output_folder, "valid_data",
                                               "epoch_" + str(self.current_epoch).zfill(len(str(self.max_n_epochs))))
            self._save_output(output_folder_valid, loader.dataset, X, y_target, y_pred, idx)

        loss_avg = loss_avg / N
        self.valid_loss = [self.valid_loss[1], loss_avg]
        metrics = [metrics[m] / N for m in range(M)]

        if self.valid_output is not None:
            f = open(self.valid_output, 'a')
            f.write(f'{self.current_epoch} {loss_avg:>.4f}')
            for m in range(M):
                f.write(f' {metrics[m]:>.5f}')
            f.write(f'\n')
            f.close()

            
    
    ### Testing loop
    def test(self, loader, metrics_list, save_output:bool=False):
        N = len(loader.dataset)
        M = len(self.metrics_test)

        loss_idx = 0.0
        metrics_idx = np.zeros((M, 1))
        augmentation = loader.dataset.base_augmentation

        self.model.eval()

        for X, y, idx in loader:
            X, y = augmentation(X.to(self.device), y.to(self.device))
            
            with torch.no_grad():
                logits = self.model(X)
                loss = self.loss_fn(logits, y)
                loss_idx = loss.item()

            y_target = torch.argmax(y, dim=1)
            y_pred = torch.argmax(F.softmax(logits, dim=1), dim=1)

            if self.metrics_test is not None:
                metrics_idx = [self.metrics_test[m](y_pred, y_target) for m in range(M)]

            if save_output and self.output_folder is not None and idx < 3:
                self._save_output(os.path.join(self.output_folder, "test_data"),
                                  loader.dataset, X, y_target, y_pred, idx)

            if save_output and self.test_output is not None:
                sid = "".join(loader.dataset.label_files[idx][0].split("/")[-1].split(".")[0:2])[8:12]
                f = open(self.test_output, 'a')
                f.write(f'{sid} {loss_idx:>.4f}')
                for m in range(M):
                    f.write(f' {metrics_idx[m]:>.4f}')
                f.write(f'\n')
                f.close()



    # Run at the end of each epoch
    def _epoch_end(self):
        # Save model
        if self.model_output is not None:
            torch.save({'epoch': self.current_epoch,
                        'model_state': self.model.state_dict(),
                        'optimizer_state': self.optimizer.state_dict(),
                        'valid_loss': self.valid_loss[1]
                        }, self.model_output + "_last")
            if self.valid_loss[0] is not None:
                if self.valid_loss[1] < self.valid_loss[0]:
                    torch.save({'epoch': self.current_epoch,
                                'model_state': self.model.state_dict(),
                                'optimizer_state': self.optimizer.state_dict(),
                                'valid_loss': self.valid_loss[1]
                    }, self.model_output + "_best")

        # Print stuff
        self.epoch_et = time.time()
        if self.current_epoch % self.print_training_metrics_on_epoch == 0:
            print(f'Epoch={self.current_epoch} : ' + \
                  f'avg_train_loss={self.current_train_loss_avg:>.4f}, ' + \
                  f'last_lr={self.scheduler.get_last_lr()[0]:>.5f}, ' + \
                  f'avg_step_time={self.train_step_time:>.2f}s ' + \
                  f'epoch_time={((self.epoch_et - self.epoch_st)/60):>.1f}min'
            )
            
        # Update/reset
        self.scheduler.step()
        self.current_epoch += 1
        self.epoch_st = time.time()
        


    ### Function to write image data
    def _save_output(self, folder, dataset, inputs, target, output, idx):
        if folder is not None:
            for t in range(torch.squeeze(output).shape[0]):
                basename = dataset.label_files[idx][t].split("/")[-1].split(".")[0:2]
                if not os.path.exists(folder):
                    os.makedirs(folder, exist_ok=True)
                    
                for i in range(len(dataset.image_files[idx][t])):
                    input_str = dataset.image_files[idx][t][i].split(".")[-2:]
                    input_path = os.path.join(folder, ".".join(basename) + "." + ".".join(input_str))
                    dataset._save_output(inputs[:,i,t,...], input_path, dtype=np.float32)

                target_path = os.path.join(folder, ".".join(basename) + ".target.mgz")
                dataset._save_output(target[:,t,...], target_path, dtype=np.int32, is_onehot=True)
                output_path = os.path.join(folder, ".".join(basename) + ".output.mgz")
                dataset._save_output(output[:,t,...], output_path, dtype=np.int32, is_onehot=True)
