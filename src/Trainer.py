# -*- coding: utf-8 -*-
"""
Created on Sun Jul 18 18:01:54 2021

@author: AliG
"""

import time

import torch
import torch.nn as nn

from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from cfcDataLoader import get_dataloader,cfcDataset

import matplotlib.pyplot as plt
from metrics import Meter,DiceBCELoss

import pandas as pd
from tqdm import tqdm

from model import UNet



class Trainer:
    def __init__(self,
                 net: nn.Module,
                 dataset: torch.utils.data.Dataset,
                 criterion: nn.Module,
                 lr:float,
                 batch_size:int,
                 num_epochs: int,
                 csv_path:str
                 ):
        ### INITIALIZATION ####
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print('device: ',self.device)
    
        self.net = net
        self.net = self.net.to(self.device)
        self.criterion = criterion
        self.optimizer = Adam(self.net.parameters(),lr = lr)
        self.scheduler = ReduceLROnPlateau(self.optimizer,mode='min',patience = 2,verbose = True)
        self.phases = ['train','val']
        self.num_epochs = num_epochs
    
        self.dataloaders = {
            phase: get_dataloader(
                dataset = dataset,
                csv_path = csv_path,
                phase= phase,
                batch_size = batch_size,
                )
            for phase in self.phases
            }
        self.best_loss = float('inf')
        self.losses = {phase: [] for phase in self.phases}
        self.dice_scores = {phase: [] for phase in self.phases}
        self.iou_scores = {phase: [] for phase in self.phases}
    
    def run(self):
        for epoch in range(self.num_epochs):
            self.do_epoch(epoch,'train')
            with torch.no_grad():
                val_loss = self.do_epoch(epoch,'val')
                self.scheduler.step(val_loss)
                
            if val_loss < self.best_loss:
                print(f"\n{'#'*20}\n CHECKPOINT \n{'#'*20}\n")
                self.best_loss = val_loss
                torch.save(self.net_state_dict(),'best_model.pth')
                
        self.save_train_history()
    
    def do_epoch(self,epoch:int,phase:str):
        print(f"{phase} epoch: {epoch} | time: {time.strftime('%H:%M:%S')}\n")
        
        self.net.train() if phase == 'train' else self.net.eval()
        
        meter = Meter()
        dataloader = self.dataloaders[phase]
        total_batches = len(dataloader)
        running_loss = 0.0
        
        self.optimizer.zero_grad()
        for itr,data in enumerate(tqdm(dataloader)):
            images, targets = data['image'],data['mask']
            loss,logits = self.compute_loss_and_logits(images,targets)
            loss = loss
            
            if phase == 'train':
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
            running_loss += loss.item()
            meter.update(logits.detach().cpu(), targets.detach().cpu())
            
        epoch_loss = (running_loss) / total_batches
        epoch_dice , epoch_iou = meter.get_metrics()
            
        self.losses[phase].append(epoch_loss)
        self.dice_scores[phase].append(epoch_dice)
        self.iou_scores[phase].append(epoch_iou)
            
        return epoch_loss
    
    def compute_loss_and_logits(self, images: torch.Tensor,targets:torch.Tensor):
        images = images.to(self.device,dtype = torch.float)
        targets = targets.to(self.device,dtype = torch.float)
        
        logits = self.net(images)
        
        loss = self.criterion(logits,targets)
        return loss,logits
    
    def save_train_history(self):
        """writing model weights and training logs to files."""
        torch.save(self.net.state_dict(),
                   f"last_epoch_model.pth")

        logs_ = [self.losses, self.dice_scores, self.jaccard_scores]
        log_names_ = ["_loss", "_dice", "_jaccard"]
        logs = [logs_[i][key] for i in list(range(len(logs_)))
                         for key in logs_[i]]
        log_names = [key+log_names_[i] 
                     for i in list(range(len(logs_))) 
                     for key in logs_[i]
                    ]
        pd.DataFrame(
            dict(zip(log_names, logs))
        ).to_csv("train_log.csv", index=False)
        
if __name__ == '__main__':
    model = UNet()
    
    trainer = Trainer(net=model,
                  dataset=cfcDataset,
                  criterion=DiceBCELoss(),
                  lr=3e-4,
                  batch_size=4,
                  num_epochs=35,
                  csv_path = 'fold.csv',)
    
    trainer.run()