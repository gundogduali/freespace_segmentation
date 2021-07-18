# -*- coding: utf-8 -*-
"""
Created on Sun Jul 18 17:09:37 2021

@author: AliG
"""

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

class DiceBCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        
        inputs = F.sigmoid(inputs)       
        
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()                            
        dice_loss = 1 - (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        Dice_BCE = BCE + dice_loss
        
        return Dice_BCE
    
    
    
def dice(inputs:torch.Tensor,targets:torch.Tensor,smooth:float = 0.5):
    inputs = F.sigmoid(inputs)
    
    inputs = inputs.view(-1)
    targets = targets.view(-1)
    
    intersection = (inputs * targets).sum()
    dice = (2 * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
    return dice
    
def IoU(inputs:torch.Tensor,targets:torch.Tensor,smooth:float = 0.5):
    inputs = F.sigmoid(inputs)       
        
    inputs = inputs.view(-1)
    targets = targets.view(-1)
        
    intersection = (inputs * targets).sum()
    total = (inputs + targets).sum()
    union = total - intersection 
        
    IoU = (intersection + smooth)/(union + smooth)
                
    return IoU
    
class Meter:
    def __init__(self,smooth:float = 0.5):
        self.smooth:float = smooth
        self.dice_scores:list = []
        self.iou_scores:list = []
            
    def update(self,logits:torch.Tensor,targets:torch.Tensor):
        dice_score = dice(logits,targets,self.smooth)
        iou = IoU(logits,targets,self.smooth)
            
        self.dice_scores.append(dice_score)
        self.iou_scores.append(iou)
            
    def get_metrics(self) -> np.ndarray:
        dice = np.mean(self.dice_scores)
        iou = np.mean(self.iou_scores)
        return dice,iou
        