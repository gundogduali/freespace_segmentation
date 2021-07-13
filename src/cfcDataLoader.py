# -*- coding: utf-8 -*-
"""
Created on Tue Jul 13 21:25:26 2021

@author: AliG
"""

from torch.utils.data import Dataset,DataLoader

import pandas as pd
import numpy as np
import albumentations as A
import cv2

from train_test_split import split

class cfcDataset(Dataset):
    def __init__(self,df:pd.DataFrame,phase: str = 'test', transform = None):
        self.df = df
        self.phase = phase
        self.transform = transform
        
    def __len__(self):
        return self.df.shape[0]
    
    def __getitem__(self,idx):
        row = self.df.iloc[idx]
        id_ = row['ids']
        image_path = row['image_paths']
        mask_path = row['mask_paths']
        
        img = self.load_img(image_path)
        img = self.resize(img)
        
        if self.phase != 'test':
            mask = self.load_img(mask_path,is_mask = True)
            mask = self.resize(mask)
            mask = np.clip(mask,0,1)
            mask = self.preprocess_mask_labels(mask,2)
            
            if self.transform is not None:
                transformed = self.transform(image = img,
                                             mask = mask.astype(np.uint8))
                img = transformed['image']
                mask = transformed['mask']
                
                
            img = self.normalize(img)
            img = np.moveaxis(img,-1,0)
            mask = np.moveaxis(mask,-1,0)
            
            return {
                'Id':id_,
                'image':img,
                'mask':mask,
                }
        img = self.normalize(img)
        img = np.moveaxis(img,-1,0)
        
        return{
            'Id':id_,
            'image':img,
            }
    
    def load_img(self,file_path,is_mask:bool = False):
        if is_mask:
            image = cv2.imread(file_path,0)
        else:
            image = cv2.imread(file_path)
        return image
    
    def resize(self,image):
        image = cv2.resize(image,(224,224))
        return image
    
    def normalize(self,data:np.ndarray):
        data_min = np.min(data)
        return (data - data_min) / (np.max(data) - data_min)
    
    def preprocess_mask_labels(self,mask:np.ndarray,n_classes):
        if len(mask.shape) != 2:
            print("It should be same with the layer dimension, in this case it is 2")
            return
        
        encoded_data = np.zeros((*mask.shape, n_classes), dtype=np.int32)
        encoded_labels = [[0,1],[1,0]]
        
        for i in range(n_classes):
            bl_mat = mask[:,:] == i
            encoded_data[bl_mat] = encoded_labels[i]
        
        return encoded_data
    
train_transform = A.Compose(
    [
        A.RandomBrightnessContrast(p=0.75,brightness_limit = (-0.25, 0.25),contrast_limit = (-0.15, 0.4)),
        A.GaussianBlur(p = 0.25),   
        A.CropNonEmptyMaskIfExists(140,140,p = 0.33),
        A.HorizontalFlip(p = 0.5),
        A.GaussNoise(p = 0.35),
        A.Resize(224,224),
    ]
)

def get_dataloader(
        dataset:Dataset,
        csv_path:str,
        phase = str,
        batch_size: int = 1,
        ):
    df = pd.read_csv(csv_path)
    is_shuffle = False
    transform = None
    
    if phase != 'test':
        train_df,val_df = split(df)
        is_shuffle = True
        if phase == 'train':
            df = train_df
            train_ds = dataset(df,phase,train_transform)
            
            dataloader = DataLoader(
                train_ds,
                batch_size = batch_size,
                num_workers=0,
                pin_memory= True,
                shuffle = is_shuffle
                )
            return dataloader
        else:
            df = val_df
    dataset = dataset(df,phase,transform)
    
    dataloader = DataLoader(
        dataset,
        batch_size = batch_size,
        num_workers = 0,
        pin_memory = True,
        shuffle = is_shuffle
        )
    return dataloader


# dataloader = get_dataloader(cfcDataset,'data.csv','train',4)
# data = next(iter(dataloader))