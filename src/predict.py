# -*- coding: utf-8 -*-
"""
Created on Tue Aug  3 13:49:31 2021

@author: AliG
"""

import os
import cv2
import torch
import numpy as np
import torch.nn.functional as F
import tqdm

from model import UNet
from constant import IMAGE_DIR
from cfcDataLoader import get_dataloader,cfcDataset

def id_to_image(idx):
    path = os.path.join(IMAGE_DIR + f'\{idx}.jpg')
    image = cv2.imread(path)
    image = cv2.resize(image,(1280,720))
    return image

def write_mask_on_image(image,mask,idx):
    save_file_name= "../data/predicts"
    if not os.path.exists(save_file_name):
        os.mkdir(save_file_name)
        
    mask_image = image.copy()
    mask_ind = mask == 0
    mask_image[mask_ind,:] = (0,0,255)
    opac_image = (image/2 + mask_image/2).astype(np.uint8)
    cv2.imwrite(cv2.imwrite(os.path.join(save_file_name,idx+'.png'),opac_image))
    
def predict(model,dataloader):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    model.eval()
    for data in tqdm.tqdm(dataloader):
        id_,img = data['Id'][0],data['image']
        img = img.to(device,dtype = torch.float)
        with torch.no_grad():
            output = model(img)
            output = F.upsample_bilinear(output,size = (720,1280))
            output = output.data.cpu().numpy()
            for mask in zip(output):
                image = id_to_image(id_)
                mask_out = np.argmax(mask,axis = 0)
                write_mask_on_image(image, mask_out, id_)
           
                
dict_path = '../data/models/best_model.pth'
test_csv = '../data/test.csv'

if __name__ == '__main__':
    model = UNet()
    model.load_state_dict(torch.load(dict_path))
    test_dataloader = get_dataloader(cfcDataset,test_csv,'test',batch_size = 1)
    predict(model,test_dataloader)