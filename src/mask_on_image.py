# -*- coding: utf-8 -*-
"""
Created on Fri Jun 18 00:54:18 2021

@author: AliG
"""

import os
import cv2
import tqdm
import numpy as np
import matplotlib.pyplot as plt
from constant import *

mask_list = os.listdir(MASK_DIR)

for f in mask_list:
    if f.startswith('.'):
        mask_list.remove(f)
        
for mask_name in tqdm.tqdm(mask_list):
    mask_name_without_ex = mask_name.split('.')[0]
    
    mask_path = os.path.join(MASK_DIR,mask_name)
    image_path = os.path.join(IMAGE_DIR,mask_name_without_ex+'.jpg')
    image_out_path = os.path.join(IMAGE_OUT_DIR,mask_name)
    
    mask = cv2.imread(mask_path,0).astype(np.uint8)
    image = cv2.imread(image_path).astype(np.uint8)
    
    cpy_image = image.copy()
    image[mask==1,:] = (255,0,0)
    opac_image = (image/2 + cpy_image/2).astype(np.uint8)
    
    cv2.imwrite(image_out_path,opac_image)
    
    if VISUALIZE:
        plt.figure()
        plt.imshow(opac_image)
        plt.show()