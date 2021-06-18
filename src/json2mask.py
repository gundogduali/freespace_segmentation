# -*- coding: utf-8 -*-
"""
Created on Thu Jun 17 17:19:50 2021

@author: AliG
"""

import os
from constant import JSON_DIR,MASK_DIR
import cv2

json_list = os.listdir(JSON_DIR)

import tqdm

# iterator_example =range(10000000)

# for i in tqdm.tqdm(iterator_example):
#     pass

import json
import numpy as np

for json_name in tqdm.tqdm(json_list):
    json_path = os.path.join(JSON_DIR,json_name)
    json_file = open(json_path,'r')
    
    json_dict = json.load(json_file)
    
    
    mask = np.zeros((json_dict['size']['height'],json_dict['size']['width']),dtype=np.uint8)
    mask_path = os.path.join(MASK_DIR,json_name[:-9]+".png")
    
    for obj in json_dict['objects']:
        if obj['classTitle']=='Freespace':
            mask = cv2.fillPoly(mask,np.array([obj['points']['exterior']]),color=1)
            if len(obj['points']['interior']) != 0:
                print(json_name)
                mask = cv2.fillPoly(mask,np.array([obj['points']['interior']]),color=0)
            
    cv2.imwrite(mask_path,mask.astype(np.uint8))