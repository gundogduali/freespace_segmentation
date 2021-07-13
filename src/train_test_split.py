# -*- coding: utf-8 -*-
"""
Created on Tue Jul 13 21:04:06 2021

@author: AliG
"""

import os
import json
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from constant import IMAGE_DIR,MASK_DIR,JSON_DIR

def make_df():
    image_list = os.listdir(IMAGE_DIR)
    image_name_list = []
    image_paths = []
    mask_paths = []
    json_paths = []
    scene_types = []
    
    for image in image_list:
        image_name = image.split('.')[0]
        
        image_path = os.path.join(IMAGE_DIR + f'\{image}')
        mask_path = os.path.join(MASK_DIR + f'\{image_name}.png')
        json_path = os.path.join(JSON_DIR + f'\{image_name}.png.json')
        
        json_paths.append(json_path)
        image_name_list.append(image_name)
        image_paths.append(image_path)
        mask_paths.append(mask_path)
        
    for json_path in json_paths:
        json_file = open(json_path,'r')
        
        json_dict = json.load(json_file)
        for obj in json_dict['tags']:
            if obj['name'] == 'Scene Type':
                scene_types.append(obj['value'])
    
    data = {'ids':image_name_list,'image_paths':image_paths,'mask_paths':mask_paths,'scene_types':scene_types}
    
    df = pd.DataFrame(data)
    df.to_csv('train_data.csv',index=False)

def split(df):
    skf = StratifiedKFold(n_splits = 7,random_state = 69,shuffle=True)
    
    for i, (train_index, val_index) in enumerate(
        skf.split(df, df["scene_types"])
        ):
        df.loc[val_index, "fold"] = i
        
    train_df = df.loc[df['fold'] != 0].reset_index(drop = True)
    val_df = df.loc[df['fold'] == 0].reset_index(drop = True)
    
    return train_df,val_df

# make_df()
# tr,val = split(df)