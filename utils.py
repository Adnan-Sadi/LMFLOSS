# -*- coding: utf-8 -*-
"""
@author: Adnan-Sadi
"""
import os
import numpy as np
import configparser

"""
Function gettign the number of images per class from
a particular directory

Args:
    path: data directory
   
Returns:
    img_per_cls: a list of image counts per class
"""
def get_img_num_per_cls(path):
    folders = os.listdir(path)
    img_per_cls = []

    for folder in folders:
        img_path = path +'\\'+ folder
        img_count = len(os.listdir(img_path))
        img_per_cls.append(img_count)

    return img_per_cls


"""
Function getting the m_list value of the LDAM loss. The m_list
values are used as class weights in this study.  

Args:
    cls_num_list: list of image counts per class
    max_m: 'maximum margin' hyper parameter used in lDAM loss, 
            defaults to 0.5 
    
Returns:
    m_list: m_list(class_weights/margins) value
"""
def get_mlist(cls_num_list, max_m=0.5):
    m_list = 1.0 / np.sqrt(np.sqrt(cls_num_list))
    m_list = m_list * (max_m / np.max(m_list))
    
    return m_list


"""
Function for setting dataset directory and model save directory.  

Args:
    name: name of the dataset
    
Returns:
   data_dir: dataset directory
   model_save_dir: model save directory
"""
def choose_dataset(name):
    config = configparser.ConfigParser()
    config.read_file(open(r'config.ini'))
    

    data_dir = config.get('dataset-paths', name)
    model_save_dir = config.get('model-save-paths', name)
    
    return data_dir, model_save_dir

"""
Function for getting the paramter count of a pytorch model

Args:
    model: pytorch model
"""
def parameter_count(model):
    total_params = sum(p.numel() for p in model.parameters())
    print(f'{total_params:,} total parameters.')
    total_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad)
    print(f'{total_trainable_params:,} training parameters.')
    
    return


"""
Function for getting class weights from the bwcce loss separately  

Args:
    img_per_cls: list of image counts per class
"""
def get_bwcce_wts(img_per_cls):
    K = len(img_per_cls)-1 
    beta_values = [(1 - (x / sum(img_per_cls))) / K for x in img_per_cls]
    print()
    print(beta_values)
    
    return beta_values


