#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 28 14:46:18 2023

@author: tori
"""

import torch
import torchvision
import torchvision.transforms as T
from CustomCocoDataset import CustomCocoDataset
from drawBoxes import show_image_with_boxes, load_example_image, run_example, show_image_with_boxes_opencv

if __name__ == "__main__":
    
    data_name = 'data0'
    
    # path to your own data and coco file
    data_dir = '../../../learningStuff/data/' + data_name + '/'
    #data_dir = 'data/data1/coco/'
    
    data_dir_images = data_dir + 'images'
    data_dir_annotations = data_dir + 'annotations/instances_default.json'


    # create own Dataset
    dataset = CustomCocoDataset(root=data_dir_images,
                              annotation=data_dir_annotations)
    
    for data in dataset:
        img, info = data
        
        drawn_boxes = show_image_with_boxes(img, info["boxes"], info["labels"])
        input("Press Enter to continue...")
        
        
    