#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 19 10:06:19 2022

@author: tori
"""
import torch
import torchvision
from torchvision.io import read_image
from torchvision.utils import draw_bounding_boxes
import torchvision.transforms as T
from PIL import Image


import sys
from CustomCocoDataset import CustomCocoDataset

import matplotlib.pyplot as plt



"""
@param img png or jpg (as taken from dataset loaded)
@boxes Tensor of size [N,4] containing bounding boxes coordinates in (xmin, ymin, xmax, ymax) format.
"""
def show_image_with_boxes(img, boxes, labels=None):
    
    # img = read_image(image_file_name)
    # img = draw_bounding_boxes(img, boxes, width=3, colors=(255,255,0), labels=labels)
    # img = torchvision.transforms.ToPILImage()(img)
    # img.show()
    
    if labels is not None:
        assert len(boxes.shape) == (len(labels.shape)+1)
        
    if len(boxes.shape) < 2:
        boxes = boxes.unsqueeze(0)
        if labels is not None:
            labels = labels.unsqueeze(0)
 
    
    transformToTensor = T.Compose([
        T.PILToTensor()
    ])
    drawn_boxes = draw_bounding_boxes(transformToTensor(img), boxes, colors="green", width=2)
    #drawn_boxes.show()
    plt.imshow(drawn_boxes.permute(1, 2, 0))
    
    
def load_example_image():
    
    data_dir = 'data2/images'
    data_dir_annotations = 'data2/coco/annotations/instances_default.json'
    dataset = CustomCocoDataset(root=data_dir,
                              annotation=data_dir_annotations)
    
    img, info = dataset[0]
    
    #show_image_with_boxes(img, info["boxes"], info["labels"])
    
    return img, info
    
