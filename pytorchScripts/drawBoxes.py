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

import cv2

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

"""
@param img is tensor format
@boxes Tensor of size [N,4] containing bounding boxes coordinates in (xmin, ymin, xmax, ymax) format.
"""    
def show_image_with_boxes_opencv(img, boxes, labels=None):
    
    #box from model has format: [x_0, y_0, x_1, y_1]
    #cv2.rectangle(cv_image, (x0, y1), (x1, y0), (255,0,0), 2)

    cv_image = torchvision.transforms.functional.convert_image_dtype(
                img.cpu(), torch.uint8).numpy().transpose([1,2,0])

    cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
    
    for box in boxes:
        cv2.rectangle(cv_image, (round(box[0].item()), round(box[1].item())),
                                (round(box[2].item()), round(box[3].item())),
                                (255,0,0), 2)

    
    cv2.imshow("test_boxes", cv_image)
    cv2.waitKey()
    
    #TODO label draw


    
    
def load_example_image():
    
    data_dir = 'data/data2/images'
    data_dir_annotations = 'data/data2/coco/annotations/instances_default.json'
    dataset = CustomCocoDataset(root=data_dir,
                              annotation=data_dir_annotations)
    
    img, info = dataset[0]
    
    #show_image_with_boxes(img, info["boxes"], info["labels"])
    
    return img, info
    
def run_example():
    
    img, info = load_example_image()    
    
    transform_chain = T.Compose([T.PILToTensor(),
                                 T.ConvertImageDtype(torch.float)
                                 ])
    
    img_tensor = transform_chain(img)

    boxes = torch.Tensor([[200, 100, 650, 500], [100,400,800,560]])
    show_image_with_boxes_opencv(img_tensor, boxes)
    
    
    
    
    
