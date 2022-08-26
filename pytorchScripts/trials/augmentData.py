#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 16 16:01:11 2022

@author: tori
"""
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

import torch
import torchvision.transforms as T
from torchvision.utils import draw_bounding_boxes
import torchvision.transforms.functional as F

#%matplotlib inline

from CustomCocoDataset import CustomCocoDataset

def show(imgs):
    if not isinstance(imgs, list):
        imgs = [imgs]
    fig, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = F.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
        
        

data_dir = 'data/coco/images'
data_dir_annotations = 'data/coco/instances_default.json'
dataset = CustomCocoDataset(root=data_dir,
                          annotation=data_dir_annotations)

img, info = dataset[0]

#plt.imshow(  img  )

train_transforms = T.Compose([T.ToTensor(),
                              T.ConvertImageDtype(torch.uint8)]) 

train_transforms(img)


drawn_boxes = draw_bounding_boxes(train_transforms(img), info['boxes'], colors="green", width=10)
show(  drawn_boxes  )

seq = Sequence([T.RandomHorizontalFlip(1)])
img_, bboxes_ = seq(train_transforms(img), info['boxes'])

drawn_boxes = draw_bounding_boxes(train_transforms(img), info['boxes'], colors="green", width=10)
show(  drawn_boxes  )
