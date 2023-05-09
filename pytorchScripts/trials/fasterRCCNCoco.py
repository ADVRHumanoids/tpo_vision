#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 16 10:46:19 2022

@author: tori

example taken from https://medium.com/fullstackai/how-to-train-an-object-detector-with-your-own-coco-dataset-in-pytorch-319e7090da5
"""
import os
import torch
import torch.utils.data
import torchvision
from PIL import Image
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

import sys
sys.path.insert(1, '../')
from CustomCocoDataset import *

from engine import train_one_epoch, evaluate
import utils

def get_transform(train=False):
    custom_transforms = []
    custom_transforms.append(torchvision.transforms.ToTensor())
    if train:
        x=3
        #custom_transforms.append(torchvision.transforms.RandomHorizontalFlip())
        #custom_transforms.append(torchvision.transforms.RandomVerticalFlip())
        #custom_transforms.append(torchvision.transforms.RandomAdjustSharpness(1.3))
        #custom_transforms.append(torchvision.transforms.RandomAdjustSharpness(0.7))
        #custom_transforms.append(torchvision.transforms.RandomAutocontrast())
    return torchvision.transforms.Compose(custom_transforms)


# path to your own data and coco file
#train_data_dir = 'data/coco/images'
#train_coco = 'data/coco/instances_default.json'
train_data_dir = '/home/tori/TelePhysicalOperation/learningStuff/data/data0/images'
train_coco = '/home/tori/TelePhysicalOperation/learningStuff/data/data0/annotations/instances_default.json'


# create own Dataset
my_dataset = CustomCocoDataset(root=train_data_dir,
                          annotation=train_coco,
                          transforms=get_transform()
                          )

# collate_fn needs for batch
def collate_fn(batch):
    return tuple(zip(*batch))

# Batch size
train_batch_size = 1

# own DataLoader
data_loader = torch.utils.data.DataLoader(my_dataset,
                                          batch_size=train_batch_size,
                                          shuffle=True,
                                          num_workers=4,
                                          collate_fn=collate_fn)

# select device (whether GPU or CPU)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# DataLoader is iterable over Dataset
for imgs, annotations in data_loader:
    imgs = list(img.to(device) for img in imgs)
    annotations = [{k: v.to(device) for k, v in t.items()} for t in annotations]
    print(annotations)
   

def get_model_instance_segmentation(num_classes):
    # load an instance segmentation model pre-trained pre-trained on COCO
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False)
    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model
    

# 2 classes; Only target class or background
num_classes = 2
num_epochs = 10
model = get_model_instance_segmentation(num_classes)

# move model to the right device
model.to(device)
    
# parameters
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

len_dataloader = len(data_loader)

for epoch in range(num_epochs):
    model.train()
    i = 0    
    for imgs, annotations in data_loader:
        i += 1
        imgs = list(img.to(device) for img in imgs)
        annotations = [{k: v.to(device) for k, v in t.items()} for t in annotations]
        loss_dict = model(imgs, annotations)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        print(f'Iteration: {i}/{len_dataloader}, Loss: {losses}')

torch.save(model, "bohboh.pt")
