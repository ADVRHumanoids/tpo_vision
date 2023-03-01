#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 28 18:07:19 2023

@author: tori
"""
import sys
import os
sys.path.insert(1, '/home/tori/TelePhysicalOperation/YoloTutorial/yolov5/')
sys.path.insert(1, '/home/tori/TelePhysicalOperation/YoloTutorial/yolov5/utils')

import torch


#model = torch.hub.load('ultralytics/yolov5', 'yolov5s', autoshape=False, pretrained=True)  # load pretrained but with something to train

yolo_train_file = '/home/tori/TelePhysicalOperation/YoloTutorial/yolov5/train.py'
img_size = '1280' #if rectangular images, put the bigger dimension. --rect option is another thing, it does not make use of mosaic stuff who knows what it is
batch_size = '1'
epochs = '3'
data_file = '/home/tori/TelePhysicalOperation/learningStuff/data/data2/yolo/data.yaml'
weights = 'yolov5s6.pt' #the initial network to start with

command = " ".join(['python', 
                    yolo_train_file,
                    '--img', img_size,
                    '--batch', batch_size,
                    '--epochs', epochs,
                    '--data', data_file,
                    '--weights', weights,   
                    ])


os.system(command)
