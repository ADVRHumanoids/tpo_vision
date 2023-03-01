#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 28 18:07:19 2023

@author: tori
"""
import os

import argparse

parser = argparse.ArgumentParser(description='')
parser._action_groups.pop()
required = parser.add_argument_group('required arguments')
optional = parser.add_argument_group('optional arguments')
optional.add_argument('--batch', type=str, default="1")
optional.add_argument('--epochs', type=str, default="3")
optional.add_argument('--weights', type=str, default="yolov5s6.pt")

args, unknown = parser.parse_known_args()

tpo_dir = os.getcwd() + '/../../../'

yolo_train_file = tpo_dir + '/YoloTutorial/yolov5/train.py'
img_size = '1280' #if rectangular images, put the bigger dimension. --rect option is another thing, it does not make use of mosaic stuff who knows what it is
batch_size = args.batch
epochs = args.epochs
data_file = tpo_dir + 'learningStuff/data/data2/yolo/data.yaml'
weights = args.weights #the initial network to start with

command = " ".join(['python', 
                    yolo_train_file,
                    '--img', img_size,
                    '--batch', batch_size,
                    '--epochs', epochs,
                    '--data', data_file,
                    '--weights', weights,   
                    ])


os.system(command)
