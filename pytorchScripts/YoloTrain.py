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
optional.add_argument('--weights', type=str, default="yolov5s6")
optional.add_argument('--data', type=str, default="laser_v3")

args, unknown = parser.parse_known_args()

tpo_dir = os.getcwd() + '/../../../'

yolo_train_file = tpo_dir + 'YoloTutorial/yolov5/train.py'
img_size = '1280' #if rectangular images, put the bigger dimension. --rect option is another thing, it does not make use of mosaic stuff who knows what it is
batch_size = args.batch
epochs = args.epochs
data_file = tpo_dir + 'learningStuff/data/' + args.data +'/yolo/data.yaml'
weights = args.weights + '.pt' #the initial network to start with

model_out_name = (args.weights + "_e" + epochs + "_b"+ batch_size + "_tvt" + 
               str(30) + 
               str(20) + 
               str(10) +
               "_" + args.data)

command = " ".join(['python', 
                    yolo_train_file,
                    '--img', img_size,
                    '--batch', batch_size,
                    '--epochs', epochs,
                    '--data', data_file,
                    '--weights', weights, 
                    '--name', model_out_name  
                    ])                             

os.system(command)

os.rename(tpo_dir + 'YoloTutorial/yolov5/runs/train/'+model_out_name+'/weights/best.pt',
		  tpo_dir + 'YoloTutorial/yolov5/runs/train/'+model_out_name+'/weights/'+model_out_name+'.pt')    
