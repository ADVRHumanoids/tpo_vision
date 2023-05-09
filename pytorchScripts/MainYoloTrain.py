#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 28 18:07:19 2023

@author: tori
"""

import os

if __name__ == "__main__":

	os.system('python YoloTrain.py --batch 8 --epochs 100 --weights yolov5s6 --data laser_v3')
	os.system('python YoloTrain.py --batch 8 --epochs 200 --weights yolov5s6 --data laser_v3')
	
	os.system('python YoloTrain.py --batch 8 --epochs 100 --weights yolov5m6 --data laser_v3')	
	os.system('python YoloTrain.py --batch 8 --epochs 200--weights yolov5m6 --data laser_v3')
	
	os.system('python YoloTrain.py --batch 8 --epochs 100 --weights yolov5l6 --data laser_v3')	
	os.system('python YoloTrain.py --batch 8 --epochs 200 --weights yolov5l6 --data laser_v3')
