#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  6 10:31:38 2022

@author: tori
"""

from TrainModel  import run

if __name__ == "__main__":
    
    #run(batch_size=8, num_epochs=30, model_type = 'fasterrcnn_mobilenet_low', val_percentage=0.20, test_percentage=0.10)
    #run(batch_size=8, num_epochs=10, model_type = 'fasterrcnn_mobilenet_low', val_percentage=0.20, test_percentage=0.10)
    
    #run(batch_size=8, num_epochs=10, model_type = 'faster_rcnn_v1', val_percentage=0.20, test_percentage=0.10)
    #run(batch_size=8, num_epochs=30, model_type = 'faster_rcnn_v1', val_percentage=0.20, test_percentage=0.10)
    
   # run(batch_size=8, num_epochs=10, model_type = 'faster_rcnn_v2', val_percentage=0.20, test_percentage=0.10)
    #run(batch_size=8, num_epochs=30, model_type = 'faster_rcnn_v2', val_percentage=0.20, test_percentage=0.10)
    
    #run(batch_size=1, num_epochs=10, model_type = 'faster_rcnn_v2', val_percentage=0.4, test_percentage=0, data_name="laser_v3")
    #run(batch_size=8, num_epochs=30, model_type = 'fasterrcnn_mobilenet_high', val_percentage=0.20, test_percentage=0.10)
   
   #run(batch_size=8, num_epochs=100, model_type = 'faster_rcnn_v2', val_percentage=0.20, test_percentage=0.10)
   
    #Original example 
    #run(batch_size=1, num_epochs=10, model_type = 'faster_rcnn_v1', val_percentage=0, test_percentage=0, data_name="data0")
    
    #The fasterRCNNCoco2.py https://github.com/ADVRHumanoids/tpo_vision/blob/0f2806938a70e99494d1b29012c6d22efcb8d7cf/pytorchScripts/fasterRCCNCoco2.py
    run(batch_size=2, num_epochs=5, model_type = 'faster_rcnn_v1', val_percentage=0.18, test_percentage=0, data_name="data2")
   
   
