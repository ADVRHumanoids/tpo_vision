#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  8 13:00:25 2022

@author: tori

From the testing part of https://www.kaggle.com/code/havinath/object-detection-using-pytorch-training
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
from ensemble_boxes import weighted_boxes_fusion

detection_threshold = 0.5
results = []
outputs = []
test_images = []

def make_ensemble_predictions(images, models):
    result = []
    for net in models:
        net.eval()
        outputs = net(images)
        result.append(outputs)
    return result

def run_wbf(predictions, image_index, image_size=1024, iou_thr=0.55, skip_box_thr=0.4, weights=None):
    boxes = [prediction[image_index]['boxes'].data.cpu().numpy()/(image_size-1) for prediction in predictions]
    scores = [prediction[image_index]['scores'].data.cpu().numpy() for prediction in predictions]
    labels = [np.ones(prediction[image_index]['scores'].shape[0]) for prediction in predictions]
    boxes, scores, labels = weighted_boxes_fusion(boxes, scores, labels, weights=None, iou_thr=iou_thr, skip_box_thr=skip_box_thr)
    boxes = boxes*(image_size-1)
    return boxes, scores, labels

def test(test_data_loader, models, device):

    with torch.no_grad():

        for images, image_ids in test_data_loader:
            images = list(image.to(device) for image in images)    
            predictions = make_ensemble_predictions(images, models)
        
            for i, image in enumerate(images):
                test_images.append(image) #Saving image values
                boxes, scores, labels = run_wbf(predictions, image_index=i)
        
                boxes = boxes.astype(np.int32).clip(min=0, max=1023)
                    
                preds = boxes
                preds_sorted_idx = np.argsort(scores)[::-1]
                preds_sorted = preds[preds_sorted_idx]
                boxes = preds
                
                output = {
                    'boxes': boxes,
                    'scores': scores
                }

                outputs.append(output) #Saving outputs and scores
                image_id = image_ids[i]
        
        
        ########### show images results
        fig, axs = plt.subplots(2, 2, figsize=(32, 16))
        axs = axs.ravel()
        for i in range(4):
            #sample = test_images[i].permute(1,2,0).cpu().numpy()
            sample = torchvision.transforms.functional.convert_image_dtype(
                test_images[i].cpu(), torch.uint8).numpy().transpose([1,2,0])
            sample = cv2.cvtColor(sample, cv2.COLOR_BGR2RGB)
            boxes = outputs[i]['boxes']
            scores = outputs[i]['scores']
            boxes = boxes[scores >= detection_threshold].astype(np.int32)
        
            for box in boxes:
                cv2.rectangle(sample,
                              (box[0], box[1]),
                              (box[2], box[3]),
                              (220, 0, 0), 2)
        
            axs[i].set_axis_off()
            axs[i].imshow(sample)