#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  8 15:02:25 2022

@author: tori
"""
import torch
import torch.utils.data
import torchvision

import sys
sys.path.insert(1, 'detection')
from engine import train_one_epoch, evaluate
import utils

from CustomCocoDataset import CustomCocoDataset


import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
from ensemble_boxes import weighted_boxes_fusion

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

from torchmetrics.detection.mean_ap import MeanAveragePrecision

detection_threshold = 0.5
results = []

test_images = []

def make_ensemble_predictions(images, models):
    result = []
    for net in models:
        net.eval()
        outputs = net(images)
        result.append(outputs)
    return result

def run_wbf(predictions, image_index, image_width, image_height, iou_thr=0.1, skip_box_thr=0.4, weights=None):
    
    # boxes are in format [xmin, ymin, xmax, ymax], which is ok for weighted_boxes_fusion but 
    # we have to normalize them
    normalizer = [image_width-1, image_height-1, image_width-1, image_height-1]
    boxes = [prediction[image_index]['boxes'].data.cpu().numpy() / normalizer for prediction in predictions]
    
    scores = [prediction[image_index]['scores'].data.cpu().numpy() for prediction in predictions]
    labels = [np.ones(prediction[image_index]['scores'].shape[0]) for prediction in predictions]
    
    boxes, scores, labels = weighted_boxes_fusion(boxes, scores, labels, weights=None, iou_thr=iou_thr, skip_box_thr=skip_box_thr)
    
    # denormalize again the boxes
    boxes = boxes * normalizer
    
    return boxes, scores, labels

def test(test_data_loader, models, device):
    
    metric = MeanAveragePrecision()

    with torch.no_grad():

        for images, image_ids in test_data_loader:
            
            print ("aaaaaaaaaaaa")
            print(image_ids)
            print(" ")
            
            images = list(image.to(device) for image in images)    
            predictions = make_ensemble_predictions(images, models)
            
            
            print ("bbbbbbbbbbb")
            print(predictions)
            print(" ")
            outputs = []
            for i, image in enumerate(images):
                test_images.append(image) #Saving image values
                boxes, scores, labels = run_wbf(predictions, image_index=i, image_width=image.shape[2], image_height=image.shape[1])
        
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
              
            print ("ccccccccccc")
            print(outputs)
            print(" ")       
            print(" ")
        
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



data_dir = 'data/laser_v3/'

data_dir_images = data_dir + 'images'
data_dir_annotations = data_dir + 'annotations/instances_default.json'

def get_transform(train=None):
    custom_transforms = []
    custom_transforms.append(torchvision.transforms.ToTensor())
    if train:
        custom_transforms.append(torchvision.transforms.RandomHorizontalFlip())
        custom_transforms.append(torchvision.transforms.RandomVerticalFlip())
        #custom_transforms.append(torchvision.transforms.RandomPosterize())
       # custom_transforms.append(torchvision.transforms.RandomSolarize())
        custom_transforms.append(torchvision.transforms.RandomAdjustSharpness(2))
        custom_transforms.append(torchvision.transforms.RandomAdjustSharpness(0))
        custom_transforms.append(torchvision.transforms.RandomAutocontrast())
        #custom_transforms.append(torchvision.transforms.RandomEqualize())
    return torchvision.transforms.Compose(custom_transforms)

def get_test_transform():
    return A.Compose([
        # A.Resize(512, 512),
        ToTensorV2(p=1.0)
    ])


dataset = CustomCocoDataset(root=data_dir_images,
                          annotation=data_dir_annotations,
                          transforms=get_transform(True)
                          )
dataset_val = CustomCocoDataset(root=data_dir_images,
                          annotation=data_dir_annotations,
                          transforms=get_transform(False)
                          )
dataset_test = CustomCocoDataset(root=data_dir_images,
                          annotation=data_dir_annotations,
                          transforms=get_transform(False)
                          )

indices = torch.randperm(len(dataset)).tolist()

dataset_test = torch.utils.data.Subset(dataset_test, indices[-4:])

data_loader_test = torch.utils.data.DataLoader(dataset_test, 
                                               batch_size=1, 
                                               shuffle=False, 
                                               num_workers=4,
                                               drop_last=False,
                                                   collate_fn=utils.collate_fn)


###testing setup
models = []

model_path = 'model1.pt'
model = None
if torch.cuda.is_available():
    device = torch.device('cuda')
    print("CUDA available, use GPU")
    model = torch.load(model_path)

else:
    device = torch.device('cpu')
    print("CUDA not available, use CPU") 
    model = torch.load(model_path, map_location=torch.device('cpu'))

model.eval()
model.to(device)

models.append(model)    


test(data_loader_test, models, device)


