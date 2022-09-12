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
import utils

from CustomCocoDataset import CustomCocoDataset


import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

from torchmetrics.detection.mean_ap import MeanAveragePrecision

detection_threshold = 0.4
test_images_to_print = []
outputs_to_print = []

def test(data_loader_test, model, device):
    
    with torch.no_grad():
        
        model.to(device)
        model.eval()    
        
        metric = MeanAveragePrecision(box_format='xyxy', iou_type='bbox')
        
        for images, targets in data_loader_test:
            
            #images and targets have the dim of the batch size given to the data loader
            #print ("targets: ")
            #print(targets)
            #print(" ")
            
            images = list(image.to(device) for image in images)    
            predictions = model(images)   
            
            #print ("predictions:")
            #print(predictions)
            #print(" ")
            
            for i, pred in enumerate(predictions):

                if pred['scores'].size(0) > 0:
                    
                    best_index = torch.argmax(pred['scores'])
                    #if pred['scores'][best_index] > detection_threshold :
                        
                    #after 2 hours lost, this is the only method I found to create
                    # the tensor as size (1, 4) for boxes and size (1,1) for lab and scores
                    pred['boxes'] = torch.Tensor([np.array(pred['boxes'][best_index].detach().numpy())])
                    pred['labels'] = torch.IntTensor([pred['labels'][best_index]])
                    pred['scores'] = torch.Tensor([pred['scores'][best_index]])
    
                    # else : 
                    #     for key in pred:
                    #         pred['boxes'] = torch.Tensor([[]])
                    #         pred['labels'] = torch.IntTensor([])
                    #         pred['scores'] = torch.Tensor([])
                        
                else : 
                     for key in pred:
                         pred['boxes'] = torch.Tensor([[]])
                         pred['labels'] = torch.IntTensor([])
                         pred['scores'] = torch.Tensor([])
                 
               # print ("predictions after:")
                #print(predictions)
                #print(" ")
                       
                # to print images
                test_images_to_print.append(images[i]) #Saving image values
                
                if (pred['boxes'].size(0) > 0):
                    output = {
                        'boxes': pred['boxes'][0].detach().numpy().astype(np.int32),
                        'scores': pred['scores']
                    }
                else :
                    output = {
                        'boxes': [],
                        'scores': []
                    }

                outputs_to_print.append(output) #Saving outputs and scores
                

            
            metric.update(preds=predictions, target= targets)
            metric_result = metric.compute()
            print(metric_result)
        
        ########### show images results
        fig, axs = plt.subplots(2, 2, figsize=(32, 16))
        axs = axs.ravel()
        nImgToSee = min(4, len(test_images_to_print))
        for i in range(nImgToSee):
            #sample = test_images[i].permute(1,2,0).cpu().numpy()
            sample = torchvision.transforms.functional.convert_image_dtype(
                test_images_to_print[i].cpu(), torch.uint8).numpy().transpose([1,2,0])
            sample = cv2.cvtColor(sample, cv2.COLOR_BGR2RGB)
            box = outputs_to_print[i]['boxes']
            score = outputs_to_print[i]['scores']
            
            if len(box) > 0:
                cv2.rectangle(sample,
                              (box[0], box[1]),
                              (box[2], box[3]),
                              (220, 0, 0), 2)
                
                cv2.putText(sample, str(score[0].detach().numpy()), (round(box[0].item()), round(box[3].item()+20)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)

            axs[i].set_axis_off()
            axs[i].imshow(sample)



data_dir = 'data/laser_v3/'
detach
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
                                               batch_size=2, 
                                               shuffle=False, 
                                               num_workers=4,
                                               drop_last=False,
                                                   collate_fn=utils.collate_fn)


###testing setup

model_path = 'faster_rcnn_v2_e30_b8_t20.pt'
model = None
if torch.cuda.is_available():
    device = torch.device('cuda')
    print("CUDA available, use GPU")
    model = torch.load(model_path)

else:
    device = torch.device('cpu')
    print("CUDA not available, use CPU") 
    model = torch.load(model_path, map_location=torch.device('cpu'))


test(data_loader_test, model, device)


