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
import sys
import math

from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.rpn import AnchorGenerator

sys.path.insert(1, 'detection')
from engine import train_one_epoch, evaluate
import utils
import transforms

from CustomCocoDataset import CustomCocoDataset
from testingNet import *
from PennFudanDataset import *

"""
example taken from https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html
"""
def get_model_mobilenet(num_classes):
    # load a pre-trained model for classification and return
    # only the features
    backbone = torchvision.models.mobilenet_v2(weights="DEFAULT").features
    # FasterRCNN needs to know the number of
    # output channels in a backbone. For mobilenet_v2, it's 1280
    # so we need to add it here
    backbone.out_channels = 1280
    
    # let's make the RPN generate 5 x 3 anchors per spatial
    # location, with 5 different sizes and 3 different aspect
    # ratios. We have a Tuple[Tuple[int]] because each feature
    # map could potentially have different sizes and
    # aspect ratios
    anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),),
                                       aspect_ratios=((0.5, 1.0, 2.0),))
    
    # let's define what are the feature maps that we will
    # use to perform the region of interest cropping, as well as
    # the size of the crop after rescaling.
    # if your backbone returns a Tensor, featmap_names is expected to
    # be [0]. More generally, the backbone should return an
    # OrderedDict[Tensor], and in featmap_names you can choose which
    # feature maps to use.
    roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'],
                                                    output_size=7,
                                                    sampling_ratio=2)
    
    # put the pieces together inside a FasterRCNN model
    model = FasterRCNN(backbone,
                       num_classes=2,
                       rpn_anchor_generator=anchor_generator,
                       box_roi_pool=roi_pooler)

    return model

"""
example taken from https://medium.com/fullstackai/how-to-train-an-object-detector-with-your-own-coco-dataset-in-pytorch-319e7090da5
"""
def get_model_fasterrcnn(num_classes):
    # load an instance segmentation model pre-trained pre-trained on COCO
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model

"""
from https://pytorch.org/vision/0.13/models.html#object-detection-instance-segmentation-and-person-keypoint-detection
"""
def get_model_fasterrcnn(num_classes):
    # load an instance segmentation model pre-trained pre-trained on COCO
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model

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

if __name__ == '__main__':
    
    torch.manual_seed(14)
    
    # select device (whether GPU or CPU)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    # 2 classes; Only target class or background
    num_classes = 2
    
    # path to your own data and coco file
    data_dir = 'data2/images'
    data_dir_annotations = 'data2/coco/annotations/instances_default.json'

    # create own Dataset
    dataset = CustomCocoDataset(root=data_dir,
                              annotation=data_dir_annotations,
                              transforms=get_transform(True)
                              )
    dataset_test = CustomCocoDataset(root=data_dir,
                              annotation=data_dir_annotations,
                              transforms=get_transform(False)
                              )
    
    #dataset = PennFudanDataset('PennFudanPed', get_transform(train=True))
    #dataset_test = PennFudanDataset('PennFudanPed', get_transform(train=False))
    
    # split the dataset in train and test set
    test_percentage = 0.18 # percentage over the total images which is used for testing
    
    indices = torch.randperm(len(dataset)).tolist()
    percent = math.ceil(len(indices) * test_percentage)
    dataset = torch.utils.data.Subset(dataset, indices[:-percent])
    dataset_test = torch.utils.data.Subset(dataset_test, indices[-percent:])
    
    # own DataLoader
    data_loader = torch.utils.data.DataLoader(dataset,
                                              batch_size=1,
                                              shuffle=True,
                                              num_workers=4,
                                              collate_fn=utils.collate_fn)
    
    data_loader_test = torch.utils.data.DataLoader(dataset_test, 
                                                   batch_size=1, 
                                                   shuffle=False, 
                                                   num_workers=4,
                                                   collate_fn=utils.collate_fn)
    

    # DataLoader is iterable over Dataset
    #for imgs, annotations in data_loader:
    #    imgs = list(img.to(device) for img in imgs)
    #    annotations = [{k: v.to(device) for k, v in t.items()} for t in annotations]
    #    print(annotations)
       

    model = get_model_fasterrcnn(num_classes)
    
    # move model to the right device
    model.to(device)
    
    # construct optimizer and scheduler
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=3,
                                                   gamma=0.1)
    
    len_dataloader = len(data_loader)
    
    num_epochs = 5
    
    for epoch in range(num_epochs):
        # train for one epoch, printing every 10 iterations
        train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
        # update the learning rate
        lr_scheduler.step()
        # evaluate on the test dataset
        evaluate(model, data_loader_test, device=device)
        
    
    
    torch.save(model, "model2.pt")