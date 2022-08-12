#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 16 15:29:18 2022

@author: tori
"""


import torch
from PIL import Image


import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def testNet(model, dataset_test, device):
    # pick one image from the test set
    img, _ = dataset_test[0]
    # put the model in evaluation mode
    model.eval()
    with torch.no_grad():
        prediction = model([img.to(device)])
        
    print(prediction)
    
    original = Image.fromarray(img.mul(255).permute(1, 2, 0).byte().numpy())
    
    predicted = Image.fromarray(prediction[0]['boxes'][0, 0].mul(255).byte().cpu().numpy())
    
    imgplot = plt.imshow(original)
    plt.show()
    
    imgplot = plt.imshow(predicted)
    plt.show()