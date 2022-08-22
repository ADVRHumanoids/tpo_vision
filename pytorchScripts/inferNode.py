#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 19 10:48:52 2022

@author: tori
"""
import torch
import torchvision.transforms as T

from drawBoxes import show_image_with_boxes, load_example_image, show_image_with_boxes_opencv, run_example

if __name__ == '__main__':
    
    img, info = load_example_image()   
    
    if torch.cuda.is_available():
        device = torch.device('cuda')
        model = torch.load('model1.pt')
    
    else:
        device = torch.device('cpu')
        model = torch.load('model1.pt', map_location=torch.device('cpu'))
        
    model.eval()
    model.to(device)
    
    # wants a tensor
    transform_chain = T.Compose([T.PILToTensor(),
                                 T.ConvertImageDtype(torch.float)
                                 ])

    with torch.no_grad():

        images = [(transform_chain(img).to(device))]
        out = model(images)[0]
        #out = non_max_suppression(detections, 80, self.confidence_th, self.nms_th)
    
        images[0] = images[0].detach().cpu()
    
    threshold = 0.3
    if (max(out['scores']) > threshold):
    
        #IDK if the best box is always the first one...
        best_index = torch.argmax(out['scores'])
        
        show_image_with_boxes(img, out['boxes'][best_index], out['labels'][best_index])
    else :
        print(f"no detection found under the threshold {threshold}")

    del model

    
    
