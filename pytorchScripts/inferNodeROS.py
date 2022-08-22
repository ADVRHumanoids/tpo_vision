#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 19 10:48:52 2022

@author: tori
with spunti taken from https://github.com/vvasilo/yolov3_pytorch_ros/blob/master/src/yolov3_pytorch_ros/detector.py
"""
#import sys
import os
import numpy as np

# Pytorch stuff
import torch
import torchvision.transforms
#from PIL import Image as PILImage

#Opencv stuff
import cv2
from cv_bridge import CvBridge, CvBridgeError

# ROS imports
import rospy
import rospkg
from sensor_msgs.msg import CompressedImage as ROSCompressedImage
from sensor_msgs.msg import Image as ROSImage
from sensor_msgs.msg import CameraInfo

from tpo_msgs.msg import KeypointImage

#TEST TRIALS; remove this
#from drawBoxes import show_image_with_boxes, load_example_image


class getCameraInfo:
    
    cam_info = {}
    
    def __init__(self, image_info_topic):
        self.sub = rospy.Subscriber(image_info_topic, CameraInfo, self.__callback)
        print("wating for camerainfo...")
        rospy.wait_for_message(image_info_topic, CameraInfo, timeout=None)
        print("... camerainfo arrived")

    def __callback(self, msg):
        self.cam_info["width"] = msg.width
        self.cam_info["height"] = msg.height
        self.sub.unregister()


class DetectorManager():
    
    ros_image_input = ros_image_output = None
    cv_image_input = cv_image_output = np.zeros((100,100,3), np.uint8)
    tensor_images = []
    
    def __init__(self):
        
        model_path = os.path.join(rospkg.RosPack().get_path('tpo_vision'), "pytorchScripts", "model1.pt")
        
        ############ PYTHORCH STUFF
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            rospy.loginfo("CUDA available, use GPU")
            self.model = torch.load(model_path)

        else:
            self.device = torch.device('cpu')
            rospy.loginfo("CUDA not available, use CPU") 
            self.model = torch.load(model_path, map_location=torch.device('cpu'))
        
        self.model.eval()
        self.model.to(self.device)
        
        # wants a tensor
        self.transform_chain = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                    torchvision.transforms.ConvertImageDtype(torch.float)
                                    ])
        self.threshold = 0.3
        
        ############ ROS STUFF
        
        ### Params
        camera_image_topic = rospy.get_param('camera_image_topic', '/D435_head_camera/color/image_raw/')
        self.camera_image_transport = rospy.get_param('transport', 'compressed')
        ros_image_input_topic = camera_image_topic + self.camera_image_transport
        
        pub_out_keypoint_topic = rospy.get_param('pub_out_keypoint_topic', "detection_output_keypoint")
        self.pub_out_images = rospy.get_param('pub_out_images', True)
        pub_out_images_topic = rospy.get_param('pub_out_images_topic', "detection_output_img/compressed")
        
        #camera_info_topic = rospy.get_param('camera_info_topic', '/D435_head_camera/color/camera_info')
        #getCameraInfo(camera_info_topic)
        #self.cam_info = getCameraInfo.cam_info
        
        self.bridge = CvBridge()
        if self.camera_image_transport == "compressed":
            self.image_sub = rospy.Subscriber(ros_image_input_topic, ROSCompressedImage,
                                              self.__image_clbk, queue_size = 1)
            print("wating for camera message...")
            rospy.wait_for_message(ros_image_input_topic, ROSCompressedImage, timeout=None)
            print("... camera message arrived")
            self.ros_image_input = ROSCompressedImage
            
        else:
            self.image_sub = rospy.Subscriber(ros_image_input_topic, ROSImage,
                                              self.__image_clbk, queue_size = 10)
            print("wating for camera message...")
            rospy.wait_for_message(ros_image_input_topic, ROSImage, timeout=None)
            print("... camera message arrived")
            self.ros_image_input = ROSImage
            
        self.keypoint_pub = rospy.Publisher(pub_out_keypoint_topic, KeypointImage, queue_size=10)
            
        if self.pub_out_images:
            self.image_pub = rospy.Publisher(pub_out_images_topic, ROSCompressedImage, queue_size=10)


    def __image_clbk(self, msg):
        
        self.ros_image_input = msg;
        try:
            if self.camera_image_transport == "compressed":

                self.cv_image_input = self.bridge.compressed_imgmsg_to_cv2(msg, desired_encoding="rgb8")
            else :
                self.cv_image_input = self.bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")
            
        except CvBridgeError as e:
            print(e)
    
    
    def __process_image(self):
        
        #pil_image_input = PILImage.fromarray(self.cv_image_input) #img as opencv
        #pil_image_input.show()
        
        self.tensor_images = [(self.transform_chain(self.cv_image_input).to(self.device))]        

        #beh = torchvision.transforms.functional.to_pil_image(self.tensor_images[0], "RGB")
       # beh.show()

       
    def infer(self):
        
        self.__process_image()
        
        with torch.no_grad():

            out = self.model(self.tensor_images)[0]
            #out = non_max_suppression(out, 80, self.confidence_th, self.nms_th)
    
            #images[0] = images[0].detach().cpu()
        

        if (max(out['scores']) > self.threshold):
        
            #IDK if the best box is always the first one, so lets the argmax
            best_index = torch.argmax(out['scores'])
            
            #show_image_with_boxes(img, out['boxes'][best_index], out['labels'][best_index])
            
            self.__pubKeypoint(out['boxes'][best_index], out['scores'][best_index], out['labels'][best_index])
             
            if self.pub_out_images:
                self.__pubImageWithRectangle(out['boxes'][best_index], out['labels'][best_index])
            
            return True
        else :
            print(f"no detection found under the threshold {self.threshold}")
            return False

    def __pubImageWithRectangle(self, box, label=None):
        
        #first convert back to unit8
        self.cv_image_output = torchvision.transforms.functional.convert_image_dtype(
            self.tensor_images[0].cpu(), torch.uint8).numpy().transpose([1,2,0])
        self.cv_image_output = cv2.cvtColor(self.cv_image_output, cv2.COLOR_BGR2RGB)
        
        cv2.rectangle(self.cv_image_output, 
                      (round(box[0].item()), round(box[1].item())),
                      (round(box[2].item()), round(box[3].item())),
                      (255,0,0), 2)
        
        if label:
            cv2.putText(self.cv_image_output, str(label.item()), (round(box[0].item()), round(box[3].item()+10)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)

        
        #cv2.imshow("test_boxes", self.cv_image_output)
        #cv2.waitKey()
        
        self.ros_image_output = self.bridge.cv2_to_compressed_imgmsg(self.cv_image_output)
        self.ros_image_output.header.seq = self.ros_image_input.header.seq
        self.ros_image_output.header.frame_id = self.ros_image_input.header.frame_id
        self.ros_image_output.header.stamp = rospy.Time.now()
        
        self.image_pub.publish(self.ros_image_output)

            
    def __pubKeypoint(self, box, score, label):
        
        msg = KeypointImage()
        msg.header.frame_id = self.ros_image_input.frame_id
        msg.header.seq = self.ros_image_input.seq
        msg.header.stamp = rospy.Time.now()
        
        #box from model has format: [x_0, y_0, x_1, y_1]
        msg.x_pixel = box[0] + (box[2] - box[0])/2
        msg.y_pixel = box[1] + (box[3] - box[1])/2
        msg.label = label
        msg.confidence = score
        
        self.keypoint_pub.publish(msg)
        

if __name__=="__main__":
    # Initialize node
    rospy.init_node("laserSpotDetectionDL")

    # Define detector object
    dm = DetectorManager()

    
    dm.infer()
    
    #rate = rospy.Rate(10) # ROS Rate
   # while not rospy.is_shutdown():
     #   dm.infer()

     #   rate.sleep()
    

    
