#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
From https://learnopencv.com/object-tracking-using-opencv-cpp-python/  (detection type explained)
and 
https://pyimagesearch.com/2018/07/30/opencv-object-tracking/
"""
import sys
import numpy as np

import cv2
from cv_bridge import CvBridge, CvBridgeError

# ROS imports
import rospy
import rospkg
from sensor_msgs.msg import CompressedImage as ROSCompressedImage
from sensor_msgs.msg import Image as ROSImage
from sensor_msgs.msg import CameraInfo

(major_ver, minor_ver,subminor_ver) = cv2.__version__.split(".")[:3]
print(f'Using opecv {major_ver}.{minor_ver}.{subminor_ver}')

class OpenCVTracker:
    
    ros_image_input = ros_image_output = None
    cv_image_input = cv_image_output = np.zeros((100,100,3), np.uint8)
    tensor_images = []
    new_image = False
    
    def __image_clbk(self, msg):
        
        self.ros_image_input = msg;
        try:
            if self.camera_image_transport == "compressed":

                self.cv_image_input = self.bridge.compressed_imgmsg_to_cv2(msg, desired_encoding="bgr8")
            else :
                self.cv_image_input = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            
        except CvBridgeError as e:
            rospy.logerror(e)
            
        self.new_image = True


    def run(self):
        
        ##################### ROS Stuff
        camera_image_topic = rospy.get_param('camera_image_topic', '/D435_head_camera/color/image_raw/')
        self.camera_image_transport = rospy.get_param('transport', 'compressed')
        ros_image_input_topic = camera_image_topic + self.camera_image_transport
        
        self.bridge = CvBridge()
        if self.camera_image_transport == "compressed":
            self.image_sub = rospy.Subscriber(ros_image_input_topic, ROSCompressedImage,
                                              self.__image_clbk, queue_size = 1)
            rospy.loginfo("waiting for camera message...")
            rospy.wait_for_message(ros_image_input_topic, ROSCompressedImage, timeout=None)
            rospy.loginfo("... camera message arrived")
            self.ros_image_input = ROSCompressedImage
            
        else:
            self.image_sub = rospy.Subscriber(ros_image_input_topic, ROSImage,
                                              self.__image_clbk, queue_size = 10)
            rospy.loginfo("waiting for camera message...")
            rospy.wait_for_message(ros_image_input_topic, ROSImage, timeout=None)
            rospy.loginfo("... camera message arrived")
            self.ros_image_input = ROSImage
        ##################### 
        

        # Set up tracker.
        # Instead of MIL, you can also use
        # Boosting : bad, it only manage sometimes to track for some instant after the beginning
        #           learnopecv says no reason to use it since better based on similar principle exist(MIL, KCF)
        # MIL: not so bad. also recovery after lase switch off
        #      learnopecv says that KCF is better
        # KCF: very bad do not trak nothing, at least no false positive
        # TLD: very bad, also low fps (10, previous was like 300fps)
        # MEDIANFLOW: bad, track for some instant (sometime) then the bounding box grow bigger and bigger
        # GOTURN: 
        #        it uses cnn
        # MOSSE: bad, no track
        # CSRT: not so bad, but no recovery after laser switch off or lost, less fps than MIL it seems

        tracker_types = ['BOOSTING', 'MIL','KCF', 'TLD', 'MEDIANFLOW', 'GOTURN', 'MOSSE', 'CSRT']
        tracker_type = tracker_types[5]
        print(f'tracking type: {tracker_type}')

        if int(major_ver) == 3 and int(minor_ver) < 3:
            tracker = cv2.Tracker_create(args["tracker"].upper())
        else:
            if tracker_type == 'BOOSTING':
                tracker = cv2.TrackerBoosting_create()
            if tracker_type == 'MIL':
                tracker = cv2.TrackerMIL_create()
            if tracker_type == 'KCF':
                tracker = cv2.TrackerKCF_create()
            if tracker_type == 'TLD':
                tracker = cv2.TrackerTLD_create()
            if tracker_type == 'MEDIANFLOW':
                tracker = cv2.TrackerMedianFlow_create()
            if tracker_type == 'GOTURN':
                tracker = cv2.TrackerGOTURN_create()
            if tracker_type == 'MOSSE':
                tracker = cv2.TrackerMOSSE_create()
            if tracker_type == "CSRT":
                tracker = cv2.TrackerCSRT_create()


        # Read first frame.
        frame = self.cv_image_input
        
        # Define an initial bounding box
        bbox = (287, 23, 86, 320)
        bbox = cv2.selectROI(frame, True)

        # Initialize tracker with first frame and bounding box
        ok = tracker.init(frame, bbox)

        while True:
            # Read a new frame
            frame = self.cv_image_input
            
            # Start timer
            timer = cv2.getTickCount()

            # Update tracker
            ok, bbox = tracker.update(frame)

            # Calculate Frames per second (FPS)
            fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer);

            # Draw bounding box
            if ok:
                # Tracking success
                p1 = (int(bbox[0]), int(bbox[1]))
                p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
                cv2.rectangle(frame, p1, p2, (255,0,0), 2, 1)
            else :
                # Tracking failure
                cv2.putText(frame, "Tracking failure detected", (100,80), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2)

            # Display tracker type on frame
            cv2.putText(frame, tracker_type + " Tracker", (100,20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50),2);
        
            # Display FPS on frame
            cv2.putText(frame, "FPS : " + str(int(fps)), (100,50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50), 2);

            # Display result
            cv2.imshow("Tracking", frame)

            # Exit if ESC pressed
            k = cv2.waitKey(1) & 0xff
            if k == 27 : break


if __name__=="__main__":
    # Initialize node
    rospy.init_node("OpenCVTracker")

    # Define detector object
    tracker = OpenCVTracker()
    tracker.run()

    
    #rate = rospy.Rate(1) # ROS Rate
    
    
    #while not rospy.is_shutdown():
        #tic = rospy.Time().now()
        #dm.infer()
        #toc = rospy.Time().now()
        #rospy.logwarn ('Inference time: %s s', (toc-tic).to_sec())
        #rate.sleep()
    
