#!/usr/bin/env python

import rospy
import cv2 as cv
import os
import sys
import numpy as np
import numpy.ma as ma
import time
import message_filters
import math

from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError #To convert from ROS image format to opencv image format and viceversa.
from datetime import datetime
from geometry_msgs.msg import Twist
from std_msgs.msg import String

bridge = CvBridge()

rospy.init_node('detect_object', anonymous=True)
	
print ('Current OPENCV version: ', cv.__version__)

class detect_object:
	def __init__(self):
		self.bridge = CvBridge()
		image_sub = message_filters.Subscriber("/D435/D435_head_camera/color/image_raw", Image)  #for the wirst: "/D435i/D435i_camera/color/image_raw"
		depth_sub = message_filters.Subscriber("/D435/D435_head_camera/aligned_depth_to_color/image_raw", Image) #for the wirst: "/D435i/D435i_camera/aligned_depth_to_color/image_raw"
		self.ts = message_filters.ApproximateTimeSynchronizer([image_sub, depth_sub], 10, 0.5)
		self.ts.registerCallback(self.callback)

	def callback(self, img_msg, depth_msg):
		try:
			imageRGB = self.bridge.imgmsg_to_cv2(img_msg, "bgr8")
			cv.imshow('Color_IMG', imageRGB)
			#cv.waitKey(3)
			depth_image = self.bridge.imgmsg_to_cv2(depth_msg, "passthrough")
			depth_array = np.array(depth_image, dtype=np.float32)
			cv.normalize(depth_array, depth_array, 0, 1, cv.NORM_MINMAX)
			#cv.imshow('Depth_IMG', depth_array)
			#cv.waitKey(3)
			
			height= imageRGB.shape[0]
			width= imageRGB.shape[1]
			print ('Image size:', width, 'x', height, 'pixels')

			#cv.waitKey(0)
			
			
		except CvBridgeError as e:
			print (e)

def main(args):
	fp = detect_object()

	try:
		rospy.spin()
	except KeyboardInterrupt:
		print ("Shutting down")
	cv.destroyAllWindows()

if __name__ == '__main__':
	main(sys.argv)
