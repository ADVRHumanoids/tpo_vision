#!/usr/bin/env python
import rospy
import cv2 as cv
from sensor_msgs.msg import CameraInfo 

from image_ros_to_opencv import image_converter
from laser_tracker import LaserTracker

class getCameraInfo:
    
    cam_info = {}
    
    def __init__(self, image_info_topic):
        self.sub = rospy.Subscriber(image_info_topic, CameraInfo, self.callback)
        print("wating for camerainfo...")
        rospy.wait_for_message(image_info_topic, CameraInfo, timeout=None)
        print("... camerainfo arrived")

    def callback(self, msg):
        self.cam_info["width"] = msg.width
        self.cam_info["height"] = msg.height
        self.sub.unregister()


if __name__ == '__main__':
    
    rospy.init_node('laser_spot_detection', anonymous=False)
    
    if True:
        getCameraInfo("/D435_head_camera/color/camera_info")
        cam_info = getCameraInfo.cam_info
        
    else:
        cam_info = {}
        cam_info["width"] = 640
        cam_info["height"] = 480
        
    cam_info["color"] = ""
    
    ic = image_converter("/D435_head_camera/color/image_raw/compressed")
    
    tracker = LaserTracker(cam_width=cam_info["width"], cam_height=cam_info["height"], cam_color=cam_info["color"],
                           display_thresholds=True)
    tracker.setup_windows()
        
    #in python spinOnce does not exist, it is not necessary
    rate = rospy.Rate(100) # ROS Rate
    while not rospy.is_shutdown():

        img = ic.cv_image
        #cv.imshow("Image window", img)
        #cv.waitKey(3)
        hsv_image = tracker.detect(img)
        tracker.display(hsv_image, img)
        cv.waitKey(3)
        
        rate.sleep()

    cv.destroyAllWindows()
