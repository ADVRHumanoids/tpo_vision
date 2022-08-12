import rospy
from cv_bridge import CvBridge, CvBridgeError
import numpy as np

from sensor_msgs.msg import CompressedImage

class RosToOpenCVImageConverter:
    
  ros_image = CompressedImage()
  cv_image = np.zeros((100,100,3), np.uint8)

  def __init__(self, image_sub_topic):

    self.bridge = CvBridge()
    self.image_sub = rospy.Subscriber(image_sub_topic, CompressedImage, self.__callback)
    print("wating for image...")
    rospy.wait_for_message(image_sub_topic, CompressedImage, timeout=None)
    print("... first image arrived")

  def __callback(self, msg):
    self.ros_image = msg;
    try:
      self.cv_image = self.bridge.compressed_imgmsg_to_cv2(msg)
      #cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
    except CvBridgeError as e:
      print(e)


