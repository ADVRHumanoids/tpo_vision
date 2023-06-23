#!/usr/bin/env python3

"""
This node check if the "tracking_frame" is near (linearly) one of the "goal_frames" in a range specified by "threshold" (x y z) argument, and publish a ros message with the name of the closer frame. Be sure to put a threshold smaller that the distance of the goal_frames.
"""

import rospy, tf2_ros
from std_msgs.msg import String
from geometry_msgs.msg import TransformStamped
from visualization_msgs.msg import Marker, MarkerArray

if __name__ == '__main__':
    
    rospy.init_node('check_closer_frame')
    
    tracking_frame = rospy.get_param('~tracking_frame', 'laser')
    goal_frames = rospy.get_param('~goal_frames')
    closeness_topic_name = rospy.get_param('~closeness_topic_name', 'closer_frame')
    marker_topic_name = rospy.get_param('~marker_topic_name', 'keyboard_markers')
    threshold_x = rospy.get_param('~threshold_x', 0.01)
    threshold_y = rospy.get_param('~threshold_y', 0.01)
    threshold_z = rospy.get_param('~threshold_z', 0.005)
    rate_value = rospy.get_param('~rate', 100)
    
    tf_buffer = tf2_ros.Buffer()
    tf_listener = tf2_ros.TransformListener(tf_buffer)
    closeness_pub = rospy.Publisher(closeness_topic_name, String, queue_size=10)
    
    ##Marker
    
    
    markers = MarkerArray();
    marker_pub = rospy.Publisher(marker_topic_name, MarkerArray, queue_size=10)
    
    frame_to_marker_id = {}

    
    for i in range(len(goal_frames)):
        marker = Marker()
        marker.header.stamp = rospy.get_rostime()
        marker.ns = "keyboard"
        marker.type = Marker.MESH_RESOURCE
        marker.action = Marker.ADD #add
        marker.mesh_resource = "package://iit_gazebo_worlds_pkg/simpleKeyboard/meshes/arrow.dae"
        marker.pose.position.x = 0
        marker.pose.position.y = 0
        marker.pose.position.z = 0.01
        marker.scale.x = 0.06
        marker.scale.y = 0.08
        marker.scale.z = 0.05
        marker.color.a = 0.4

        marker.header.frame_id = goal_frames[i]
        marker.id = i
        frame_to_marker_id[goal_frames[i]] = i
        if goal_frames[i] == "x_pos_link" :
            marker.pose.orientation.x = 0.0
            marker.pose.orientation.y = 0.0
            marker.pose.orientation.z = 0.0
            marker.pose.orientation.w = 1.0
            marker.color.r = 1
            marker.color.g = 0
            marker.color.b = 0
        elif goal_frames[i] == "x_neg_link" :
            marker.pose.orientation.x = 0.0
            marker.pose.orientation.y = 0.0
            marker.pose.orientation.z = 1.0
            marker.pose.orientation.w = 0.0 
            marker.color.r = 1
            marker.color.g = 0
            marker.color.b = 0
        elif goal_frames[i] == "y_pos_link" :
            marker.pose.orientation.x = 0.0
            marker.pose.orientation.y = 0.0
            marker.pose.orientation.z = 0.7071068
            marker.pose.orientation.w = 0.7071068
            marker.color.r = 0
            marker.color.g = 1
            marker.color.b = 0
        elif goal_frames[i] == "y_neg_link" :
            marker.pose.orientation.x = 0.0
            marker.pose.orientation.y = 0.0
            marker.pose.orientation.z = -0.7071068
            marker.pose.orientation.w = 0.7071068
            marker.color.r = 0
            marker.color.g = 1
            marker.color.b = 0
        elif goal_frames[i] == "z_pos_link" :
            marker.pose.orientation.x = 0.0
            marker.pose.orientation.y = 0.0
            marker.pose.orientation.z = -0.3826834
            marker.pose.orientation.w = 0.9238795
            marker.color.r = 0
            marker.color.g = 0
            marker.color.b = 1    
        elif goal_frames[i] == "z_neg_link" :
            marker.pose.orientation.x = 0.0
            marker.pose.orientation.y = 0.0
            marker.pose.orientation.z = 0.9238795
            marker.pose.orientation.w = 0.3826834
            marker.color.r = 0
            marker.color.g = 0
            marker.color.b = 1  
        
        
        markers.markers.append(marker)
        

    marker_pub.publish(markers)
        
    
    tf_msgs = {}
    for i in range(len(goal_frames)):
        tf_msgs[goal_frames[i]] = TransformStamped()
        
    closer_frame = "none";

    rate = rospy.Rate(rate_value)
    while not rospy.is_shutdown():
        
        closer_frame = "none"
        for marker in markers.markers:
            marker.color.a = 0.5
        
        try:
            for i in range(len(goal_frames)):
                tf_msgs[goal_frames[i]] = tf_buffer.lookup_transform(tracking_frame, goal_frames[i], rospy.Time())
                if (abs(tf_msgs[goal_frames[i]].transform.translation.x) < threshold_x and
                    abs(tf_msgs[goal_frames[i]].transform.translation.y) < threshold_y and
                    abs(tf_msgs[goal_frames[i]].transform.translation.z) < threshold_z ) :
                
                    #We take the first one, assuming that can be near to only one frame.
                    closer_frame = goal_frames[i]
                    markers.markers[frame_to_marker_id[goal_frames[i]]].color.a = 1
                    break;
                    
                
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as ex:
            rospy.logwarn_throttle(1, "%s", ex)
        
        closeness_pub.publish(closer_frame)
        marker_pub.publish(markers)
        rate.sleep()
