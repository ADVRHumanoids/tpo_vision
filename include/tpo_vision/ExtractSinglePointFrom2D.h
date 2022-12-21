/*
 * Copyright 2022 <copyright holder> <email>
 * 
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * 
 *     http://www.apache.org/licenses/LICENSE-2.0
 * 
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef EXTRACTSINGLEPOINTFROM2D_H
#define EXTRACTSINGLEPOINTFROM2D_H

#include <ros/ros.h>
#include <tf2_ros/transform_broadcaster.h>
#include <geometry_msgs/TransformStamped.h>

#include <sensor_msgs/CameraInfo.h>
#include <sensor_msgs/Image.h>
#include <cv_bridge/cv_bridge.h>

#include <pcl_ros/point_cloud.h>

typedef pcl::PointCloud<pcl::PointXYZ> PointCloud;
/**
* This is just a debug-trial classes
**/
class ExtractSinglePointFrom2D { 
    
public:
    ExtractSinglePointFrom2D(ros::NodeHandle* nh);
    
    bool run(const int pixel_x, const int pixel_y);

private:
    ros::NodeHandle* nh;
    
    tf2_ros::TransformBroadcaster tf_broadcaster;
    
    ros::Subscriber cloud_sub;
    void cloudClbk(const PointCloud::ConstPtr& msg);
    PointCloud::Ptr cloud;  
        
    ros::Subscriber camera_info_sub;
    void cameraInfoClbk(const sensor_msgs::CameraInfoConstPtr& msg);
    sensor_msgs::CameraInfo cam_info;
    cv::Mat disp2Depth; //the Q required for reprojectImageTo3D, built from cam param
    
    ros::Subscriber depth_image_sub;
    void depthImageClbk(const sensor_msgs::ImageConstPtr& msg);
    cv_bridge::CvImage cv_bridge_image;
};

#endif //EXTRACTSINGLEPOINTFROM2D_H
