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

#ifndef LASER3DTRACKING_H
#define LASER3DTRACKING_H

#include <ros/ros.h>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2_ros/transform_listener.h>
#include <geometry_msgs/TransformStamped.h>
#include <ddynamic_reconfigure/ddynamic_reconfigure.h>

#include <pcl_ros/point_cloud.h>
#include <pcl_ros/transforms.h>

#include <tpo_msgs/KeypointImage.h>

#include <Eigen/Dense>

#include <utils/SecondOrderFilter.h>

typedef pcl::PointCloud<pcl::PointXYZ> PointCloud;

class Laser3DTracking {
    
    
public:
    Laser3DTracking(ros::NodeHandle* nh, const double& period);
    
    bool isReady();
    
    int run();
    
private:
    ros::NodeHandle* nh;
    double period;
    
    std::string camera_frame;
    std::string ref_frame ;
    std::string laser_spot_frame;
    
    double detection_confidence_threshold;
    double cloud_detection_max_sec_diff;
    ros::Subscriber keypoint_sub;
    void keypointSubClbk(const tpo_msgs::KeypointImageConstPtr& msg);
    tpo_msgs::KeypointImage keypoint_image;
    
    tf2_ros::TransformBroadcaster tf_broadcaster;
    std::vector<geometry_msgs::TransformStamped> ref_T_spot; //one for raw, other for filtered
    
    tf2_ros::Buffer tf_buffer;
    std::unique_ptr<tf2_ros::TransformListener> tf_listener;

    ros::Subscriber cloud_sub;
    void cloudClbk(const PointCloud::ConstPtr& msg);
    PointCloud::Ptr cloud;  

    /***************************************************** */

    bool sendTransformFrom2D();
    bool updateTransform();
    
    /***************  FILTER    **********/
    tpo::utils::FilterWrap<Eigen::Vector3d>::Ptr _laser_pos_filter;
    double _filter_damping, _filter_bw;
    
    std::unique_ptr<ddynamic_reconfigure::DDynamicReconfigure> _ddr_server;
    void ddr_callback_filter_damping(double new_value);
    void ddr_callback_filter_bw(double new_value);
    
};

#endif // LASER3DTRACKING_H
