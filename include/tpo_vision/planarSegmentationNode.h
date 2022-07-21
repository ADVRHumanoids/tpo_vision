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

#ifndef PLANARSEGMENTATIONNODE_H
#define PLANARSEGMENTATIONNODE_H

#include <ros/ros.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/transform_broadcaster.h>
#include <geometry_msgs/TransformStamped.h>
#include <sensor_msgs/CameraInfo.h>

#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>

#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>

#include <pcl_ros/point_cloud.h>
#include <pcl_ros/transforms.h>

#include <pcl/ModelCoefficients.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/common/common.h>
#include <pcl/point_types.h>
#include <pcl/kdtree/kdtree.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/features/moment_of_inertia_estimation.h>
#include <pcl/features/fpfh_omp.h>

#include <pcl/visualization/cloud_viewer.h> //to see the cloud, just for debug


typedef pcl::PointCloud<pcl::PointXYZ> PointCloud;

class PlanarSegmentation {
    
    
public:
    PlanarSegmentation(ros::NodeHandle* nh);
    
    void getTransforms();
    
    int run();
    
private:
    ros::NodeHandle* nh;
    
    std::string camera_frame = "D435_head_camera_color_optical_frame";
    std::string ref_frame = "pelvis" ;
    
    tf2_ros::Buffer tf_buffer;
    std::unique_ptr<tf2_ros::TransformListener> tf_listener;
    tf2_ros::TransformBroadcaster tf_broadcaster;
    geometry_msgs::TransformStamped cam_T_pelvis;
    geometry_msgs::TransformStamped pelvis_T_wheel;
    double pelvis_high = 0;

    ros::Subscriber cloud_sub;
    void cloudClbk(const PointCloud::ConstPtr& msg);
    ros::Publisher cloud_plane_pub, cloud_objects_pub;
    std::vector<ros::Publisher> cloud_tmp_pub;
    
    //bounding box marker pub
    ros::Publisher marker_pub;
    visualization_msgs::MarkerArray markerArrayMsg;
    void addBoundingBoxMarker(unsigned int id, std::string frame_id, double x, double y, double z);

    //phases methods
    bool filterOnZaxis();
    bool extractPlaneAndObjects(bool publishPlane, bool publishObjectsOnTable);
    bool clusterExtraction(bool publishSingleObjCloud);
    std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> cloud_clusters;
    unsigned int n_clusters = 0;
    unsigned int max_clusters = 10;
    geometry_msgs::Transform momentOfInertia(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, 
                                             double* x_size = nullptr, double* y_size = nullptr, double* z_size = nullptr);
    
    PointCloud::Ptr cloud;  
    PointCloud::Ptr cloud_plane;  
    PointCloud::Ptr cloud_objects;  
    
    pcl::visualization::PCLVisualizer::Ptr viewer;
    
    /***************************************************** */
    ros::Subscriber camera_info_sub;
    void cameraInfoClbk(const sensor_msgs::CameraInfoConstPtr& msg);
    sensor_msgs::CameraInfo cam_info;
    cv::Mat disp2Depth; //the Q required for reprojectImageTo3D, built from cam param
    
    ros::Subscriber depth_image_sub;
    void depthImageClbk(const sensor_msgs::ImageConstPtr& msg);
    cv_bridge::CvImage cv_bridge_image;
    public: bool extractSinglePointFrom2D(int pixel_x, int pixel_y);
    
    ros::Publisher tmp_pub;

};

#endif // PLANARSEGMENTATIONNODE_H
