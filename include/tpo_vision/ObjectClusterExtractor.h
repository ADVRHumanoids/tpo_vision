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

#ifndef OBJECT_CLUSTER_EXTRACTOR_H
#define OBJECT_CLUSTER_EXTRACTOR_H

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

#include <tpo_msgs/ClusterObject.h>

typedef pcl::PointXYZ Point;
typedef pcl::PointCloud<Point> PointCloud;

class ObjectCluster {
    
public:
    enum class Type {
        
        Surface, //eg table 
        Object,  //object graspable with a single end effector
        Container, // bigger object where small object can be put above/inside, maybe graspable with two arms?
        None,
    };
    
    ObjectCluster();
    
    Type _type;
    
    PointCloud::Ptr _cloud;
    
    bool momentOfInertiaOBB();
    bool momentOfInertiaAABB();

    void fillMarker(unsigned id);
    void fillTransform(unsigned id);
    
    bool fillTransformGoal();
    
    visualization_msgs::Marker _marker;
    geometry_msgs::TransformStamped _ref_T_cloud;
    geometry_msgs::TransformStamped _ref_T_goal;
    
    bool findPoint(Point searchPoint, float radius = 0.03);
    
    bool categorizeCluster();
    
    pcl::MomentOfInertiaEstimation <Point> _feature_extractor;
    
    //This will be different according to AABB or OBB method
    Eigen::Vector3f _dimensions;
    Eigen::Vector3f _position;
    Eigen::Quaternionf _rotation;

    
private:
    
    pcl::KdTreeFLANN<Point> _kdtree;
    std::vector<int> _point_idx_found; //to store index of surrounding points 
    std::vector<float> _point_radius_squared_distance; // to store distance to surrounding points
    bool _selected_cluster;


};



class ObjectClusterExtractor {
    
    
public:
    ObjectClusterExtractor(ros::NodeHandle* nh);
    
    void getTransforms();
    
    int run();
    
private:
    ros::NodeHandle* nh;
    
//     std::string camera_frame;
    std::string ref_frame;
    std::string input_cloud_topic;
    std::string selecting_frame;
    
    tf2_ros::Buffer tf_buffer;
    std::unique_ptr<tf2_ros::TransformListener> tf_listener;
    geometry_msgs::TransformStamped cam_T_pelvis;
    geometry_msgs::TransformStamped pelvis_T_wheel;
    geometry_msgs::TransformStamped refcloud_T_selecting_frame;
    double pelvis_high = 0;

    ros::Subscriber cloud_sub;
    void cloudClbk(const PointCloud::ConstPtr& msg);
    ros::Publisher cloud_plane_pub, cloud_objects_pub;
    std::vector<ros::Publisher> cloud_tmp_pub;
    
    //bounding box marker pub
    ros::Publisher marker_pub;
    visualization_msgs::MarkerArray markerArrayMsg;
    tf2_ros::TransformBroadcaster tf_broadcaster;
    std::vector<geometry_msgs::TransformStamped> transforms;
    
    ros::ServiceServer selected_object_srv;
    bool selectedObjectClbk(tpo_msgs::ClusterObject::Request &req, tpo_msgs::ClusterObject::Response &res);

    //phases methods
    bool filter_on_z_axis, extract_plane_and_objects;
    bool filterOnZaxis();
    bool publishPlane, publishObjectsOnTable, publishSingleObjCloud;
    bool extractPlaneAndObjects();
    bool clusterExtraction();
    std::vector<ObjectCluster> object_clusters;
    unsigned int n_clusters = 0;
    int max_clusters;
    bool publishSingleObjTF;
    bool publishSingleObjBoundingBox;
    
    PointCloud::Ptr cloud;  
    PointCloud::Ptr cloud_plane;  
    PointCloud::Ptr cloud_objects;  
        
    //ros::Publisher tmp_pub;
    
    int selected_cluster = -1;

};



#endif // OBJECT_CLUSTER_EXTRACTOR_H
