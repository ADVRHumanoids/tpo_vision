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

#include <tpo_vision/ObjectClusterExtractor.h>

ObjectCluster::ObjectCluster() {
    
    _marker.ns = "object_cluster_bboxes";
    _marker.type = visualization_msgs::Marker::CUBE;
    _marker.action = visualization_msgs::Marker::ADD;
    _marker.color.a = 0.3; // Don't forget to set the alpha!
    _marker.color.r = 0.0;
    _marker.color.g = 1.0;
    _marker.color.b = 0.0;
    _marker.pose.orientation.x = 0.0;
    _marker.pose.orientation.y = 0.0;
    _marker.pose.orientation.z = 0.0;
    _marker.pose.orientation.w = 1.0;
    
    _kdtree.setSortedResults (true);
    _selected_cluster = false;
    
    _type = ObjectCluster::Type::None;

}

bool ObjectCluster::momentOfInertiaOBB() {
    
    _feature_extractor.setInputCloud (_cloud);
    _feature_extractor.compute ();

    bool ret_value = true;

    pcl::PointXYZ _min_point_OBB;
    pcl::PointXYZ _max_point_OBB;
    pcl::PointXYZ _position_OBB;    
    Eigen::Matrix3f _rotational_matrix_OBB;
    
    ret_value = (ret_value && _feature_extractor.getOBB (_min_point_OBB, _max_point_OBB, _position_OBB, _rotational_matrix_OBB));
    ret_value = (ret_value && _feature_extractor.getMassCenter (_position));
        
    _rotation = Eigen::Quaternionf (_rotational_matrix_OBB);
    _rotation.normalize();
    
    _dimensions << 
        std::abs(_max_point_OBB.x - _min_point_OBB.x),
        std::abs(_max_point_OBB.y - _min_point_OBB.y),
        std::abs(_max_point_OBB.z - _min_point_OBB.z);
        
        
    return ret_value;
}

bool ObjectCluster::momentOfInertiaAABB() {
    
    _feature_extractor.setInputCloud (_cloud);
    _feature_extractor.compute ();

    pcl::PointXYZ _min_point_AABB;
    pcl::PointXYZ _max_point_AABB;
    bool ret_value = true;
    ret_value = (ret_value && _feature_extractor.getAABB (_min_point_AABB, _max_point_AABB));
    
    _dimensions <<
        std::abs(_max_point_AABB.x - _min_point_AABB.x),
        std::abs(_max_point_AABB.y - _min_point_AABB.y),
        std::abs(_max_point_AABB.z - _min_point_AABB.z);

    _rotation.setIdentity();
    
    _position <<
        0.5 * (_max_point_AABB.x + _min_point_AABB.x),
        0.5 * (_max_point_AABB.y + _min_point_AABB.y),
        0.5 * (_max_point_AABB.z + _min_point_AABB.z);
    
    return ret_value;
}

void ObjectCluster::fillMarker(unsigned id) {
    
    _marker.id = id;
    _marker.header.frame_id = _cloud->header.frame_id;
    _marker.header.stamp = ros::Time();
    _marker.pose.position.x = _position (0); 
    _marker.pose.position.y = _position (1); 
    _marker.pose.position.z = _position (2); 
    _marker.pose.orientation.x = _rotation.x();
    _marker.pose.orientation.y = _rotation.y();
    _marker.pose.orientation.z = _rotation.z();
    _marker.pose.orientation.w = _rotation.w();
    _marker.scale.x = _dimensions(0);
    _marker.scale.y = _dimensions(1);
    _marker.scale.z = _dimensions(2);
    
}

void ObjectCluster::fillTransform(unsigned id) {
    
    _ref_T_cloud.header.stamp = ros::Time::now();
    _ref_T_cloud.header.frame_id = _cloud->header.frame_id;
    _ref_T_cloud.child_frame_id = "box_cloud_" + std::to_string(id);

    _ref_T_cloud.transform.translation.x = _position (0); 
    _ref_T_cloud.transform.translation.y = _position (1);
    _ref_T_cloud.transform.translation.z = _position (2);
    
    _ref_T_cloud.transform.rotation.x = _rotation.x();
    _ref_T_cloud.transform.rotation.y = _rotation.y();
    _ref_T_cloud.transform.rotation.z = _rotation.z();
    _ref_T_cloud.transform.rotation.w = _rotation.w();
}


bool ObjectCluster::categorizeCluster() {
    
    //Simple categories based on size
    if (_dimensions(0) <= 0.2 && _dimensions(1) <= 0.08 && _dimensions(2) <= 0.5) {
        
        _type = Type::Object;
        _marker.color.r = 0.5;
        _marker.color.g = 1.0;
        _marker.color.b = 1.0;
    
    } else if ( _dimensions(0) > 0.2 && _dimensions(0) < 1 &&
                _dimensions(1) > 0.1 && _dimensions(1) < 1 &&
                _dimensions(2) > 0.05 && _dimensions(2) < 1 )
    {
        
        _type = Type::Container;
        _marker.color.r = 0.1;
        _marker.color.g = 1.0;
        _marker.color.b = 0.1;
        
    } else {
        
        _type = Type::None;
        _marker.color.r = 0.8;
        _marker.color.g = 0.0;
        _marker.color.b = 0.8;
    }
    
    return true;
}

bool ObjectCluster::findPoint(Point searchPoint, float radius) {
    
    _kdtree.setInputCloud (_cloud);
    
    if ( _kdtree.radiusSearch (searchPoint, radius, _point_idx_found, _point_radius_squared_distance, 1) > 0 )
    {
        //for (size_t i = 0; i < pointIdxRadiusSearch.size (); ++i)
//             std::cout << "    "  <<   _cloud)->points[ pointIdxRadiusSearch[0] ].x 
//                     << " " << _cloud->points[ pointIdxRadiusSearch[0] ].y 
//                     << " " << _cloud->points[ pointIdxRadiusSearch[0] ].z 
//                     << " (squared distance: " << pointRadiusSquaredDistance[0] << ")" << std::endl;
//                     std::cout << std::endl;
                
        _selected_cluster = true;
       _marker.color.a = 0.6;
        
    } else {
        _marker.color.a = 0.3; 
        _selected_cluster = false;
    }
    
    return _selected_cluster;
}

bool ObjectCluster::fillTransformGoal() {
    
    if (_type == Type::None){ 
        return false;
    }
    

    _ref_T_goal.header.stamp = ros::Time::now();
        
    _ref_T_goal.header.frame_id = _cloud->header.frame_id;
    
    if (_type == Type::Object){
        _ref_T_goal.child_frame_id = "object_goal";

        //AABB  
        _ref_T_goal.transform.translation.x = _position(0) - _dimensions(0)/2.0;
        _ref_T_goal.transform.translation.y = _position(1);
        _ref_T_goal.transform.translation.z = _position(2);
        
    } else if (_type == Type::Container) {
        _ref_T_goal.child_frame_id = "container_goal";

        _ref_T_goal.transform.translation.x = _position(0) - (2.0/3.0 * (_dimensions(0)/2.0));
        _ref_T_goal.transform.translation.y = _position(1);
        _ref_T_goal.transform.translation.z = _position(2) + _dimensions(2)/2 + 0.1;
        
    }
    
    _ref_T_goal.transform.rotation.x = 0;
    _ref_T_goal.transform.rotation.y = 0;
    _ref_T_goal.transform.rotation.z = 0;
    _ref_T_goal.transform.rotation.w = 1;
    
        //OBB TRIAL, IT IS WRONG
    //         msg.transform.translation.x = object_clusters.at(i).mass_center(0) + object_clusters.at(i).min_point_OBB.y;
    //         msg.transform.translation.y = object_clusters.at(i).mass_center(1) + (object_clusters.at(i).max_point_OBB.z - object_clusters.at(i).min_point_OBB.z)/2;
    //         msg.transform.translation.z = object_clusters.at(i).mass_center(2) + (object_clusters.at(i).max_point_OBB.x - object_clusters.at(i).min_point_OBB.x)/2;

    return true;
}


/******************************************************************************** *****/

ObjectClusterExtractor::ObjectClusterExtractor (ros::NodeHandle* nh) {
    
    this->nh = nh;
    
    nh->param<int>("max_clusters", max_clusters, 10);
    nh->param<bool>("publishPlane", publishPlane, true);
    nh->param<bool>("publishObjectsOnTable", publishObjectsOnTable, true);
    nh->param<bool>("publishSingleObjCloud", publishSingleObjCloud, true);
    nh->param<bool>("publishSingleObjTF", publishSingleObjTF, true);
    nh->param<bool>("publishSingleObjBoundingBox", publishSingleObjBoundingBox, true);

    tf_listener = std::make_unique<tf2_ros::TransformListener>(tf_buffer);
    
    cloud_sub = nh->subscribe<PointCloud>("/D435_head_camera/depth/color/points", 1, &ObjectClusterExtractor::cloudClbk, this);
    
    if (publishPlane) { 
        cloud_plane_pub = nh->advertise<PointCloud>("cloud_plane", 1);
    }
    
    if (publishObjectsOnTable) {
        cloud_objects_pub = nh->advertise<PointCloud>("cloud_objects", 1);
    }
    
    if (publishSingleObjCloud) {
        cloud_tmp_pub.resize(max_clusters);
        for (int i=0; i<max_clusters; i++) {
            cloud_tmp_pub.at(i) = nh->advertise<PointCloud>("cloud_tmp_"+ std::to_string(i), 1);
        }
    }
    
    object_clusters.resize(max_clusters); //max cluster we detect
    for (int i=0; i<object_clusters.size(); i++) {
        object_clusters.at(i)._cloud = boost::make_shared<PointCloud>();
    }
    
    cloud = boost::make_shared<PointCloud>();
    cloud_plane = boost::make_shared<PointCloud>();
    cloud_objects = boost::make_shared<PointCloud>();
    
    if (publishSingleObjBoundingBox){
        marker_pub = nh->advertise<visualization_msgs::MarkerArray>("objects_bounding", 1);
    }
    
    selected_object_srv = nh->advertiseService("object_selected", &ObjectClusterExtractor::selectedObjectClbk, this);
    
    //for some debug
    //tmp_pub = nh->advertise<sensor_msgs::Image>("/image_trial",1);
    
}

int ObjectClusterExtractor::run () {
    
    if (cloud->size() == 0) {
        std::cout << "Point cloud empty" << std::endl;

        return -1;
    }
    
    //change reference frame
    pcl_ros::transformPointCloud (ref_frame, *cloud, *cloud, tf_buffer);

    filterOnZaxis();
    
    extractPlaneAndObjects();

    //Euclidean Cluster Extraction, to divide each box
    clusterExtraction();
        
    if (publishSingleObjBoundingBox)
    {  
        markerArrayMsg.markers.clear();
    }
    
    if (publishSingleObjTF) { 
        transforms.clear();
    }

    
    //moment of inertia
    for (int i=0; i<n_clusters; i++) {
        
//         if (object_clusters.at(i).momentOfInertiaOBB()) {
//             std::cout << "WARN: momentOfInertiaOBB gets return false for cluster '" << std::to_string(i) << "'" << std::endl;
//         }
        
        if (object_clusters.at(i).momentOfInertiaAABB()) {
            //return always false IDK why...
            //std::cout << "WARN: momentOfInertiaAABB gets return false for cluster '" << std::to_string(i) << "'" << std::endl;
        }
        object_clusters.at(i).categorizeCluster();
        
    }
    
    selected_cluster = -1;
    Point searchPoint(refcloud_T_laser.transform.translation.x, refcloud_T_laser.transform.translation.y, refcloud_T_laser.transform.translation.z);
    for (int i=0; i<n_clusters; i++) {
        
        if (object_clusters.at(i).findPoint(searchPoint)) {
                
            if (object_clusters.at(i).fillTransformGoal()) {
                
                selected_cluster = i;
            }
        }
    }
    
    
    if (publishSingleObjBoundingBox)
    {   

        for (int i=0; i<n_clusters; i++) {
            object_clusters.at(i).fillMarker(i);
            markerArrayMsg.markers.push_back(object_clusters.at(i)._marker);
        }
        
        marker_pub.publish(markerArrayMsg);
    }
    
    if (publishSingleObjTF) { 

        for (int i=0; i<n_clusters; i++) {
            object_clusters.at(i).fillTransform(i);
            transforms.push_back(object_clusters.at(i)._ref_T_cloud);
        }

        tf_broadcaster.sendTransform(transforms);
    }
    
    return 0;
}

bool ObjectClusterExtractor::filterOnZaxis() {
    
    //filter along z axis
    pcl::PassThrough<Point> pass;
    pass.setInputCloud (cloud);
    pass.setFilterFieldName ("z");
    //we remove stuff that are too low (like floor) or too high
    pass.setFilterLimits (-pelvis_high + 0.1 , 2);
    //pass.setNegative (true); //Set to true if we want to return the data outside the interval specified by setFilterLimits (min, max) Default: false.
    pass.filter (*cloud);
    
    return true;
}

bool ObjectClusterExtractor::extractPlaneAndObjects() {
    
    pcl::ModelCoefficients::Ptr coefficients (new pcl::ModelCoefficients);
    pcl::PointIndices::Ptr inliers (new pcl::PointIndices);
    // Create the segmentation object
    pcl::SACSegmentation<Point> seg;
    // Optional
    seg.setOptimizeCoefficients (true);
    // Mandatory
    seg.setModelType (pcl::SACMODEL_PERPENDICULAR_PLANE);
    seg.setAxis(Eigen::Vector3f::UnitZ());
    seg.setEpsAngle(0.15); //almost 8.6 degrees as threshold to consider planes as parallel
    seg.setMethodType (pcl::SAC_RANSAC);
    seg.setDistanceThreshold (0.01);

    seg.setInputCloud (cloud);
    seg.segment (*inliers, *coefficients);

    if (inliers->indices.size () == 0)
    {
        PCL_ERROR ("Could not estimate a planar model for the given dataset.\n");
        return false;
    }
                                        
    pcl::ExtractIndices<Point> extract;
    extract.setInputCloud (cloud);
    extract.setIndices (inliers);
    extract.setNegative (false);

    // Get the points associated with the planar surface
    extract.filter (*cloud_plane);

    //*************************pub pointcloud plane  
    if (publishPlane) {
        cloud_plane_pub.publish(cloud_plane);
    }

    //exctract objects on the plane
    pcl::PassThrough<Point> pass;

    cloud_objects = cloud;
    
    //first remove the table-plane
    extract.setInputCloud (cloud_objects);
    extract.setIndices (inliers);
    extract.setNegative (true);

    // Get the points associated with the planar surface
    extract.filter (*cloud_objects);
    
    //get table margins, and remove stuff outside
    
    Point minPt, maxPt;
    pcl::getMinMax3D (*cloud_plane, minPt, maxPt);
    
    double margin = 0; //
    pass.setInputCloud (cloud_objects);
    pass.setFilterFieldName ("x");
    pass.setFilterLimits (minPt.x - margin,  maxPt.x + margin);
    //pass.setNegative (true); //Set to true if we want to return the data outside the interval specified by setFilterLimits (min, max) Default: false.
    pass.filter (*cloud_objects);
    
    pass.setInputCloud (cloud_objects);
    pass.setFilterFieldName ("y");
    pass.setFilterLimits (minPt.y - margin,  maxPt.y + margin);
    //pass.setNegative (true); //Set to true if we want to return the data outside the interval specified by setFilterLimits (min, max) Default: false.
    pass.filter (*cloud_objects);
    
    pass.setInputCloud (cloud_objects);
    pass.setFilterFieldName ("z");
    pass.setFilterLimits (minPt.z - margin,  2);
    //pass.setNegative (true); //Set to true if we want to return the data outside the interval specified by setFilterLimits (min, max) Default: false.
    pass.filter (*cloud_objects);
    
    if (publishObjectsOnTable) {
        cloud_objects_pub.publish(cloud_objects);
    }
    
    return true;
}

bool ObjectClusterExtractor::clusterExtraction(){
 
    // Create the filtering object: downsample the dataset using a leaf size of 1cm
    pcl::VoxelGrid<Point> vg;
    pcl::PointCloud<Point>::Ptr cloud_filtered (new pcl::PointCloud<Point>);
    cloud_filtered = cloud;
    vg.setInputCloud (cloud_objects);
    vg.setLeafSize (0.01f, 0.01f, 0.01f);
    vg.filter (*cloud_filtered);
    
    // Creating the KdTree object for the search method of the extraction
    pcl::search::KdTree<Point>::Ptr tree (new pcl::search::KdTree<Point>);
    tree->setInputCloud (cloud_filtered);

    std::vector<pcl::PointIndices> cluster_indices;
    pcl::EuclideanClusterExtraction<Point> ec;
    ec.setClusterTolerance (0.02); // 2cm
    ec.setMinClusterSize (100);
    ec.setMaxClusterSize (2500);
    ec.setSearchMethod (tree);
    ec.setInputCloud (cloud_filtered);
    ec.extract (cluster_indices);

    if (object_clusters.size() < cluster_indices.size()) {
        std::cout << "WARN! clusters found exceed the max number of " <<
            object_clusters.size() << " I will not store the exceedent ones " << std::endl;
    }
    
    n_clusters = 0;
    for (std::vector<pcl::PointIndices>::const_iterator it = cluster_indices.begin (); it != cluster_indices.end (); ++it)
    {

        object_clusters.at(n_clusters)._cloud->points.clear();
        object_clusters.at(n_clusters)._cloud->header.frame_id = cloud->header.frame_id;
        
        for (std::vector<int>::const_iterator pit = it->indices.begin (); pit != it->indices.end (); ++pit){
            object_clusters.at(n_clusters)._cloud->push_back ((*cloud_filtered)[*pit]); //*
        }
        object_clusters.at(n_clusters)._cloud->width = object_clusters.at(n_clusters)._cloud->size ();
        object_clusters.at(n_clusters)._cloud->height = 1;
        object_clusters.at(n_clusters)._cloud->is_dense = true;

        if (publishSingleObjCloud && n_clusters<cloud_tmp_pub.size()) {
            cloud_tmp_pub.at(n_clusters).publish(object_clusters.at(n_clusters)._cloud);
        }
        
        n_clusters++;
    }

    //std::cout << "Found " << cluster_indices.size() << " clusters " << std::endl;
    return true;
}


void ObjectClusterExtractor::cloudClbk(const PointCloud::ConstPtr& msg)
{
    *cloud = *msg;
}

bool ObjectClusterExtractor::selectedObjectClbk(tpo_msgs::ClusterObject::Request &req, 
                                                tpo_msgs::ClusterObject::Response &res) {
    
    //req is empty
    
    res.header.stamp = ros::Time::now();
    
    if (selected_cluster == -1) {
        ROS_WARN_STREAM_THROTTLE(1, "No clusters selected!");
        res.type = "none";
        
    } else {
    
        //cloud frame id is ref_frame, which is torso_2 as in header file
        res.header.frame_id = object_clusters.at(selected_cluster)._cloud->header.frame_id;
        res.child_frame_id = "box_cloud_" + std::to_string(selected_cluster);
        
        res.ref_T_object.translation.x = object_clusters.at(selected_cluster)._position (0); 
        res.ref_T_object.translation.y = object_clusters.at(selected_cluster)._position (1);
        res.ref_T_object.translation.z = object_clusters.at(selected_cluster)._position (2);
        
        res.ref_T_object.rotation.x = object_clusters.at(selected_cluster)._rotation.x();
        res.ref_T_object.rotation.y = object_clusters.at(selected_cluster)._rotation.y();
        res.ref_T_object.rotation.z = object_clusters.at(selected_cluster)._rotation.z();
        res.ref_T_object.rotation.w = object_clusters.at(selected_cluster)._rotation.w();
        
        res.dimensions.x = object_clusters.at(selected_cluster)._dimensions(0);
        res.dimensions.y = object_clusters.at(selected_cluster)._dimensions(1);
        res.dimensions.z = object_clusters.at(selected_cluster)._dimensions(2);
        
        switch (object_clusters.at(selected_cluster)._type) {
            case ObjectCluster::Type::Surface: {
                res.type = "surface";
                break;
            }
            case ObjectCluster::Type::Container: {
                res.type = "container";
                break;
            }
            case ObjectCluster::Type::Object: {
                res.type = "object";
                break;
            }
            default: {
                res.type = "none";
            }
        }
    }
        
    
    return true;
}

void ObjectClusterExtractor::getTransforms()
{
    try {
        
      //cam_T_pelvis = tf_buffer.lookupTransform(camera_frame, reference_frame, ros::Time(0));
      pelvis_T_wheel = tf_buffer.lookupTransform("pelvis", "wheel_1", ros::Time(0));
      pelvis_high = std::abs(pelvis_T_wheel.transform.translation.z);
      refcloud_T_laser = tf_buffer.lookupTransform(ref_frame, "laser", ros::Time(0));
      
    } 
    catch (tf2::TransformException &ex) {
      ROS_WARN("%s",ex.what());
      ros::Duration(1.0).sleep();
    }
}




/******************************** *****************************************/



/******************************** *****************************************/

int main ( int argc, char **argv ) {

    ros::init ( argc, argv, "TPOObjectClusterExtractor" );
    ros::NodeHandle nh("~");
    
    ObjectClusterExtractor objectClusterExtractor(&nh);
    
    ros::Rate r(10);
    while(ros::ok()) {
        
       objectClusterExtractor.getTransforms();
       objectClusterExtractor.run();


        ros::spinOnce();
        r.sleep();
    }
    
    return 0;
    
}
