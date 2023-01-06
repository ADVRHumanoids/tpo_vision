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

#include <tpo_vision/planarSegmentationNode.h>

PlanarSegmentation::PlanarSegmentation (ros::NodeHandle* nh) {
    
    this->nh = nh;
    
    nh->param<int>("max_clusters", max_clusters, 10);
    nh->param<bool>("publishPlane", publishPlane, true);
    nh->param<bool>("publishObjectsOnTable", publishObjectsOnTable, true);
    nh->param<bool>("publishSingleObjCloud", publishSingleObjCloud, true);
    nh->param<bool>("publishSingleObjTF", publishSingleObjTF, true);
    nh->param<bool>("publishSingleObjBoundingBox", publishSingleObjBoundingBox, true);

    tf_listener = std::make_unique<tf2_ros::TransformListener>(tf_buffer);
    
    cloud_sub = nh->subscribe<PointCloud>("/D435_head_camera/depth/color/points", 1, &PlanarSegmentation::cloudClbk, this);
    
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
    
    cloud_clusters.resize(max_clusters); //max cluster we detect
    for (int i=0; i<cloud_clusters.size(); i++) {
        cloud_clusters.at(i) = boost::make_shared<PointCloud>();
    }
    
    cloud = boost::make_shared<PointCloud>();
    cloud_plane = boost::make_shared<PointCloud>();
    cloud_objects = boost::make_shared<PointCloud>();
    
//     viewer = boost::make_shared<pcl::visualization::PCLVisualizer>("3D Viewer");
//     viewer->setBackgroundColor (0, 0, 0);
//     viewer->addCoordinateSystem (1.0);
//     viewer->initCameraParameters ();
    
    if (publishSingleObjBoundingBox){
        marker_pub = nh->advertise<visualization_msgs::MarkerArray>("objects_bounding", 1);
    }
    
    //for some debug
    //tmp_pub = nh->advertise<sensor_msgs::Image>("/image_trial",1);

}

int PlanarSegmentation::run () {
    
//     pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
// 
//     Fill in the cloud data
//     cloud->width  = 15;
//     cloud->height = 1;
//     cloud->points.resize (cloud->width * cloud->height);
// 
//     Generate the data
//     for (auto& point: *cloud)
//     {
//         point.x = 1024 * rand () / (RAND_MAX + 1.0f);
//         point.y = 1024 * rand () / (RAND_MAX + 1.0f);
//         point.z = 1.0;
//     }
// 
//     Set a few outliers
//     (*cloud)[0].z = 2.0;
//     (*cloud)[3].z = -2.0;
//     (*cloud)[6].z = 4.0;
    
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
    
    //moment of inertia
    for (int i=0; i<n_clusters; i++) {
        
        momentOfInertia(i, cloud_clusters.at(i));
    }
    
    if (publishSingleObjBoundingBox)
    {   
        marker_pub.publish(markerArrayMsg);
    }
    
    return 0;
}

bool PlanarSegmentation::filterOnZaxis() {
    
    //filter along z axis
    pcl::PassThrough<pcl::PointXYZ> pass;
    pass.setInputCloud (cloud);
    pass.setFilterFieldName ("z");
    //we remove stuff that are too low (like floor) or too high
    pass.setFilterLimits (-pelvis_high + 0.1 , 2);
    //pass.setNegative (true); //Set to true if we want to return the data outside the interval specified by setFilterLimits (min, max) Default: false.
    pass.filter (*cloud);
    
    return true;
}

bool PlanarSegmentation::extractPlaneAndObjects() {
    
    pcl::ModelCoefficients::Ptr coefficients (new pcl::ModelCoefficients);
    pcl::PointIndices::Ptr inliers (new pcl::PointIndices);
    // Create the segmentation object
    pcl::SACSegmentation<pcl::PointXYZ> seg;
    // Optional
    seg.setOptimizeCoefficients (true);
    // Mandatory
    seg.setModelType (pcl::SACMODEL_PLANE);
    seg.setMethodType (pcl::SAC_RANSAC);
    seg.setDistanceThreshold (0.01);

    seg.setInputCloud (cloud);
    seg.segment (*inliers, *coefficients);

    if (inliers->indices.size () == 0)
    {
        PCL_ERROR ("Could not estimate a planar model for the given dataset.\n");
        return false;
    }
                                        
    pcl::ExtractIndices<pcl::PointXYZ> extract;
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
    pcl::PassThrough<pcl::PointXYZ> pass;

    cloud_objects = cloud;
    
    //first remove the table-plane
    extract.setInputCloud (cloud_objects);
    extract.setIndices (inliers);
    extract.setNegative (true);

    // Get the points associated with the planar surface
    extract.filter (*cloud_objects);
    
    //get table margins, and remove stuff outside
    
    pcl::PointXYZ minPt, maxPt;
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

bool PlanarSegmentation::clusterExtraction(){
 
    // Create the filtering object: downsample the dataset using a leaf size of 1cm
    pcl::VoxelGrid<pcl::PointXYZ> vg;
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered (new pcl::PointCloud<pcl::PointXYZ>);
    cloud_filtered = cloud;
    vg.setInputCloud (cloud_objects);
    vg.setLeafSize (0.01f, 0.01f, 0.01f);
    vg.filter (*cloud_filtered);
    
    // Create the segmentation object for the planar model and set all the parameters
    /*
    pcl::SACSegmentation<pcl::PointXYZ> seg;
    pcl::PointIndices::Ptr inliers (new pcl::PointIndices);
    pcl::ModelCoefficients::Ptr coefficients (new pcl::ModelCoefficients);
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_plane2 (new pcl::PointCloud<pcl::PointXYZ> ());
    seg.setOptimizeCoefficients (true);
    seg.setModelType (pcl::SACMODEL_PLANE);
    seg.setMethodType (pcl::SAC_RANSAC);
    seg.setMaxIterations (100);
    seg.setDistanceThreshold (0.02);

    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_f (new pcl::PointCloud<pcl::PointXYZ>);
    int i=0;
    int nr_points = (int) cloud_filtered->size ();
    while (cloud_filtered->size () > 0.3 * nr_points)
    {
        // Segment the largest planar component from the remaining cloud
        seg.setInputCloud (cloud_filtered);
        seg.segment (*inliers, *coefficients);
        if (inliers->indices.size () == 0)
        {
            std::cout << "Could not estimate a planar model for the given dataset." << std::endl;
            break;
        }

        // Extract the planar inliers from the input cloud
        pcl::ExtractIndices<pcl::PointXYZ> extract;
        extract.setInputCloud (cloud_filtered);
        extract.setIndices (inliers);
        extract.setNegative (false);

        // Get the points associated with the planar surface
        extract.filter (*cloud_plane2);
        //std::cout << "PointCloud representing the planar component: " << cloud_plane2->size () << " data points." << std::endl;

        // Remove the planar inliers, extract the rest
        extract.setNegative (true);
        extract.filter (*cloud_f);
        *cloud_filtered = *cloud_f;
    }
*/
    
    // Creating the KdTree object for the search method of the extraction
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZ>);
    tree->setInputCloud (cloud_filtered);

    std::vector<pcl::PointIndices> cluster_indices;
    pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
    ec.setClusterTolerance (0.02); // 2cm
    ec.setMinClusterSize (25);
    ec.setMaxClusterSize (2500);
    ec.setSearchMethod (tree);
    ec.setInputCloud (cloud_filtered);
    ec.extract (cluster_indices);

    n_clusters = 0;

    for (std::vector<pcl::PointIndices>::const_iterator it = cluster_indices.begin (); it != cluster_indices.end (); ++it)
    {
        if (n_clusters > cloud_clusters.size()) {
            std::cout << "WARN! clusters found exceed the max number of " <<
                cloud_clusters.size() << " I will not store the exceedent ones " << std::endl;
        }
        cloud_clusters.at(n_clusters)->points.clear();
        cloud_clusters.at(n_clusters)->header.frame_id = cloud->header.frame_id;
        
        for (std::vector<int>::const_iterator pit = it->indices.begin (); pit != it->indices.end (); ++pit){
            cloud_clusters.at(n_clusters)->push_back ((*cloud_filtered)[*pit]); //*
        }
        cloud_clusters.at(n_clusters)->width = cloud_clusters.at(n_clusters)->size ();
        cloud_clusters.at(n_clusters)->height = 1;
        cloud_clusters.at(n_clusters)->is_dense = true;


        if (publishSingleObjCloud && n_clusters<cloud_tmp_pub.size()) {
            cloud_tmp_pub.at(n_clusters).publish(cloud_clusters.at(n_clusters));
        }
        
        n_clusters++;
    }

    std::cout << "Found " << cluster_indices.size() << " clusters " << std::endl;
    return true;
}

void PlanarSegmentation::momentOfInertia(const int id, pcl::PointCloud<pcl::PointXYZ>::Ptr cloud) {
    
    pcl::MomentOfInertiaEstimation <pcl::PointXYZ> feature_extractor;
    feature_extractor.setInputCloud (cloud);
    feature_extractor.compute ();

    std::vector <float> moment_of_inertia;
    std::vector <float> eccentricity;
    pcl::PointXYZ min_point_AABB;
    pcl::PointXYZ max_point_AABB;
    pcl::PointXYZ min_point_OBB;
    pcl::PointXYZ max_point_OBB;
    pcl::PointXYZ position_OBB;
    Eigen::Matrix3f rotational_matrix_OBB;
    float major_value, middle_value, minor_value;
    Eigen::Vector3f major_vector, middle_vector, minor_vector;
    Eigen::Vector3f mass_center;

    feature_extractor.getMomentOfInertia (moment_of_inertia);
    feature_extractor.getEccentricity (eccentricity);
    feature_extractor.getAABB (min_point_AABB, max_point_AABB);
    feature_extractor.getOBB (min_point_OBB, max_point_OBB, position_OBB, rotational_matrix_OBB);
    feature_extractor.getEigenValues (major_value, middle_value, minor_value);
    feature_extractor.getEigenVectors (major_vector, middle_vector, minor_vector);
    feature_extractor.getMassCenter (mass_center);

//     viewer->addPointCloud<pcl::PointXYZ> (cloud, "sample cloud");
//     viewer->addCube (min_point_AABB.x, max_point_AABB.x, min_point_AABB.y, max_point_AABB.y, min_point_AABB.z, max_point_AABB.z, 1.0, 1.0, 0.0, "AABB");
//     viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_REPRESENTATION, pcl::visualization::PCL_VISUALIZER_REPRESENTATION_WIREFRAME, "AABB");

//    Eigen::Vector3f position (position_OBB.x, position_OBB.y, position_OBB.z);
    
//     std::cout << "rot:\n" <<
//         rotational_matrix_OBB  << std::endl;
    
    Eigen::Quaternionf quat (rotational_matrix_OBB);
    
//     std::cout << "quat:\n" <<
//         quat.x() << " " << quat.y() << " " << quat.z() << " " << quat.w()  << std::endl;
//     viewer->addCube (position, quat, max_point_OBB.x - min_point_OBB.x, max_point_OBB.y - min_point_OBB.y, max_point_OBB.z - min_point_OBB.z, "OBB");
//     viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_REPRESENTATION, pcl::visualization::PCL_VISUALIZER_REPRESENTATION_WIREFRAME, "OBB");

//     pcl::PointXYZ center (mass_center (0), mass_center (1), mass_center (2));
//     pcl::PointXYZ x_axis (major_vector (0) + mass_center (0), major_vector (1) + mass_center (1), major_vector (2) + mass_center (2));
//     pcl::PointXYZ y_axis (middle_vector (0) + mass_center (0), middle_vector (1) + mass_center (1), middle_vector (2) + mass_center (2));
//     pcl::PointXYZ z_axis (minor_vector (0) + mass_center (0), minor_vector (1) + mass_center (1), minor_vector (2) + mass_center (2));
//     viewer->addLine (center, x_axis, 1.0f, 0.0f, 0.0f, "major eigen vector");
//     viewer->addLine (center, y_axis, 0.0f, 1.0f, 0.0f, "middle eigen vector");
//     viewer->addLine (center, z_axis, 0.0f, 0.0f, 1.0f, "minor eigen vector");

 //   viewer->spinOnce (100);
    
    if (publishSingleObjTF) {
        
        geometry_msgs::TransformStamped tf;
        
        //tf.header.frame_id = ref_frame;
        tf.child_frame_id = "box_cloud_" + std::to_string(id);
        tf.header.stamp = ros::Time::now();
            
        //AABB version
//         tf.header.frame_id = ref_frame;
//         tf.transform.translation.x = 0.5 * (max_point_AABB.x + min_point_AABB.x);
//         tf.transform.translation.y = 0.5 * (max_point_AABB.y + min_point_AABB.y);
//         tf.transform.translation.z = 0.5 * (max_point_AABB.z + min_point_AABB.z);
//         tf.transform.rotation.x = 0;
//         tf.transform.rotation.y = 0;
//         tf.transform.rotation.z = 0;
//         tf.transform.rotation.w = 1;
 
        //OBB version
        tf.header.frame_id = cloud->header.frame_id;

        tf.transform.translation.x = mass_center (0); 
        tf.transform.translation.y = mass_center (1);
        tf.transform.translation.z = mass_center (2);
        
        quat.normalize();
        tf.transform.rotation.x = quat.x();
        tf.transform.rotation.y = quat.y();
        tf.transform.rotation.z = quat.z();
        tf.transform.rotation.w = quat.w();
        
        tf_broadcaster.sendTransform(tf);
        
    }
        
    
    if (publishSingleObjBoundingBox) {
        
        //AABB version
//         double x_size, y_size, z_size;
//         
//         x_size = max_point_AABB.x - min_point_AABB.x;
//         y_size = max_point_AABB.y - min_point_AABB.y;
//         z_size = max_point_AABB.z - min_point_AABB.z;
//     
//         addBoundingBoxMarker(id, "box_cloud_" + std::to_string(id), x_size, y_size, z_size);
        
        //OBB version
        visualization_msgs::Marker marker;
        marker.header.frame_id = cloud->header.frame_id;
        marker.header.stamp = ros::Time();
        marker.ns = "bounding_boxes";
        marker.id = id;
        marker.type = visualization_msgs::Marker::CUBE;
        marker.action = visualization_msgs::Marker::ADD;
        marker.pose.position.x = mass_center (0); ;
        marker.pose.position.y = mass_center (1); ;
        marker.pose.position.z = mass_center (2); ;
        marker.pose.orientation.x = quat.x();
        marker.pose.orientation.y = quat.y();
        marker.pose.orientation.z = quat.z();
        marker.pose.orientation.w = quat.w();
        marker.scale.x = max_point_OBB.x - min_point_OBB.x;
        marker.scale.y = max_point_OBB.y - min_point_OBB.y;
        marker.scale.z = max_point_OBB.z - min_point_OBB.z;
        marker.color.a = 0.3; // Don't forget to set the alpha!
        marker.color.r = 1.0;
        marker.color.g = 0.0;
        marker.color.b = 0.0;
        
        markerArrayMsg.markers.push_back(marker);
        
    }
}

void PlanarSegmentation::addBoundingBoxMarker(unsigned int id, std::string frame_id, double x, double y, double z) {
    
    visualization_msgs::Marker marker;
    marker.header.frame_id = frame_id;
    marker.header.stamp = ros::Time();
    marker.ns = "bounding_boxes";
    marker.id = id;
    marker.type = visualization_msgs::Marker::CUBE;
    marker.action = visualization_msgs::Marker::ADD;
    marker.pose.position.x = 0;
    marker.pose.position.y = 0;
    marker.pose.position.z = 0;
    marker.pose.orientation.x = 0.0;
    marker.pose.orientation.y = 0.0;
    marker.pose.orientation.z = 0.0;
    marker.pose.orientation.w = 1.0;
    marker.scale.x = x;
    marker.scale.y = y;
    marker.scale.z = z;
    marker.color.a = 0.3; // Don't forget to set the alpha!
    marker.color.r = 0.0;
    marker.color.g = 1.0;
    marker.color.b = 0.0;
    
    markerArrayMsg.markers.push_back(marker);
}


void PlanarSegmentation::cloudClbk(const PointCloud::ConstPtr& msg)
{
    *cloud = *msg;
}

void PlanarSegmentation::getTransforms()
{
    try {
        
      //cam_T_pelvis = tf_buffer.lookupTransform(camera_frame, reference_frame, ros::Time(0));
      pelvis_T_wheel = tf_buffer.lookupTransform("pelvis", "wheel_1", ros::Time(0));
      pelvis_high = std::abs(pelvis_T_wheel.transform.translation.z);
      
    } 
    catch (tf2::TransformException &ex) {
      ROS_WARN("%s",ex.what());
      ros::Duration(1.0).sleep();
    }
}




/******************************** *****************************************/



/******************************** *****************************************/

int main ( int argc, char **argv ) {

    ros::init ( argc, argv, "TPOPlanarSegmentation" );
    ros::NodeHandle nh("~");
    
    PlanarSegmentation planarSegmentation(&nh);
    
    ros::Rate r(10);
    while(ros::ok()) {
        
       planarSegmentation.getTransforms();
       planarSegmentation.run();


        ros::spinOnce();
        r.sleep();
    }
    
    return 0;
    
}
