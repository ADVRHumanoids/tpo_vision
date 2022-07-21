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
    
    tf_listener = std::make_unique<tf2_ros::TransformListener>(tf_buffer);
    
    cloud_sub = nh->subscribe<PointCloud>("/D435_head_camera/depth/color/points", 1, &PlanarSegmentation::cloudClbk, this);
    
    cloud_plane_pub = nh->advertise<PointCloud>("cloud_plane", 1);
    cloud_objects_pub = nh->advertise<PointCloud>("cloud_objects", 1);
    cloud_tmp_pub.resize(3);
    for (int i=0; i<3; i++) {
        cloud_tmp_pub.at(i) = nh->advertise<PointCloud>("cloud_tmp_"+ std::to_string(i), 1);
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
    
    //marker pub for bounding boxes of objects
    marker_pub = nh->advertise<visualization_msgs::MarkerArray>("objects_bounding", 1);
    
    /******************* ***************************/
    camera_info_sub = nh->subscribe<sensor_msgs::CameraInfo>("/D435_head_camera/aligned_depth_to_color/camera_info", 1, &PlanarSegmentation::cameraInfoClbk, this);
 
    depth_image_sub = nh->subscribe<sensor_msgs::Image>("/D435_head_camera/aligned_depth_to_color/image_raw", 1, &PlanarSegmentation::depthImageClbk, this);
    
    tmp_pub = nh->advertise<sensor_msgs::Image>("/image_trial",1);

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
    
    bool publishPlane = true;
    bool publishObjectsOnTable = true;  
    extractPlaneAndObjects(publishPlane, publishObjectsOnTable);

    //Euclidean Cluster Extraction, to divide each box
    bool publishSingleObjCloud = true;
    clusterExtraction(publishSingleObjCloud);
    
    markerArrayMsg.markers.clear();
    //moment of inertia
    for (int i=0; i<n_clusters; i++) {
        geometry_msgs::TransformStamped tf;
        double x, y, z;
        tf.transform = momentOfInertia(cloud_clusters.at(i), &x, &y, &z);
        tf.header.frame_id = ref_frame;
        tf.child_frame_id = "box_" + std::to_string(i);
        tf.header.stamp = ros::Time::now();
        
        tf_broadcaster.sendTransform(tf);
        
        addBoundingBoxMarker(i, tf.child_frame_id, x, y, z);

    }
    
    marker_pub.publish(markerArrayMsg);

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

bool PlanarSegmentation::extractPlaneAndObjects(bool publishPlane, bool publishObjectsOnTable) {
    
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

bool PlanarSegmentation::clusterExtraction(bool publishSingleObjCloud){
 
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

geometry_msgs::Transform PlanarSegmentation::momentOfInertia(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, double* x_size, double* y_size, double* z_size) {
    
    geometry_msgs::Transform tf;
    
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

    Eigen::Vector3f position (position_OBB.x, position_OBB.y, position_OBB.z);
    Eigen::Quaternionf quat (rotational_matrix_OBB);
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
        
    tf.translation.x = 0.5 * (max_point_AABB.x + min_point_AABB.x);
    tf.translation.y = 0.5 * (max_point_AABB.y + min_point_AABB.y);
    tf.translation.z = 0.5 * (max_point_AABB.z + min_point_AABB.z);
    
//     tf.rotation.x = quat.x();
//     tf.rotation.y = quat.y();
//     tf.rotation.z = quat.z();
//     tf.rotation.w = quat.w();
    tf.rotation.x = 0;
    tf.rotation.y = 0;
    tf.rotation.z = 0;
    tf.rotation.w = 1;
    
    if (x_size != nullptr && y_size != nullptr && z_size!= nullptr) {
        *x_size = max_point_AABB.x - min_point_AABB.x;
        *y_size = max_point_AABB.y - min_point_AABB.y;
        *z_size = max_point_AABB.z - min_point_AABB.z;
    }
        
    return tf;
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
void PlanarSegmentation::cameraInfoClbk(const sensor_msgs::CameraInfoConstPtr& msg) {
    
    cam_info = *msg;
    
    disp2Depth = cv::Mat::zeros(4, 4, CV_64FC1);
    
    disp2Depth.at<double>(0,0) = 1;
    disp2Depth.at<double>(1,1) = 1;
    
//     disp2Depth.at<double>(0,3) = - msg->K.at(3); //K is stored row-major
//     disp2Depth.at<double>(1,3) = - msg->K.at(5); //K is stored row-major
//     disp2Depth.at<double>(2,3) = - msg->K.at(0); //fx is the same of fy???
    
     //P is stored row-major
    disp2Depth.at<double>(0,3) = - msg->K.at(2); //cx
    disp2Depth.at<double>(1,3) = - msg->P.at(5); //cy
    disp2Depth.at<double>(2,3) = - msg->P.at(0); //fx is the same of fy???
    
    disp2Depth.at<double>(2,3) = - 1.0 / (msg->P.at(3)); // -1/Tx
    disp2Depth.at<double>(2,3) = - 1.0 / (msg->P.at(3)); //(cx - cx') / Tx
    
    camera_info_sub.shutdown();
}

void PlanarSegmentation::depthImageClbk(const sensor_msgs::ImageConstPtr& msg) {
    
    cv::Mat im;
    
    try
    {
        //original is CV_16U but, reprojectImageTo3D does not accept this, 
        //instead it accepts only  CV_8UC1 CV_16SC1 CV_32SC1 CV_32FC1
        //so we convert to CV_16SC1
        cv_bridge_image = *(cv_bridge::toCvShare(msg, sensor_msgs::image_encodings::TYPE_16UC1));
       // im = cv::Mat(msg->height, msg->width, CV_16SC1);
        im = cv::Mat(msg->height, msg->width, CV_16SC1, const_cast<uchar*>(&msg->data[0]), msg->step);
        cv::Point pt1 = cv::Point(986, 145);
        cv::circle(im, pt1, 0, cv::Scalar(0, 255, 0), 100, 8);
        
        //auto point = cloud->at(986, 145);

    }
    catch (cv_bridge::Exception& e)
    {
        ROS_ERROR("cv_bridge exception: %s", e.what());
        return;
    }
    
//     cv_bridge::CvImage trial_img(msg->header, msg->encoding, cv_bridge_image.image);
//     
//     tmp_pub.publish(trial_img);
//     
//     cv::imshow("Frame", im);
//     cv::waitKey(0);
    
}

bool PlanarSegmentation::extractSinglePointFrom2D(int pixel_x, int pixel_y) {
    
//     std::cout << cam_param << std::endl;
    
    if (cv_bridge_image.image.empty()) {
        std::cout << "image not yet arrived and/or is empty" << std::endl;
        return false;
    }

//     if (cam_param.empty()) {
//         std::cout << "cam param not yet arrived and/or is empty" << std::endl;
//         return false;
//     }
//     
    
//     cv::Mat cv_image_3d(cv_bridge_image.image.size().width, cv_bridge_image.image.size().height, cv_bridge_image.image.type());
//     std::cout << "projecting" << std::endl;
//     cv::reprojectImageTo3D(cv_bridge_image.image, cv_image_3d, cam_param, true);
//     std::cout << "3d image is size: " << cv_image_3d.size() << std::endl;
// 
//     std::cout << "POINT: " << cv_image_3d.at<cv::Point3d>(pixel_x, pixel_y) << std::endl;

    double depth_value = cv_bridge_image.image.at<double>(pixel_x,pixel_y);
    double cx, cy, fx, fy;
    cx = cam_info.K.at(2);
    cy = cam_info.K.at(5);
    fx = cam_info.K.at(0);
    fy = cam_info.K.at(4);
    double x = (pixel_x - cx) * depth_value / fx;
    double y = (pixel_y - cy) * depth_value / fy;
    double z = depth_value;
    
    std::cout << "c " << cx << " " << cy << std::endl;
    std::cout << "f " << fx << " " << fy << std::endl;
    std::cout << "depth_value " << depth_value << std::endl;
    std::cout << "x y z " << x << " " << y << " " << z << std::endl;
    
    geometry_msgs::TransformStamped t;
    t.header.frame_id = "D435_head_camera_color_optical_frame";
    t.header.stamp = ros::Time::now();
    t.child_frame_id = "laser_point";
    
    t.transform.translation.x = cloud->at(651, 217).x;
    t.transform.translation.y = cloud->at(651, 217).y;
    t.transform.translation.z = cloud->at(651, 217).z;
    
    t.transform.rotation.w = 1;
    
    tf_broadcaster.sendTransform(t);
 
    return true;
}


/******************************** *****************************************/

int main ( int argc, char **argv ) {

    ros::init ( argc, argv, "TPOPlanarSegmentation" );
    ros::NodeHandle nh("~");
    
    PlanarSegmentation planarSegmentation(&nh);
    
    ros::Rate r(10);
    while(ros::ok()) {
        
        //planarSegmentation.getTransforms();
       //planarSegmentation.run();
        planarSegmentation.extractSinglePointFrom2D(651, 217);


        ros::spinOnce();
        r.sleep();
    }
    
    return 0;
    
}
