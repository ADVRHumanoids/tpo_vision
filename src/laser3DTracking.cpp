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

#include <tpo_vision/laser3DTracking.h>

Laser3DTracking::Laser3DTracking (ros::NodeHandle* nh) {
    
    this->nh = nh;
    
    tf_listener = std::make_unique<tf2_ros::TransformListener>(tf_buffer);
    
    std::string pc_topic, image_topic, camera_info_topic, image_transport;
    nh->param<std::string>("point_cloud_topic", pc_topic, "/D435_head_camera/depth/color/points");
    
    nh->param<std::string>("ref_frame", ref_frame, "pelvis");
    nh->param<std::string>("laser_spot_frame", laser_spot_frame, "laser_spot");
    nh->param<std::string>("camera_frame", camera_frame, "D435_head_camera_color_optical_frame");
    
    std::string keypoint_topic;
    nh->param<std::string>("keypoint_topic", keypoint_topic, "detection_output_keypoint");
    
    keypoint_sub = nh->subscribe<tpo_msgs::KeypointImage>(keypoint_topic, 10, &Laser3DTracking::keypointSubClbk, this);
     
    /******************* CLOUD ***************************/
    cloud_sub = nh->subscribe<PointCloud>(pc_topic, 1, &Laser3DTracking::cloudClbk, this);
    cloud = boost::make_shared<PointCloud>();
    
    ref_T_spot.header.frame_id = ref_frame;
    ref_T_spot.child_frame_id = laser_spot_frame;
    
}

bool Laser3DTracking::isReady() {
    
    if (cloud->size() == 0) {
        ROS_WARN_STREAM_ONCE("Point cloud not yet arrived...");
        return false;
    } else {
        ROS_WARN_STREAM_ONCE("... Point cloud arrived");
    }
    
    if (keypoint_sub.getNumPublishers() < 1) {
        ROS_WARN_STREAM_ONCE("Nobody is publishing the 2d keypoints on '"<< keypoint_sub.getTopic() << "'...");
        return false;
    } else {
        ROS_WARN_STREAM_ONCE("Someone is publishing the 2d keypoints on '"<< keypoint_sub.getTopic() << "'...");

    }
    
    ROS_INFO_STREAM("Ready!");
    return true;
}

int Laser3DTracking::run () {

    //change reference frame
    pcl_ros::transformPointCloud (ref_frame, *cloud, *cloud, tf_buffer);
    
    if (keypoint_image.confidence <= 0 ){
    
        sendTransformFrom2D(keypoint_image.x_pixel, keypoint_image.y_pixel);
    } else {
        ROS_WARN_STREAM("Confidence of arrived keypoint detection message is zero!'");

    }

    return 0;
}


bool Laser3DTracking::sendTransformFrom2D(int pixel_x, int pixel_y) {
    
//     std::cout << cam_param << std::endl;
    
//     if (cv_bridge_image.image.empty()) {
//         std::cout << "image not yet arrived and/or is empty" << std::endl;
//         return false;
//     }

    ref_T_spot.header.stamp = ros::Time::now();
    
    auto pointXYZ = cloud->at(pixel_x, pixel_y);
    
    ref_T_spot.transform.translation.x = pointXYZ.x;
    ref_T_spot.transform.translation.y = pointXYZ.y;
    ref_T_spot.transform.translation.z = pointXYZ.z;
    
    ref_T_spot.transform.rotation.w = 1;
    
    tf_broadcaster.sendTransform(ref_T_spot);
 
    return true;
}

void Laser3DTracking::cloudClbk(const PointCloud::ConstPtr& msg)
{
    *cloud = *msg;
}

void Laser3DTracking::keypointSubClbk(const tpo_msgs::KeypointImageConstPtr& msg)
{
    keypoint_image = *msg;
}


/**
void Laser3DTracking::cameraInfoClbk(const sensor_msgs::CameraInfoConstPtr& msg) {
    
    cam_info = msg;
    
//     disp2Depth = cv::Mat::zeros(4, 4, CV_64FC1);
//     
//     disp2Depth.at<double>(0,0) = 1;
//     disp2Depth.at<double>(1,1) = 1;
//     
//     disp2Depth.at<double>(0,3) = - msg->K.at(3); //K is stored row-major
//     disp2Depth.at<double>(1,3) = - msg->K.at(5); //K is stored row-major
//     disp2Depth.at<double>(2,3) = - msg->K.at(0); //fx is the same of fy???
    
     //P is stored row-major
//     disp2Depth.at<double>(0,3) = - msg->K.at(2); //cx
//     disp2Depth.at<double>(1,3) = - msg->P.at(5); //cy
//     disp2Depth.at<double>(2,3) = - msg->P.at(0); //fx is the same of fy???
//     
//     disp2Depth.at<double>(2,3) = - 1.0 / (msg->P.at(3)); // -1/Tx
//     disp2Depth.at<double>(2,3) = - 1.0 / (msg->P.at(3)); //(cx - cx') / Tx
    
    camera_info_sub.shutdown();
}
**/



/******************************** *****************************************/

int main ( int argc, char **argv ) {

    ros::init ( argc, argv, "Laser3DTracking" );
    ros::NodeHandle nh("~");
    
    Laser3DTracking laser3DTracking(&nh);
    
    ros::Rate r0(100);
    while(ros::ok()) {
        
        if (laser3DTracking.isReady()) {
            break;
        }

        ros::spinOnce();
        r0.sleep();
    }
    
    
    ros::Rate r(10);
    while(ros::ok()) {
        
        laser3DTracking.run();

        ros::spinOnce();
        r.sleep();
    }
    
    return 0;
    
}
