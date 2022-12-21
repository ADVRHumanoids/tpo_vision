/*
 * Copyright 2022 Davide Torielli
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

#include <tpo_vision/ExtractSinglePointFrom2D.h>

ExtractSinglePointFrom2D::ExtractSinglePointFrom2D(ros::NodeHandle* nh) {
    
    this->nh = nh;
    
    cloud_sub = nh->subscribe<PointCloud>("/D435_head_camera/depth/color/points", 1, &ExtractSinglePointFrom2D::cloudClbk, this);
    cloud = boost::make_shared<PointCloud>();
    
    camera_info_sub = nh->subscribe<sensor_msgs::CameraInfo>("/D435_head_camera/aligned_depth_to_color/camera_info", 1, &ExtractSinglePointFrom2D::cameraInfoClbk, this);
 
    depth_image_sub = nh->subscribe<sensor_msgs::Image>("/D435_head_camera/aligned_depth_to_color/image_raw", 1, &ExtractSinglePointFrom2D::depthImageClbk, this);
    
}

void ExtractSinglePointFrom2D::cameraInfoClbk(const sensor_msgs::CameraInfoConstPtr& msg) {
    
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

void ExtractSinglePointFrom2D::depthImageClbk(const sensor_msgs::ImageConstPtr& msg) {
    
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

bool ExtractSinglePointFrom2D::run(const int pixel_x, const int pixel_y) {
    
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
    
    t.transform.translation.x = cloud->at(pixel_x, pixel_y).x;
    t.transform.translation.y = cloud->at(pixel_x, pixel_y).y;
    t.transform.translation.z = cloud->at(pixel_x, pixel_y).z;
    
    t.transform.rotation.w = 1;
    
    tf_broadcaster.sendTransform(t);
 
    return true;
}

void ExtractSinglePointFrom2D::cloudClbk(const PointCloud::ConstPtr& msg)
{
    *cloud = *msg;
}


/**************************************************** */
int main ( int argc, char **argv ) {

    ros::init ( argc, argv, "ExtractSinglePointFrom2D" );
    ros::NodeHandle nh("~");
    
    ExtractSinglePointFrom2D extractSinglePointFrom2D(&nh);
    
    ros::Rate r(10);
    while(ros::ok()) {
        
        extractSinglePointFrom2D.run(651, 217);

        ros::spinOnce();
        r.sleep();
    }
    
    return 0;
    
}
