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

Laser3DTracking::Laser3DTracking (ros::NodeHandle* nh, const double& period)
{
    
    this->nh = nh;
    this->period = period;
    
    //tf_listener = std::make_unique<tf2_ros::TransformListener>(tf_buffer);
    
    std::string pc_topic, image_topic, camera_info_topic, image_transport;
    nh->param<std::string>("point_cloud_topic", pc_topic, "/D435_head_camera/depth/color/points");
    
    //nh->param<std::string>("ref_frame", ref_frame, "pelvis");
    nh->param<std::string>("laser_spot_frame", laser_spot_frame, "laser_spot");
    nh->param<std::string>("camera_frame", camera_frame, "D435_head_camera_color_optical_frame");
    
    nh->param<double>("detection_confidence_threshold", detection_confidence_threshold, 0.2);
    nh->param<double>("cloud_detection_max_sec_diff", cloud_detection_max_sec_diff, 1);

    std::string keypoint_topic;
    nh->param<std::string>("keypoint_topic", keypoint_topic, "detection_output_keypoint");
    
    keypoint_sub = nh->subscribe<tpo_msgs::KeypointImage>(keypoint_topic, 10, &Laser3DTracking::keypointSubClbk, this);
     
    /******************* CLOUD ***************************/
    cloud_sub = nh->subscribe<PointCloud>(pc_topic, 1, &Laser3DTracking::cloudClbk, this);
    cloud = boost::make_shared<PointCloud>();
    
    /*************** ROS PCL FILTER **************/    
//     if( ros::console::set_logger_level(ROSCONSOLE_DEFAULT_NAME, ros::console::levels::Debug) ) {
//         ros::console::notifyLoggerLevelsChanged();
//     }
    
    nh->param<bool>("pcl_filter", pcl_filter, true);
    nh->param<bool>("pub_pcl_filtered", pub_pcl_filtered, false);
    if (pcl_filter) {
        
        _filter_chain = std::make_unique<filters::FilterChain<sensor_msgs::PointCloud2>>("sensor_msgs::PointCloud2");
        if (!_filter_chain->configure("/cloud_filter_chain"))
        {
            ROS_ERROR_STREAM("Configuration of filter chain for is invalid, the chain will not be run.");
            throw std::runtime_error("Filter configuration error");
        }
        if (pub_pcl_filtered) {
            _filtered_pc_pub = nh->advertise<PointCloud>("cloud_filtered", 1);
            ROS_INFO("Using PCL filter and publishing the filtered cloud");

        } else {
            ROS_INFO("Using PCL filter");
        }
        
    } else {
        ROS_INFO("Not using PCL filter");
    }

    
    ref_T_spot.resize(2);
    //ref_T_spot.at(0).header.frame_id = ref_frame;
    ref_T_spot.at(0).child_frame_id = laser_spot_frame;
    ref_T_spot.at(0).transform.rotation.w = 1;

    //ref_T_spot.at(1).header.frame_id = ref_frame;
    ref_T_spot.at(1).child_frame_id = laser_spot_frame + "_raw";
    ref_T_spot.at(1).transform.rotation.w = 1;

    
    /************************************************ FILTER  ***************************/
    
    nh->param<double>("damping", _filter_damping, 1);
    nh->param<double>("bw", _filter_bw, 9);

    _laser_pos_filter = std::make_shared<tpo::utils::FilterWrap<Eigen::Vector3d>>(_filter_damping, _filter_bw, period, 3);
        
    _ddr_server = std::make_unique<ddynamic_reconfigure::DDynamicReconfigure>(*nh);
    _ddr_server->registerVariable<double>("damping", _filter_damping, boost::bind(&Laser3DTracking::ddr_callback_filter_damping, this, _1), "damping", (double)0, (double)10, "laser_filter");
    _ddr_server->registerVariable<double>("bw", _filter_bw, boost::bind(&Laser3DTracking::ddr_callback_filter_bw, this, _1), "bw", (double)0, (double)50, "laser_filter");
    _ddr_server->registerVariable<double>("detection_confidence_threshold", &detection_confidence_threshold, "Under this confidence (coming from 2d pixel image) the point is considered invalid, and no relative tf is published", 0, 1, "detection");
    _ddr_server->registerVariable<double>("cloud_detection_max_sec_diff", &cloud_detection_max_sec_diff, "If point cloud and detection keypoint have a timestamp with difference bigger than this value, no relative tf is published", 0, 10, "detection");
    _ddr_server->publishServicesTopics();
    
}

bool Laser3DTracking::filterCloud() { 
    
    //template<typename T> void toROSMsg(const pcl::PointCloud<T> &pcl_cloud, sensor_msgs::PointCloud2 &cloud)
    pcl::toROSMsg<pcl::PointXYZ>(*cloud, ros_pc);
    
    if (!_filter_chain->update(ros_pc, ros_pc)) {
        ROS_ERROR("Filtering cloud failed."); 
        return false;
    }
    
    pcl::fromROSMsg<pcl::PointXYZ>(ros_pc, *cloud);

    if(pub_pcl_filtered) {
        _filtered_pc_pub.publish(cloud);
    }
    
    return true;
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

    if (pcl_filter) {
        //auto tick = ros::Time::now();
        if (! filterCloud()) {
            return -1;
        }
        //ROS_INFO("filter cloud time: %f", (double)(ros::Time::now() - tick).toSec() );
    }
    
    if (! sendTransformFrom2D()) {
        return -1;
    }

    return 0;
}

bool Laser3DTracking::sendTransformFrom2D() {
    
    {
    const std::lock_guard<std::mutex> lock(cloud_mutex);

    //header in pcl cloud is a uint in microsecond, not a ros::Time
    ros::Time cloud_time;
    cloud_time.fromNSec(cloud->header.stamp*1000);
    //ROS_INFO("cloud time %s;\timage time %s", 
    //        std::to_string(cloud_time.toSec()).c_str(),
    //        std::to_string(keypoint_image.header.stamp.toSec()).c_str());
    
    
    if (keypoint_image.confidence <= detection_confidence_threshold ){
    
        //ROS_WARN("Confidence of arrived keypoint detection message is below the threshold (%s < %s)", 
        //         std::to_string(keypoint_image.confidence).c_str(), std::to_string(detection_confidence_threshold).c_str());
        return false;
    }
    
    ros::Duration time_diff;
    
//     std::cout << "aaaaaaaaaaaaaa" << std::endl;
//     std::cout << " ros: " << cloud_time.toNSec() << std::endl;
//     std::cout << " pcl: " << cloud->header.stamp * 1000 << std::endl;
//     std::cout << "aaaaaaaaaaaaaa" << std::endl;

    if (keypoint_image.header.stamp < cloud_time) {
        
        time_diff = cloud_time - keypoint_image.header.stamp;
        
        if (time_diff.toSec() > cloud_detection_max_sec_diff) {
            ROS_WARN("keypoint is too old wrt to cloud: (%ss, %ss, diff : %ss > %ss)", 
                    std::to_string(keypoint_image.header.stamp.toSec()).c_str(), 
                    std::to_string(cloud_time.toSec()).c_str(),
                    std::to_string(time_diff.toSec()).c_str(),
                    std::to_string(cloud_detection_max_sec_diff).c_str()
                    );    
            
            return false;
        } 
        
    } else {
        
        time_diff = keypoint_image.header.stamp - cloud_time; 
        
        if (time_diff.toSec() > cloud_detection_max_sec_diff) {
            ROS_WARN("cloud is too old wrt to keypoint (this is strange): (%ss, %ss, diff : %ss > %ss)", 
                    std::to_string(cloud_time.toSec()).c_str(), 
                    std::to_string(keypoint_image.header.stamp.toSec()).c_str(),
                    std::to_string(time_diff.toSec()).c_str(),
                    std::to_string(cloud_detection_max_sec_diff).c_str()
                    );    

            return false;
        } 
    }
    
    //std::cout << "difffff " << time_diff.toSec() << std::endl;
    
    if (!updateTransform()){
        return false;
    }
    
    } //const std::lock_guard<std::mutex> lock(cloud_mutex);
    
    tf_broadcaster.sendTransform(ref_T_spot);
 
    return true;
}

bool Laser3DTracking::updateTransform ()
{
    auto pointXYZ = cloud->at(keypoint_image.x_pixel, keypoint_image.y_pixel);
    
    if (pointXYZ.z == 0 || std::isnan(pointXYZ.x) || std::isnan(pointXYZ.y) || std::isnan(pointXYZ.z) ) {
    
//         ROS_ERROR("Z distance is very small, do not ignore this: x:%f, y:%f, z:%f", pointXYZ.x, pointXYZ.y, pointXYZ.z);
//         ROS_ERROR("x_pix:%d, y_pix%d", keypoint_image.x_pixel, keypoint_image.y_pixel);
//         ROS_ERROR("cloud index from pixels:%d\n", keypoint_image.x_pixel * cloud->width + keypoint_image.y_pixel);
        ROS_INFO("pixel has no corresponding pc (probably there is a hole, or it is a filtered out part of the robot), dropping it");
        return false;
    }
    std::cout << pointXYZ.x << " " << pointXYZ.y << " " << pointXYZ.z << std::endl;
    
    ref_T_spot.at(0).header.stamp = ros::Time::now();
    ref_T_spot.at(1).header.stamp = ref_T_spot.at(0).header.stamp;
    
    ref_T_spot.at(0).header.frame_id = cloud->header.frame_id;
    ref_T_spot.at(1).header.frame_id = cloud->header.frame_id;
    
    Eigen::Vector3d vec, vec_filt;
    vec << pointXYZ.x, pointXYZ.y, pointXYZ.z;
    vec_filt = _laser_pos_filter->process(vec);
    
    ref_T_spot.at(0).transform.translation.x = vec_filt(0);
    ref_T_spot.at(0).transform.translation.y = vec_filt(1);
    ref_T_spot.at(0).transform.translation.z = vec_filt(2);
    ref_T_spot.at(1).transform.translation.x = pointXYZ.x;
    ref_T_spot.at(1).transform.translation.y = pointXYZ.y;
    ref_T_spot.at(1).transform.translation.z = pointXYZ.z;
    

    
    
    return true;
}

void Laser3DTracking::cloudClbk(const PointCloud::ConstPtr& msg)
{
    const std::lock_guard<std::mutex> lock(cloud_mutex);
    *cloud = *msg;
    //std::cout << cloud->header.frame_id << std::endl;
    //change reference frame
    //pcl_ros::transformPointCloud (ref_frame, *cloud, *cloud, tf_buffer);
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

void Laser3DTracking::ddr_callback_filter_damping(double new_value) {
    
    _filter_damping = new_value;
    _laser_pos_filter->reset(_filter_damping, _filter_bw);
}
void Laser3DTracking::ddr_callback_filter_bw(double new_value) {
    
    _filter_bw = new_value;
    _laser_pos_filter->reset(_filter_damping, _filter_bw);

}


/******************************** *****************************************/

int main ( int argc, char **argv ) {

    ros::init ( argc, argv, "Laser3DTracking" );
    ros::NodeHandle nh("~");
    
    double rate;
    nh.param<double>("rate", rate, 100);
    
    Laser3DTracking laser3DTracking(&nh, 1.0/rate);
    
    ros::Rate r0(100);
    while(ros::ok()) {
        
        if (laser3DTracking.isReady()) {
            break;
        }

        ros::spinOnce();
        r0.sleep();
    }
    
    
    ros::Rate r(rate);
    while(ros::ok()) {
        
        laser3DTracking.run();

        ros::spinOnce();
        r.sleep();
    }
    
    return 0;
    
}
