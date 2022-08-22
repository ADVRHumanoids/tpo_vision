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

#include <tpo_vision/laserSpotDetection.h>

LaserSpotDetection::LaserSpotDetection(ros::NodeHandle* nh)
{
    dynamicParamF = boost::bind(&LaserSpotDetection::dynamicParamClbk, this, _1, _2);
    dynamicParamServer.setCallback(dynamicParamF);
    
    //for detect4
    frames_buffer.set_capacity(queue_size);
    //good values for detect4 in the office room
    //contrast = 14;
    //brightness = -120;
    contrast = 2;
    brightness = 1;
    
    /******************* ROS Stuff ***************/
    this->nh = nh;
    std::string camera_image_topic, camera_image_transport, camera_info_topic, pub_out_keypoint_topic, pub_out_images_topic;
    nh->param<std::string>("camera_image_topic", camera_image_topic, "/D435_head_camera/color/image_raw");
    nh->param<std::string>("transport", camera_image_transport, "compressed");
    nh->param<std::string>("camera_info_topic", camera_info_topic, "/D435_head_camera/color/camera_info");
    
    
    nh->param<std::string>("pub_out_keypoint_topic", pub_out_keypoint_topic, "detection_output_keypoint");

    nh->param<bool>("pub_out_images", pub_out_images, true);
    nh->param<std::string>("pub_out_images_topic", pub_out_images_topic, "detection_output_img/compressed");

    //Draw onpecv image directly inside this class for debugging
    nh->param<bool>("show_images", show_images, false);

    camera_info_sub = nh->subscribe<sensor_msgs::CameraInfo>(camera_info_topic, 1, &LaserSpotDetection::cameraInfoClbk, this);

    color_image_transport = std::make_unique<image_transport::ImageTransport>(*(this->nh));
    //color_image_sub = nh->subscribe<sensor_msgs::CompressedImage>(camera_image_topic, 1, &LaserSpotDetection::colorImageClbk, this);

    color_image_sub = color_image_transport->subscribe(
        camera_image_topic, 1, &LaserSpotDetection::colorImageClbk, this, image_transport::TransportHints(camera_image_transport));
    
    keypoint_pub = nh->advertise<tpo_msgs::KeypointImage>(pub_out_keypoint_topic, 10);

    if (pub_out_images) {
        output_image_pub = nh->advertise<sensor_msgs::CompressedImage>(pub_out_images_topic, 10);
    }
    
}

LaserSpotDetection::~LaserSpotDetection() {
    cv::destroyAllWindows();
}



bool LaserSpotDetection::run() {
    
    double pixel_x;
    double pixel_y;
    double confidence;
    
    if (!detect3(cv_bridge_image.image, pixel_x, pixel_y)) {
        
        return false;
    }
    
    tpo_msgs::KeypointImage msg;
    msg.header.frame_id = ros_image_input.header.frame_id;
    msg.header.seq = ros_image_input.header.seq;
    msg.header.stamp = ros::Time::now();
    
    msg.x_pixel = pixel_x;
    msg.y_pixel = pixel_y;
    msg.label = 1;
    msg.confidence = confidence;
    
    keypoint_pub.publish(msg);
    
    if (pub_out_images) {
        
        sensor_msgs::Image msg;
        cv_bridge_output_image.image = cv_bridge_image.image;
        cv::circle(cv_bridge_output_image.image, cv::Point(pixel_x, pixel_y), 10, cv::Scalar(0,0,255), 3);
        output_image_pub.publish(cv_bridge_output_image.toCompressedImageMsg());
        
    }
    
    return true; 
}

bool LaserSpotDetection::isReady() {
    
    if (cam_info == nullptr) {
        ROS_WARN_STREAM_ONCE("Camera info not yet arrived...");
        return false;
    } else {
        ROS_WARN_STREAM_ONCE("... Camera info arrived");
    }
    
    if (cv_bridge_image.image.empty()) {
        ROS_WARN_STREAM_ONCE("Image not yet arrived");
        return false;
    } else {
        ROS_WARN_STREAM_ONCE("... Image arrived");
    }  
    
    return true;
}

void LaserSpotDetection::cameraInfoClbk(const sensor_msgs::CameraInfoConstPtr& msg) {
    
    cam_info = msg;
    
    camera_info_sub.shutdown();
}

void LaserSpotDetection::colorImageClbk(const sensor_msgs::ImageConstPtr& msg) {
    
    try
    {
        //IDK why I have to put BGR8 even if everything is RGB otherwise wrong colors
        ros_image_input = *msg;
        cv_bridge_image = *(cv_bridge::toCvShare(msg, sensor_msgs::image_encodings::BGR8));
    }
    catch (cv_bridge::Exception& e)
    {
        ROS_ERROR("cv_bridge exception: %s", e.what());
        return;
    }
    
//     std::cout << msg->encoding << std::endl;
//     std::cout << cv_bridge_image.encoding << std::endl;
//     cv_bridge::CvImage cv_bridge_image_test;
//     cv_bridge_image_test = *(cv_bridge::toCvShare(msg, sensor_msgs::image_encodings::BGR8));
//     std::cout << cv_bridge_image_test.encoding << std::endl;

    
}


/**
 * WITH Background subtraction 
 */
bool LaserSpotDetection::detect4(const cv::Mat &frame, double &pixel_x, double &pixel_y)
{
    bool ret = true;
    
    cv::Mat frame_filtered;
    cv::Mat hsv_img, hv_thresh, hv_thresh_blur;
    cv::Mat hsv_channels[3];
    cv::Mat hsv_channels_thresh[3];
    
    cv::Mat converted;
    frame.convertTo(converted, CV_32FC3);
    frames_buffer.push_back(converted);
    
    frames_buffer_sum = cv::Mat::zeros(frame.size(), CV_32FC3);
    for (const auto & fr : frames_buffer) {
        frames_buffer_sum += fr;
    }
    cv::Mat mean = frames_buffer_sum / frames_buffer.size();
    cv::Mat mean_conv;
    mean.convertTo(mean_conv, frame.type());
    
    frame_filtered = frame - mean_conv;
    
    //contrast 2.2 and bright 0 seems good
    frame_filtered = contrast * frame_filtered + brightness;
    
   //TODO complete this
   // - some logic to to "detect" camera and/or enviornment movments and wait for background stabilization?
    
    if (encoding.compare("rgb8") == 0 || encoding.compare("rgb16") == 0) {
        cv::cvtColor(frame_filtered, hsv_img, cv::COLOR_RGB2HSV);

    } else if (encoding.compare("bgr8") == 0 || encoding.compare("bgr16") == 0){
        cv::cvtColor(frame_filtered, hsv_img, cv::COLOR_BGR2HSV);
        
    } else {
        ROS_ERROR_STREAM("encoding '"<< encoding << "' not supported"); 
        return false;
    }
    
    //split hsv
    cv::split(hsv_img, hsv_channels);

    auto params = cv::SimpleBlobDetector::Params();
    
    params.filterByColor = false;
    //params.blobColor = blob_param_color;
    
    params.filterByArea = true;
    params.minArea = 10;
    params.maxArea = 25;
    
    params.filterByCircularity = false;
    params.minCircularity = 0.5;
    
    params.filterByInertia = false;
    params.filterByConvexity = false;

    auto detector = cv::SimpleBlobDetector::create(params);
    
    std::vector<cv::KeyPoint> keypoints;
    detector->detect(hsv_channels[2], keypoints);

    if (keypoints.size() == 0) {
        
        ROS_ERROR_STREAM_THROTTLE (5, "no keypoints found!");
        ret = false;
        
    } else {
        pixel_x = keypoints.at(0).pt.x;
        pixel_y = keypoints.at(0).pt.y;
        
        std::cout << keypoints.at(0).size << std::endl;
        
    }
    
    if (keypoints.size() > 1) {
        ROS_WARN_STREAM("more than one keypoint found, I am returing the first one");
        //ret = false;
    }
            
    if (show_images) {
        cv::imshow("original", frame);
        cv::imshow("after blurring", frame_filtered);
        cv::imshow("hsv", hsv_img);
        cv::imshow("h", hsv_channels[0]);
        cv::imshow("s", hsv_channels[1]);
        cv::imshow("v", hsv_channels[2]);
        //cv::imshow("and of hue and value after respective thresholds", hv_thresh);
        //cv::imshow("hv_thresh_blur", hv_thresh_blur);
        
        cv::Mat im_with_keypoints;
        cv::drawKeypoints(frame, keypoints, im_with_keypoints, cv::Scalar(0,255,0), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        cv::imshow("im_with_keypoints", im_with_keypoints);
   
        cv::waitKey(1);
    }
    
    return ret;
    
}


/**
 * hsv separation, their thresholding, some filterings, and final blob-circle detection on sum of hue and value channels
 * Not so bad but not so robust to different surfaces? ie each different surface need different threshold params ? 
 */
bool LaserSpotDetection::detect3(const cv::Mat &frame, double &pixel_x, double &pixel_y)
{
    bool ret = true;
    

    cv::Mat frame_filtered;
    cv::Mat hsv_img, hv_thresh, hv_thresh_blur;
    cv::Mat hsv_channels[3];
    cv::Mat hsv_channels_thresh[3];
    
    frame_filtered = contrast * frame + brightness;
    
    if (median_blur_size != 0) {

        if (median_blur_size%2 == 0) {
            median_blur_size++;
        }
        cv::GaussianBlur(frame_filtered,frame_filtered,cv::Size(median_blur_size, median_blur_size), 0);
   }
    
    
    if (encoding.compare("rgb8") == 0 || encoding.compare("rgb16") == 0) {
        cv::cvtColor(frame_filtered, hsv_img, cv::COLOR_RGB2HSV);

    } else if (encoding.compare("bgr8") == 0 || encoding.compare("bgr16") == 0){
        cv::cvtColor(frame_filtered, hsv_img, cv::COLOR_BGR2HSV);
        
    } else {
        ROS_ERROR_STREAM("encoding '"<< encoding << "' not supported"); 
        return false;
    }
    
    //split hsv
    cv::split(hsv_img, hsv_channels);
    
    if (hue_min > hue_max) {
        
        cv::Mat h_tmp, h_tmp2;
        cv::inRange(hsv_channels[0], hue_min, 255, h_tmp);
        cv::inRange(hsv_channels[0], 0, hue_max, h_tmp2);
        cv::bitwise_or(h_tmp, h_tmp2, hsv_channels_thresh[0]);
        
    } else {
        cv::inRange(hsv_channels[0], hue_min, hue_max, hsv_channels_thresh[0]);

    }
    
    cv::inRange(hsv_channels[1], sat_min, sat_max, hsv_channels_thresh[1]);
    cv::inRange(hsv_channels[2], val_min, val_max, hsv_channels_thresh[2]);
    
    cv::bitwise_and(hsv_channels_thresh[0], hsv_channels_thresh[2], hv_thresh);
    


    auto params = cv::SimpleBlobDetector::Params();
    
    params.filterByColor = false;
    //params.blobColor = blob_param_color;
    
    params.filterByArea = true;
    params.minArea = 10;
    params.maxArea = 25;
    
    params.filterByCircularity = false;
    params.minCircularity = 0.5;
    
    params.filterByInertia = false;
    params.filterByConvexity = false;

    auto detector = cv::SimpleBlobDetector::create(params);
    
    std::vector<cv::KeyPoint> keypoints;
    detector->detect(hv_thresh, keypoints);

    if (keypoints.size() == 0) {
        
        ROS_ERROR_STREAM_THROTTLE (5, "no keypoints found!");
        ret = false;
        
    } else {
        pixel_x = keypoints.at(0).pt.x;
        pixel_y = keypoints.at(0).pt.y;
        
        std::cout << keypoints.at(0).size << std::endl;
        
    }
    
    if (keypoints.size() > 1) {
        ROS_WARN_STREAM("more than one keypoint found, I am returing the first one");
        //ret = false;
    }
            
    if (show_images) {
        cv::imshow("original", frame);
        cv::imshow("after blurring", frame_filtered);
        //cv::imshow("hsv", hsv_img);
        cv::imshow("h_th", hsv_channels_thresh[0]);
        cv::imshow("s_th", hsv_channels_thresh[1]);
        cv::imshow("v_th", hsv_channels_thresh[2]);
        cv::imshow("and of hue and value after respective thresholds", hv_thresh);
        //cv::imshow("hv_thresh_blur", hv_thresh_blur);
        
        cv::Mat im_with_keypoints;
        cv::drawKeypoints(frame, keypoints, im_with_keypoints, cv::Scalar(0,255,0), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        cv::imshow("im_with_keypoints", im_with_keypoints);
   
        cv::waitKey(1);
    }
    
    return ret;
    
}

/************************************************************************************************ */
bool LaserSpotDetection::detect0(const cv::Mat &frame, double &pixel_x, double &pixel_y)
{
    bool ret = true;
    
    cv::Mat hsv_img, mask, hsv_img_thresholded, hsv_img_thresholded_gray, hsv_img_thresholded_last;

    if (encoding.compare("rgb8") == 0 || encoding.compare("rgb16") == 0) {
        cv::cvtColor(frame, hsv_img, cv::COLOR_RGB2HSV);

    } else if (encoding.compare("bgr8") == 0 || encoding.compare("bgr16") == 0){
        cv::cvtColor(frame, hsv_img, cv::COLOR_BGR2HSV);
        
    } else {
        ROS_ERROR_STREAM("encoding '"<< encoding << "' not supported"); 
        return false;
    }
    

    //Threshold colors all united?
    cv::inRange(hsv_img, cv::Scalar(hue_min, sat_min, val_min), cv::Scalar(hue_max, sat_max, val_max), mask);
    cv::bitwise_and(frame, frame, hsv_img_thresholded, mask);
    
    //to gray
    cv::cvtColor(hsv_img_thresholded, hsv_img_thresholded_gray, cv::COLOR_HSV2BGR);
    cv::cvtColor(hsv_img_thresholded_gray, hsv_img_thresholded_gray, cv::COLOR_BGR2GRAY);
    
    cv::threshold(hsv_img_thresholded_gray, hsv_img_thresholded_last, thresh_gray_min, thresh_gray_max, cv::THRESH_BINARY);

    auto params = cv::SimpleBlobDetector::Params();
    
    params.filterByColor = true;
    params.blobColor = blob_param_color;
    
    params.filterByArea = false;
    //params.minArea = 3;
    //params.maxArea = 10;
    
    params.filterByCircularity = false;
    //params.minCircularity = 0.5;
    
    params.filterByInertia = false;
    params.filterByConvexity = false;

    auto detector = cv::SimpleBlobDetector::create(params);
    
    std::vector<cv::KeyPoint> keypoints;
    detector->detect(hsv_img_thresholded_last, keypoints);

    if (keypoints.size() == 0) {
        
        ROS_ERROR_STREAM_THROTTLE (5, "no keypoints found!");
        ret = false;
        
    } else {
        pixel_x = keypoints.at(0).pt.x;
        pixel_y = keypoints.at(0).pt.y;
    }
    
    if (keypoints.size() > 1) {
        ROS_WARN_STREAM("more than one keypoint found, I am returing the first one");
        //ret = false;
    }
            
    if (show_images) {
        cv::imshow("mask for bitwise and", mask);
        cv::imshow("after bitwise and", hsv_img_thresholded);
        
        cv::Mat im_with_keypoints;
        cv::drawKeypoints(hsv_img_thresholded_last, keypoints, im_with_keypoints, cv::Scalar(0,255,0), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        cv::imshow("im_with_keypoints", im_with_keypoints);
   
        cv::waitKey(1);
    }
    
    return ret;
    
}

bool LaserSpotDetection::detect2(const cv::Mat &frame, double &pixel_x, double &pixel_y)
{
    bool ret = true;
    
    cv::Mat hsv_img, mask;
    if (encoding.compare("rgb8") == 0 || encoding.compare("rgb16") == 0) {
        cv::cvtColor(frame, hsv_img, cv::COLOR_RGB2HSV);

    } else if (encoding.compare("bgr8") == 0 || encoding.compare("bgr16") == 0){
        cv::cvtColor(frame, hsv_img, cv::COLOR_BGR2HSV);
        
    } else {
        ROS_ERROR_STREAM("encoding '"<< encoding << "' not supported"); 
        return false;
    }
    
    cv::inRange(hsv_img, cv::Scalar(hue_min, sat_min, val_min), cv::Scalar(hue_max, sat_max, val_max), mask);
    
    double minVal, maxVal;
    cv::Point minLoc, maxLoc;
    cv::minMaxLoc(mask, &minVal, &maxVal, &minLoc, &maxLoc);

    pixel_x = maxLoc.x;
    pixel_y = maxLoc.y;
            
    if (show_images) {
        cv::imshow("original" , frame);
        cv::imshow("hsv" , hsv_img);
        cv::imshow("mask after color inRange", mask);
        
        cv::Mat im_with_keypoints = frame;
        cv::circle(im_with_keypoints, maxLoc, 10, cv::Scalar(0,255,0), cv::LINE_AA);
        cv::imshow("im_with_keypoints", im_with_keypoints);
   
        cv::waitKey(1);
    }
    
    return ret;
    
}

void LaserSpotDetection::dynamicParamClbk(tpo_vision::LaserSpotDetectionParamsConfig &config, uint32_t level)
{
    ROS_INFO("Reconfigure Request: hue: %d %d; sat: %d %d; val: %d %d; t_gray: %d %d; blob_col: %d", 
              config.hue_min, config.hue_max, config.sat_min, config.sat_max, config.val_min, config.val_max,
              config.thresh_gray_min, config.thresh_gray_max, config.blob_param_color
    );
 
    hue_min = config.hue_min;
    hue_max = config.hue_max;
    sat_min = config.sat_min;
    sat_max = config.sat_max;
    val_min = config.val_min;
    val_max = config.val_max;
    thresh_gray_min = config.thresh_gray_min;
    thresh_gray_max = config.thresh_gray_max;
    blob_param_color = config.blob_param_color;
    
    median_blur_size = config.median_blur_ksize;
    contrast = config.contrast;
    brightness = config.brightness;
  
}

int main ( int argc, char **argv ) {

    ros::init ( argc, argv, "LaserSpotDetection" );
    ros::NodeHandle nh("~");
    
    LaserSpotDetection laserSpotDetection(&nh);
    
    ros::Rate r0(100);
    while(ros::ok()) {
        
        if (laserSpotDetection.isReady()) {
            break;
        }

        ros::spinOnce();
        r0.sleep();
    }
    
    
    ros::Rate r(10);
    while(ros::ok()) {
        
        laserSpotDetection.run();

        ros::spinOnce();
        r.sleep();
    }
    
    return 0;
    
}
