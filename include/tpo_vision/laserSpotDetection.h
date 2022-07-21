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

#ifndef LASERSPOTDETECTION_H
#define LASERSPOTDETECTION_H

#include <ros/ros.h>
#include <dynamic_reconfigure/server.h>
#include <tpo_vision/LaserSpotDetectionParamsConfig.h>

#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>

#include <boost/circular_buffer.hpp>



class LaserSpotDetection
{
    
public:
    /**
    * ``cam_width`` x ``cam_height`` -- This should be the size of the
    image coming from the camera. Default is 640x480.
    HSV color space Threshold values for a RED laser pointer are determined
    by:
    * ``hue_min``, ``hue_max`` -- Min/Max allowed Hue values
    * ``sat_min``, ``sat_max`` -- Min/Max allowed Saturation values
    * ``val_min``, ``val_max`` -- Min/Max allowed pixel values
    If the dot from the laser pointer doesn't fall within these values, it
    will be ignored.
    * ``display_thresholds`` -- if True, additional windows will display
        values for threshold image channels.
     */
    LaserSpotDetection(unsigned int cam_width = 1280, unsigned int cam_height=720, std::string encoding="rgb8", bool show_images = false);
    
    ~LaserSpotDetection();
    
    unsigned int cam_width;
    unsigned int cam_height;
    std::string encoding;
    bool show_images;
    
    bool detect(const cv::Mat &frame, double &pixel_x, double &pixel_y); 

    
private:  
    /*********************************** */
    bool detect0(const cv::Mat &frame, double &pixel_x, double &pixel_y);

    unsigned int hue_min=0; unsigned int hue_max=10;
    unsigned int sat_min=190; unsigned int sat_max=255;
    unsigned int val_min=190; unsigned int val_max=255;
    
    unsigned int thresh_gray_min = 100;
    unsigned int thresh_gray_max = 255;
    unsigned int blob_param_color = 255;
    
    
    bool detect3(const cv::Mat &frame, double &pixel_x, double &pixel_y); 
    unsigned int median_blur_size = 3;
    double contrast = 1;
    double brightness = 0;
    
    bool detect4(const cv::Mat &frame, double &pixel_x, double &pixel_y); 
    cv::Mat queue_frame;
    unsigned queue_size = 10;
    boost::circular_buffer<cv::Mat> frames_buffer;

    
    /*********************************** */
    bool detect2(const cv::Mat &frame, double &pixel_x, double &pixel_y);


    dynamic_reconfigure::Server<tpo_vision::LaserSpotDetectionParamsConfig> dynamicParamServer;
    void dynamicParamClbk(tpo_vision::LaserSpotDetectionParamsConfig &config, uint32_t level);
    dynamic_reconfigure::Server<tpo_vision::LaserSpotDetectionParamsConfig>::CallbackType dynamicParamF;




    


};

#endif // LASERSPOTDETECTION_H
