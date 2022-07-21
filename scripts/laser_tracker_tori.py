#!/usr/bin/env python
### FROM https://github.com/bradmontgomery/python-laser-tracker/blob/master/laser_tracker/laser_tracker.py

import cv2
import numpy
import sys

class LaserTrackerTori(object):

    def __init__(self, cam_width=640, cam_height=480, hue_min=0, hue_max=10,
                 sat_min=190, sat_max=255, val_min=190, val_max=255,
                 display_thresholds=False):
        """
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
        """

        self.cam_width = cam_width
        self.cam_height = cam_height
        self.hue_min = hue_min
        self.hue_max = hue_max
        self.sat_min = sat_min
        self.sat_max = sat_max
        self.val_min = val_min
        self.val_max = val_max
        self.display_thresholds = display_thresholds

        self.capture = None  # camera capture device
        self.channels = {
            'hue': None,
            'saturation': None,
            'value': None,
            'laser': None,
        }

        self.previous_position = None
        self.trail = numpy.zeros((self.cam_height, self.cam_width, 3),
                                 numpy.uint8)

    def track(self, frame, mask):
        """
        Track the position of the laser pointer.
        Code taken from
        http://www.pyimagesearch.com/2015/09/14/ball-tracking-with-opencv/
        """
        center = None

        countours = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                     cv2.CHAIN_APPROX_SIMPLE)[-2]

        # only proceed if at least one contour was found
        if len(countours) > 0:
            # find the largest contour in the mask, then use
            # it to compute the minimum enclosing circle and
            # centroid
            c = max(countours, key=cv2.contourArea)
            ((x, y), radius) = cv2.minEnclosingCircle(c)
            moments = cv2.moments(c)
            if moments["m00"] > 0:
                center = int(moments["m10"] / moments["m00"]), \
                         int(moments["m01"] / moments["m00"])
            else:
                center = int(x), int(y)

            # only proceed if the radius meets a minimum size
            if radius > 10:
                # draw the circle and centroid on the frame,
                cv2.circle(frame, (int(x), int(y)), int(radius),
                           (0, 255, 255), 2)
                cv2.circle(frame, center, 5, (0, 0, 255), -1)
                # then update the ponter trail
                if self.previous_position:
                    cv2.line(self.trail, self.previous_position, center,
                             (255, 255, 255), 2)
        print(numpy.shape(frame))
        print(numpy.shape(self.trail))
        cv2.add(self.trail, frame, frame)
        self.previous_position = center

    def detect(self, frame):
        
        hsv_img = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        #Threshold colors
        lower = numpy.array([self.hue_min, self.sat_min, self.val_min])
        upper = numpy.array([self.hue_max, self.sat_max, self.val_max])
        mask = cv2.inRange(hsv_img, lower, upper)
        hsv_img_thresholded = cv2.bitwise_and(frame, frame, mask=mask)

        # split the video frame into color channels
        h, s, v = cv2.split(hsv_img_thresholded)
        self.channels['hue'] = h
        self.channels['saturation'] = s
        self.channels['value'] = v

        # Perform an AND on HSV components to identify the laser!
        self.channels['laser'] = cv2.bitwise_and(
            self.channels['hue'],
            self.channels['value']
        )
        
        cv2.imshow('RGB_VideoFrame', frame)
        cv2.imshow('hsv_img_thresholded', hsv_img_thresholded)
        hsv_img_thresholded2 = cv2.cvtColor(hsv_img_thresholded, cv2.COLOR_HSV2BGR)
        hsv_img_thresholded2 = cv2.cvtColor(hsv_img_thresholded, cv2.COLOR_BGR2GRAY)
        cv2.imshow('hsv_img_thresholded2', hsv_img_thresholded2)
        ret,hsv_img_thresholded3 = cv2.threshold(hsv_img_thresholded2,100,255,cv2.THRESH_BINARY)
        cv2.imshow('hsv_img_thresholded3', hsv_img_thresholded3)


        cv2.imshow('h', self.channels['hue'])
        cv2.imshow('s', self.channels['saturation'])
        cv2.imshow('v', self.channels['value'])
        
        
        params = cv2.SimpleBlobDetector_Params()
        
        params.filterByColor = True
        params.blobColor = 255
        
        params.filterByArea = False
        params.minArea = 3
        params.maxArea = 10
        
        params.filterByCircularity = False
        params.minCircularity = 0.5
        
        params.filterByInertia = False
        params.filterByConvexity = False


        detector = cv2.SimpleBlobDetector_create(params)
        keypoints = detector.detect(hsv_img_thresholded3)
        im_with_keypoints = cv2.drawKeypoints(hsv_img_thresholded3, keypoints, numpy.array([]), (0,255,0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        cv2.imshow('im_with_keypoints', im_with_keypoints)
        cv2.waitKey(3)
        i = 0
        for k in keypoints:
            print(f'point {i}: {k.pt[0]}; {k.pt[1]}')
        
        #self.track(frame, self.channels['laser'])

        return hsv_img







    def display(self, img, frame):
        """Display the combined image and (optionally) all other image channels
        NOTE: default color space in OpenCV is BGR.
        """
        cv2.imshow('RGB_VideoFrame', frame)
        cv2.imshow('LaserPointer', self.channels['laser'])
        if self.display_thresholds:
            cv2.imshow('Thresholded_HSV_Image', img)
            cv2.imshow('Hue', self.channels['hue'])
            cv2.imshow('Saturation', self.channels['saturation'])
            cv2.imshow('Value', self.channels['value'])

    def create_and_position_window(self, name, xpos, ypos):
        """Creates a named widow placing it on the screen at (xpos, ypos)."""
        # Create a window
        cv2.namedWindow(name)
        # Resize it to the size of the camera image
        cv2.resizeWindow(name, self.cam_width, self.cam_height)
        # Move to (xpos,ypos) on the screen
        cv2.moveWindow(name, xpos, ypos)
        
    def setup_windows(self):
        sys.stdout.write("Using OpenCV version: {0}\n".format(cv2.__version__))

        # create output windows
        self.create_and_position_window('LaserPointer', 0, 0)
        self.create_and_position_window('RGB_VideoFrame',
                                        10 + self.cam_width, 0)
        if self.display_thresholds:
            self.create_and_position_window('Thresholded_HSV_Image', 10, 10)
            self.create_and_position_window('Hue', 20, 20)
            self.create_and_position_window('Saturation', 30, 30)
            self.create_and_position_window('Value', 40, 40)


