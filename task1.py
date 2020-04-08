#!/usr/bin/python

import sys
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

def remove_barrel(frame):
    width = frame.shape[1]
    height = frame.shape[0]

    distCoeff = np.zeros((4, 1), np.float64)

    # TODO: add your coefficients here!
    k1 = -3.0e-5  # negative to remove barrel distortion
    k2 = 0.0
    p1 = 0.0
    p2 = 0.0

    distCoeff[0, 0] = k1
    distCoeff[1, 0] = k2
    distCoeff[2, 0] = p1
    distCoeff[3, 0] = p2

    # assume unit matrix for camera
    cam = np.eye(3, dtype=np.float32)

    cam[0, 2] = width / 2.0  # define center x
    cam[1, 2] = height / 2.0  # define center y
    cam[0, 0] = 10.  # define focal length x
    cam[1, 1] = 10.  # define focal length y

    # here the undistortion will be computed
    return cv.undistort(frame, cam, distCoeff)


def main():

    video_capture = cv.VideoCapture(sys.argv[1])
    kernel = np.ones((5,5), np.uint8) 
    while True:
        ret, image = video_capture.read()
        if  len(sys.argv) >2 and sys.argv[2] == 'pro':
            image = remove_barrel(image)
        if not ret:
            break
        dst = image
        hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)

        # define range of blue color in HSV
        lower_thresh = np.array([0, 0, 90])
        upper_thresh = np.array([190, 180, 255])
    # Threshold the HSV image
        mask = cv.inRange(hsv, lower_thresh, upper_thresh)
    # Bitwise-AND mask and original imageq
        mask = cv.bitwise_not(mask)
       # cv.imshow('mask',mask)

    # Apply a morphol transf to the image
        kernel = np.ones((5, 5), np.uint8)
        #Super closing
        dilation = cv.dilate(mask, kernel, iterations=8) #3
        erosion = cv.erode(dilation, kernel, iterations=5) #1
        
 

        erosion_blurred = cv.medianBlur(erosion, 5)
        #erosion_blurred = cv.blur(dilation2, (5,5))
        #erosion_blurred = cv.GaussianBlur(dilation2,(5,5),0)
        cv.imshow('HSV_erosion_blurred', erosion_blurred)

        contours, _ = cv.findContours(erosion_blurred, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    # cv.drawContours(resized, contours2, -1, (0, 0, 255), 2)

        rettangoli = []
        for cont in contours:
            if cv.contourArea(cont) > 6000:
                if cv.arcLength(cont, True) > 600:
                    arc_len = cv.arcLength(cont, True)
                    approx = cv.approxPolyDP(cont, 0.06 * arc_len, True)

                # or try to rectangle it
                    if (len(approx) > 2):
                        x, y, w, h = cv.boundingRect(cont)
                        cv.rectangle(dst, (x, y), (x + w, y + h), (255, 0, 0), 3)
                        rettangoli.append((x, y, w, h))
       # cv.setMouseCallback('final', mouse_callback, [rois, frame])

       # font = cv.FONT_HERSHEY_SIMPLEX
        #position_text = (10, dst.shape[0] - 30)
        #cv.putText(dst, str(len(rois)), position_text, font, 3, (0, 0, 255), 2, cv.LINE_AA)
        cv.imshow('Result', dst)



        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    # cv2.destroyAllWindows()
    #return rettangoli

if __name__ == "__main__":
    main()