# USAGE python "painting detection.py" -i video.mp4

import sys
import cv2 as cv
import numpy as np
import argparse



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", required=True,
	help="path to input image")
    args = vars(ap.parse_args())
    video_capture = cv.VideoCapture(args["input"])
    kernel = np.array([[0, 1, 0],
    [1, 1, 1],
    [0, 1, 0]], np.uint8)
    while True:
        ret, image = video_capture.read()
        if not ret:
            break
        dst = image
        
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

        # Cerco la soglia adatta per creare la maschera successivamente
        thresh, _ = cv.threshold(gray, 0, 255, cv.THRESH_OTSU)

        # Applico la soglia per creare la maschera
        mask = (gray < thresh).astype(np.uint8)*255

        # Applico una trasformazione morfologica per "ammorbidire" i contorni
        dilation = cv.dilate(mask, kernel, iterations=3)
        erosion = cv.erode(dilation, kernel, iterations=1)
        
 

        erosion_blurred = cv.GaussianBlur(erosion,(5,5),0)
        #cv.imshow('HSV_erosion_blurred', erosion_blurred)
        

        contours, _ = cv.findContours(erosion_blurred, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        rettangoli = []
        for cont in contours:
            if cv.contourArea(cont) > 10000:
                if cv.arcLength(cont, True) > 800:
                    arc_len = cv.arcLength(cont, True)
                    approx = cv.approxPolyDP(cont, 0.06 * arc_len, True)

                    if (len(approx) > 2):
                        x, y, w, h = cv.boundingRect(cont)
                        cv.rectangle(dst, (x, y), (x + w, y + h), (255, 0, 0), 3)
                        rettangoli.append((x, y, w, h))
        cv.imshow('Result', dst)
        key = cv.waitKey(30)

        if key == 32:
            cv.waitKey() # Se premo barra spaziatrice il video va in pausa
        elif key == ord('q'):
            break

    video_capture.release()
    #return rettangoli



if __name__ == "__main__":
    main()