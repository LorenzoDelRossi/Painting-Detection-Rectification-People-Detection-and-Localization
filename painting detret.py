# USAGE python "painting detret.py" -i video.mp4

import sys
import cv2 as cv
import imutils as imu
import numpy as np
import argparse
from paintingretrieval import retrieval_first


def rectification(orig, pts):
    pts = pts.reshape(4, 2)
    rect = np.zeros((4, 2), dtype="float32")
    # the top-left point has the smallest sum whereas the
    # bottom-right has the largest sum
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    # compute the difference between the points -- the top-right
    # will have the minumum difference and the bottom-left will
    # have the maximum difference
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    # now that we have our rectangle of points, let's compute
    # the width of our new image
    (tl, tr, br, bl) = rect
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    # ...and now for the height of our new image
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    # take the maximum of the width and height values to reach
    # our final dimensions
    maxWidth = max(int(widthA), int(widthB))
    maxHeight = max(int(heightA), int(heightB))
    # construct our destination points which will be used to
    # map the screen to a top-down, "birds eye" view
    destiny = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")
    # calculate the perspective transform matrix and warp
    # the perspective to grab the screen
    M = cv.getPerspectiveTransform(rect, destiny)
    warp = cv.warpPerspective(orig, M, (maxWidth, maxHeight))
    return warp




def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", required=True,
	help="path to input image")
    args = vars(ap.parse_args())
    video_capture = cv.VideoCapture(args["input"])
    kernel = np.array([[0, 1, 0],
    [1, 1, 1],
    [0, 1, 0]], np.uint8)
    stanza = 0 # Stanza qui se vogliamo sempre tenere un numero di stanza dalla prima retrieval in poi
    while True:
        ret, image = video_capture.read()
        if not ret:
            break
        dst = image
        
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

        # Cerco la soglia adatta per creare la maschera successivamente
        thresh, _ = cv.threshold(gray, 0, 255, cv.THRESH_OTSU)

        # Applico la soglia per creare la maschera dell'immagine
        mask = (gray < thresh).astype(np.uint8)*255

        # Applico una trasformazione morfologica per "ammorbidire" i contorni
        dilation = cv.dilate(mask, kernel, iterations=3)
        erosion = cv.erode(dilation, kernel, iterations=1)
        
        erosion_blurred = cv.GaussianBlur(erosion, (5,5), 0)
        #cv.imshow('HSV_erosion_blurred', erosion_blurred)
        

        contours, _ = cv.findContours(erosion_blurred, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        rettangoli = []
        #stanza = 0 # Stanza inizializzata qui se vogliamo farla sparire quando non ci sono retrieval nel frame
        image_number = 0
        for cont in contours:
            if cv.contourArea(cont) > 10000:
                if cv.arcLength(cont, True) > 800:
                    arc_len = cv.arcLength(cont, True)
                    approx = cv.approxPolyDP(cont, 0.06 * arc_len, True)

                    if (len(approx) == 4):
                        x, y, w, h = cv.boundingRect(cont)
                        if h/w < 2: # Per non rettificare e cercare pezzi di quadri
                            warp = rectification(image, approx)
                            #cv.imwrite("ROI_{}.png".format(image_number), warp)
                            #cv.imshow("warped {}".format(image_number), warp)
                            #cv.imshow("box_{}".format(image_number), dst[y:y + h,x:x + w])
			    image_number += 1
                            try:
                               stanza = retrieval_first(warp, x, y, w, h)
                            except:
                                continue # Se non si riesce a fare la retrieval decentemente si va avanti invece di interrompere il video
                            #cv.imshow("cleaned_{}".format(image_number), new)
                        cv.rectangle(dst, (x, y), (x + w, y + h), (255, 0, 0), 3)
                        rettangoli.append((x, y, w, h))
                        if stanza != 0:
                            cv.putText(dst, "Stanza " + str(stanza), (0, dst.shape[0]// 2), cv.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 0), 2, cv.LINE_AA)
        cv.imshow('Result', dst)
        #cv.imshow("warp", warp)
        key = cv.waitKey(30)

        if key == 32:
            cv.waitKey() # Se premo barra spaziatrice il video va in pausa
        elif key == ord('q'):
            break

    video_capture.release()
    #return rettangoli



if __name__ == "__main__":
    main()
