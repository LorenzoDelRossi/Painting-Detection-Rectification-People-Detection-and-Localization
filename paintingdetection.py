import cv2 as cv
import numpy as np
from paintingrectification import rectification
from paintingretrieval import retrieval_first

def paintingdetection(frame):
    dst = frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Cerco la soglia adatta per creare la maschera successivamente
    thresh, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)

    # Applico la soglia per creare la maschera dell'immagine
    mask = (gray < thresh).astype(np.uint8)*255

    # Applico una trasformazione morfologica per "ammorbidire" i contorni
    dilation = cv2.dilate(mask, kernel, iterations=3)
    erosion = cv2.erode(dilation, kernel, iterations=1)

    erosion_blurred = cv2.GaussianBlur(erosion, (5,5), 0)
    #cv.imshow('HSV_erosion_blurred', erosion_blurred)


    contours, _ = cv2.findContours(erosion_blurred, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    #stanza = 0 # Stanza inizializzata qui se vogliamo farla sparire quando non ci sono retrieval nel frame
    image_number = 0
    for cont in contours:
        if cv2.contourArea(cont) > 10000:
            if cv2.arcLength(cont, True) > 800:
                arc_len = cv2.arcLength(cont, True)
                approx = cv2.approxPolyDP(cont, 0.06 * arc_len, True)

                if (len(approx) == 4):
                    x, y, w, h = cv2.boundingRect(cont)
                    if h/w < 2: # Per non rettificare e cercare pezzi di quadri
                        warp = rectification(frame, approx)
                        #cv.imwrite("ROI_{}.png".format(image_number), warp)
                        #cv.imshow("warped {}".format(image_number), warp)
                        #cv.imshow("box_{}".format(image_number), dst[y:y + h,x:x + w])
                        image_number += 1
                        try:
                            stanza = retrieval_first(warp, x, y, w, h)
                        except:
                            continue # Se non si riesce a fare la retrieval decentemente si va avanti invece di interrompere il video
                        #cv.imshow("cleaned_{}".format(image_number), new)
                    if stanza != 0:
                        cv2.putText(dst, "Room " + str(stanza), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (128, 255, 100), 2, cv2.LINE_4)


