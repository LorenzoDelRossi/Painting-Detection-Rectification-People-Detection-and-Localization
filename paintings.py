import cv2
import numpy as np
from paintingrectification import rectification
from paintingretrieval import retrieval_first, retrieval_list

def paintingspipeline(frame, stanza):
    dst = frame
    kernel = np.ones((5,5), np.uint8)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Cerco la soglia adatta per creare la maschera successivamente
    thresh, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)

    # Applico la soglia per creare la maschera dell'immagine
    mask = (gray < thresh).astype(np.uint8)*255

    # Applico una trasformazione morfologica per "ammorbidire" i contorni
    dilation = cv2.dilate(mask, kernel, iterations=3)
    erosion = cv2.erode(dilation, kernel, iterations=1)

    erosion_blurred = cv2.GaussianBlur(erosion, (5,5), 0)
    erosion_blurred_closed = cv2.morphologyEx(erosion_blurred, cv2.MORPH_CLOSE, kernel)
    #cv.imshow('erosion_blurred_closed', erosion_blurred_closed)


    contours, _ = cv2.findContours(erosion_blurred_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    #image_number = 0
    for cont in contours:
        if cv2.contourArea(cont) > 10000:
            if cv2.arcLength(cont, True) > 800:
                arc_len = cv2.arcLength(cont, True)
                approx = cv2.approxPolyDP(cont, 0.06 * arc_len, True)

                if (len(approx) == 4):
                    x, y, w, h = cv2.boundingRect(cont)
                    cv2.rectangle(dst, (x, y), (x + w, y + h), (255, 0, 0), 3)

                    if h/w < 2: # Per non rettificare e fare retrieval su pezzi di quadri
                        warp = rectification(frame, approx)
                        #cv2.imshow("warped {}".format(image_number), warp)
                        #cv2.imshow("box_{}".format(image_number), dst[y:y + h,x:x + w])
                        #image_number += 1
                        try:
                            stanza = retrieval_first(warp, x, y, w, h)
                            #stanza = retrieval_list(warp, x, y, w, h) # Per la versione che stampa la lista di similarità su terminale
                        except:
                            continue # Se non si riesce a fare la retrieval decentemente si va avanti invece di interrompere il video
                    #cv2.imshow("Full frame", dst)
    return stanza
                    #if stanza != 0:
                        #cv2.putText(dst, "Room " + str(stanza), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (254, 45, 43), 2, cv2.LINE_4)


