import numpy as np
import cv2 as cv

def rectification(orig, pts):
    pts = pts.reshape(4, 2)
    rect = np.zeros((4, 2), dtype="float32")
    # the top-left point has the smallest sum whereas the
    # bottom-right has the largest sum
    # ESEMPIO: rettangolo con punti (0,0), (0,1), (1,0), (1,1)
    s = pts.sum(axis=1) # sommo le x e le y dei 4 punti
    rect[0] = pts[np.argmin(s)] # il punto in alto a sinistra (0,0) aveva somma 0 quindi è l'argmin
    rect[2] = pts[np.argmax(s)] # il punto in basso a destra (1,1) aveva somma 2 quindi è l'argmax
    # compute the difference between the points -- the top-right
    # will have the minumum difference and the bottom-left will
    # have the maximum difference
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)] # il punto in alto a destra (0,1) aveva differenza -1 quindi è l'argmin
    rect[3] = pts[np.argmax(diff)] # il punto in basso a sinistra (1,0) aveva differenza 1 quindi è l'argmax
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
