import cv2 as cv
import numpy as np
import sys

def consider_rectangular(corners, near_middle_sides, epsilon):
    """
    Return True if corners points can be considered as a rectangle, False otherwise.
    The function checks if left points (and right points) are aligned wrt to an error epsilon, if yes return True.

    :param corners:
    :param near_middle_sides:
    :param epsilon:
    :return:
    """

    top_left, top_right, bot_right, bot_left = corners
    top, right, bot, left = near_middle_sides

    score = 0

    if -epsilon < top_right[0] - right[0] < epsilon:
        score += 1
    if -epsilon < bot_right[0] - right[0] < epsilon:
        score += 1
    if -epsilon < top_left[0] - left[0] < epsilon:
        score += 1
    if -epsilon < bot_left[0] - left[0] < epsilon:
        score += 1

    if score >= 2:
        return True
    else:
        return False


def distance(point_a, point_b):
    dist = ((point_b[0] - point_a[0]) ** 2 + (point_b[1] - point_a[1]) ** 2) ** (1 / 2)
    return dist


def points_near(contours, A, B, C, D):
    """
    Find the 4 points of contours that are closest to A, B, C, D
    :param contours:
    :param A:
    :param B:
    :param C:
    :param D:
    :return:
    """
    init_distance_value = 1920 + 1080

    np_contours = np.asarray(contours)

    distanceA = init_distance_value
    distanceB = init_distance_value
    distanceC = init_distance_value
    distanceD = init_distance_value

    for i in range(np_contours.shape[0]):
        actual_sum = np_contours[i, 0].sum()
        new_distanceA = distance(np_contours[i, 0], A)
        if new_distanceA < distanceA:
            near_A = np_contours[i, 0]
            distanceA = new_distanceA

        new_distanceB = distance(np_contours[i, 0], B)
        if new_distanceB < distanceB:
            near_B = np_contours[i, 0]
            distanceB = new_distanceB

        new_distanceC = distance(np_contours[i, 0], C)
        if new_distanceC < distanceC:
            near_C = np_contours[i, 0]
            distanceC = new_distanceC

        new_distanceD = distance(np_contours[i, 0], D)
        if distance(np_contours[i, 0], D) < distanceD:
            near_D = np_contours[i, 0]
            distanceD = new_distanceD

    return [near_A, near_B, near_C, near_D]


def find_points_near_middle_point_sides(image, contours):
    T = [image.shape[1] / 2, image.shape[0]]
    R = [image.shape[1], image.shape[0] / 2]
    B = [image.shape[1] / 2, 0]
    L = [0, image.shape[0] / 2]

    return points_near(contours, T, R, B, L)


def find_points_near_corners(image, contours):
    TL = [0, image.shape[0]]
    TR = [image.shape[1], image.shape[0]]
    BR = [image.shape[1], 0]
    BL = [0, 0]

    return points_near(contours, TL, TR, BR, BL)


def rectification(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    # cv.imshow("gray", gray)

    kernel = np.array([[0, 1, 0],
                       [1, 1, 1],
                       [0, 1, 0]], np.uint8)
    # define range of color
    thresh, _ = cv.threshold(gray, 0, 255, cv.THRESH_OTSU)

    # print(thresh)
    # Threshold the grayscale image
    mask = (gray < thresh).astype(np.uint8) * 255

    # Apply a morphological operations to the image
    dilation = cv.dilate(mask, kernel, iterations=5)  # 3
    # cv.imshow("mask", mask)

    contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    # adjust contours list
    flat_list_points_contours = []
    for sublist in contours:
        for item in sublist:
            flat_list_points_contours.append(item)

    # Draw contours
    contour_color = (0, 255, 0)
    cv.drawContours(image, contours, -1, contour_color)

    corners = find_points_near_corners(image, flat_list_points_contours)
    midpoints_sides = find_points_near_middle_point_sides(image, flat_list_points_contours)

    # for point in corners:
    #    cv.circle(image, tuple(point), 7, (0, 0, 255), cv.FILLED)

    # Draw midpoints of the sides
    # for point in midpoints_sides:
    #     cv.circle(image, tuple(point), 5, (255, 0, 0), cv.FILLED)

    cv.imshow("contours", image)

    if consider_rectangular(corners, midpoints_sides, 30):

        h = int(((corners[0][1] - corners[3][1]) + (corners[1][1] - corners[2][1]) / 2))
        w = int(((corners[1][0] - corners[0][0]) + (corners[2][0] - corners[3][0]) / 2))
        dst = np.float32([[0, h],
                          [w, h],
                          [w, 0],
                          [0, 0]])

        src = np.float32(corners)
    else:

        h = midpoints_sides[0][1] - midpoints_sides[2][1]
        w = midpoints_sides[1][0] - midpoints_sides[3][0]

        h_div2 = int(h / 2)
        x_div2 = int(w / 2)
        dst = np.float32([[x_div2, h],
                          [w, h_div2],
                          [x_div2, 0],
                          [0, h_div2]])

        src = np.float32(midpoints_sides)

    M = cv.getPerspectiveTransform(src, dst)
    warped = cv.warpPerspective(image, M, (w, h))

    cv.imshow("warped image", warped)
    cv.waitKey(1)
    return warped


def main():
    video_capture = cv.VideoCapture(sys.argv[1])

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        # if bounding box = x, y, w, h
        # frame = image[y:y+h, x:x+w, :]
        # Not run if there are more picture in the same bounding boxes
        rectified_frame = rectification(frame)
        cv.imshow("image with corners", rectified_frame)

        cv.waitKey(1)

    video_capture.release()


if __name__ == "__main__":
    main()
