# USAGE
# python peopledetection.py --input videos/airport.mp4 --output output/airport_output.avi

import numpy as np
import argparse
import imutils
import time
import cv2
import os
# Project imports
from paintingretrieval import retrieval_first
from paintingrectification import rectification

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True,
	help="path to input video")
ap.add_argument("-o", "--output", required=True,
	help="path to output video")
ap.add_argument("-c", "--confidence", type=float, default=0.7,
	help="minimum probability to filter weak detections")
ap.add_argument("-t", "--threshold", type=float, default=0.3,
	help="threshold when applying non-maxima suppression")
args = vars(ap.parse_args())

# load the COCO class labels
LABELS = open("yolo-coco/coco.names").read().strip().split("\n")

# load our YOLO object detector trained on COCO dataset (80 classes)
# and determine only the *output* layer names that we need from YOLO
print("[INFO] loading YOLO from disk...")
net = cv2.dnn.readNetFromDarknet("yolo-coco/yolov3.cfg", "yolo-coco/yolov3.weights")
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# initialize the video stream, pointer to output video file, and
# frame dimensions
vs = cv2.VideoCapture(args["input"])
writer = None
(W, H) = (None, None)

# try to determine the total number of frames in the video file
try:
	prop = cv2.cv.CV_CAP_PROP_FRAME_COUNT if imutils.is_cv2() \
		else cv2.CAP_PROP_FRAME_COUNT
	total = int(vs.get(prop))
	print("[INFO] {} total frames in video".format(total))

# an error occurred while trying to determine the total
# number of frames in the video file
except:
	print("[INFO] could not determine # of frames in video")
	print("[INFO] no approx. completion time can be provided")
	total = -1


#kernel = np.array([[0, 1, 0],
#[1, 1, 1],
#[0, 1, 0]], np.uint8)
kernel = np.ones((5,5), np.uint8)
stanza = 0 # variabile stanza utile successivamente per la localization

# loop over frames from the video file stream
while True:
	# read the next frame from the file
	(grabbed, frame) = vs.read()

	# if the frame was not grabbed, then we have reached the end
	# of the stream
	if not grabbed:
		break

	# if the frame dimensions are empty, grab them
	if W is None or H is None:
		(H, W) = frame.shape[:2]
	tmp = cv2.medianBlur(frame, 5)
	# construct a blob from the input frame and then perform a forward
	# pass of the YOLO object detector, giving us our bounding boxes
	# and associated probabilities
	blob = cv2.dnn.blobFromImage(tmp, 1 / 255.0, (416, 416),
		swapRB=True, crop=False)
	net.setInput(blob)
	start = time.time()
	layerOutputs = net.forward(ln)
	end = time.time()

	# initialize our lists of detected bounding boxes, confidences,
	# and class IDs, respectively
	boxes = []
	confidences = []
	classIDs = []

	# loop over each of the layer outputs
	for output in layerOutputs:
		# loop over each of the detections
		for detection in output:
			# extract the class ID and confidence (i.e., probability)
			# of the current object detection
			scores = detection[5:]
			classID = np.argmax(scores)
			confidence = scores[classID]

			# filter out weak predictions by ensuring the detected
			# probability is greater than the minimum probability
			if confidence > args["confidence"]:
				# scale the bounding box coordinates back relative to
				# the size of the image, keeping in mind that YOLO
				# actually returns the center (x, y)-coordinates of
				# the bounding box followed by the boxes' width and
				# height
				box = detection[0:4] * np.array([W, H, W, H])
				(centerX, centerY, width, height) = box.astype("int")

				# use the center (x, y)-coordinates to derive the top
				# and and left corner of the bounding box
				xp = int(centerX - (width / 2))
				yp = int(centerY - (height / 2))

				# update our list of bounding box coordinates,
				# confidences, and class IDs
				boxes.append([xp, yp, int(width), int(height)])
				confidences.append(float(confidence))
				classIDs.append(classID)

	# apply non-maxima suppression to suppress weak, overlapping
	# bounding boxes
	idxs = cv2.dnn.NMSBoxes(boxes, confidences, args["confidence"],
		args["threshold"])

	# ensure at least one detection exists
	if len(idxs) > 0:
		# loop over the indexes we are keeping
		for i in idxs.flatten():
			# extract the bounding box coordinates
			if classIDs[i] == 0: # se la detection ha trovato una persona
				(xp, yp) = (boxes[i][0], boxes[i][1])
				(wp, hp) = (boxes[i][2], boxes[i][3])

			# draw a bounding box rectangle and label on the frame
			#color = [int(c) for c in COLORS[classIDs[i]]]
			#cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
			# INIZIO PAINTING DETECTION

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
				erosion_blurred_closed = cv2.morphologyEx(erosion_blurred, cv2.MORPH_CLOSE, kernel)
				#cv.imshow('HSV_erosion_blurred', erosion_blurred)
			

				contours, _ = cv2.findContours(erosion_blurred_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

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
			
			# FINE PAINTING DETECTION

				cv2.rectangle(frame, (xp, yp), (xp + wp, yp + hp), (0, 255, 0), 2)
				text = "{}: {:.4f}".format(LABELS[0],
								   confidences[i])
				cv2.putText(frame, f"{LABELS[0]}: {confidences[i]}", (xp, yp - 5),
				cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
				




	# check if the video writer is None
	if writer is None:
		# initialize our video writer
		fourcc = cv2.VideoWriter_fourcc(*"MJPG")
		writer = cv2.VideoWriter(args["output"], fourcc, 30,
			(frame.shape[1], frame.shape[0]), True)

		# some information on processing single frame
		if total > 0:
			elap = (end - start)
			print("[INFO] single frame took {:.4f} seconds".format(elap))
			print("[INFO] estimated total time to finish: {:.4f}".format(
				elap * total))

	# write the output frame to disk
	writer.write(frame)

# release the file pointers
print("[INFO] cleaning up...")
writer.release()
vs.release()
#return boxes # per ritornare la lista delle bounding boxes in formato (x,y,w,h)