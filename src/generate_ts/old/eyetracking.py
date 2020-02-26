# coding: utf8
import numpy as np
import pandas as pd
#import seaborn

from imutils import face_utils
import matplotlib. pyplot as plt
import cv2
import dlib
import sys
import os

stderr = sys.stderr
sys.stderr = open(os.devnull, 'w') # hide keras messages
from keras.models import load_model
sys.stderr = stderr

import argparse

import utils.tools as ts

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def usage():
	print ("execute the script with -h for usage.")

#===========================================================
def face_land_marks (image, predictor, face):
	shape = predictor(image, face)
	shape = face_utils.shape_to_np(shape)
	return shape

#===========================================================#

def get_minmax (frame, land_marks, item_name):
	begin, end = face_utils.FACIAL_LANDMARKS_IDXS [item_name]

	mouth = np.array (land_marks [begin : end])

	xmin = mouth [np.argmin (mouth[:,0])]
	xmax = mouth [np.argmax (mouth[:,0])]

	ymin = mouth [np.argmin (mouth[:,1])]
	ymax = mouth [np.argmax (mouth[:,1])]
	return xmin, xmax, ymin, ymax

#===========================================================#

def landmark_to_rect (frame, land_marks, item_name, scale = 50.0):

	xmin, xmax, ymin, ymax = get_minmax (frame, land_marks, item_name)

	if item_name == "right_eye":
		scale = 100
		xrbmin, xrbmax, yrbmin, yrbmax = get_minmax (frame, land_marks, "right_eyebrow")
		xscale =  int ((float (scale) / 200) * (xmax[0] - xmin[0]))
		yscale =   - yrbmax[1] + ymin[1]
		#yscale =  int ((float (scale) / 200) * (ymax[0] - ymin[0]))

	elif item_name == "left_eye":
		scale = 100
		xrbmin, xrbmax, yrbmin, yrbmax = get_minmax (frame, land_marks, "left_eyebrow")
		xscale =  int ((float (scale) / 200) * (xmax[0] - xmin[0]))
		yscale =  - yrbmax[1] + ymin[1]
		#yscale =  int ((float (scale) / 200) * (ymax[0] - ymin[0]))

	else:
		xscale =  int ((float (scale) / 200) * (xmax[0] - xmin[0]))
		yscale =  int ((float (200) / 200) * (ymax[0] - ymin[0]))

	x = xmin[0] - xscale
	w = xmax[0] - x + xscale

	y = ymin[1] - yscale
	h = ymax [1] - y + yscale

	cv2.rectangle(frame, (x, y, w, h), (0,165,255), 2)

	return (x, y, w, h)

#=================================================

def plot_landMarks (image, shape):
	l = 0
	colors_names = ["red", "pink", "pink", "blue", "blue", "orange","black"]
	colors = [[255, 0, 0], [255, 128, 0], [255, 128, 0], [0,0,255], [0,0,255], [0,165,255], [0, 0, 0]]

	for (name, (i, j)) in face_utils.FACIAL_LANDMARKS_IDXS.items():
		for (x, y) in shape[i:j]:
			#cv2.circle (image, (x, y), 3, colors[l], -1)
			cv2.circle (image, (x, y), 3, colors[-1], -1)
		l = l + 1

#==================================================

def plot_face (image, face):
	x = face.left()
	y = face.top()
	w = face.right() - face.left()
	h = face.bottom() - face.top()
	cv2.rectangle(image, (x, y, w, h), (255, 0, 0), 2)

#==================================================

def isin_dlib_face (face, x, y):
	if x >= face.left() and x <= face.right() and y >= face.top() and y <= face.bottom():
		return True
	else:
		return False

#==================================================

def isin_rect (rect, x, y):
	if x >= rect[0] and x <= rect[0] + rect[2] and y >= rect[1] and y <= rect[1] + rect[3]:
		return True
	else:
		return False

#=============================================

if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument("video", help="the path of the video to process.")
	parser.add_argument("out_dir", help="the path where to store the results.")
	parser.add_argument("--show",'-s', help="Showing the video.", action="store_true")

	args = parser.parse_args()

	face_cascade = cv2.CascadeClassifier('src/opencv_models/haarcascades/haarcascade_frontalface_default.xml')
	#eye_cascade = cv2.CascadeClassifier('src/opencv_models/haarcascades/haarcascade_eye.xml')
	eye_cascade = cv2.CascadeClassifier('src/opencv_models/haarcascades/haarcascade_eye_tree_eyeglasses.xml')
	landMarksdetector = dlib.shape_predictor ("src/utils/facial_landmarks_model/shape_predictor_68_face_landmarks.dat")

	face_detector = dlib.get_frontal_face_detector ()
	#face_detector = dlib.cnn_face_detection_model_v1 ()

	if args. out_dir == 'None':
	    usage ()
	    exit ()

	if args. out_dir[-1] != '/':
	    args. out_dir += '/'

	# Input directory
	subject = args. video.split ('/')[-2]
	conversation_name = args. video.split ('/')[-1]. split ('.')[0]
	out_file = args. out_dir +  subject +  "_" + conversation_name


	# hyper-parameters for bounding boxes shape
	frame_window = 10

	# starting video streaming
	video_capture = cv2.VideoCapture(args.video)

	# Frames frequence : fps frame per seconde
	fps = video_capture.get(cv2.CAP_PROP_FPS)

	eye_tracking_file = "time_series/%s/eye_ts/%s.pkl"%(subject, conversation_name)
	eye_tracking_data = pd. read_pickle (eye_tracking_file). values

	frame_width = int( video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
	frame_height =int( video_capture.get( cv2.CAP_PROP_FRAME_HEIGHT))
	fourcc = cv2.VideoWriter_fourcc(*'XVID')

	out = cv2.VideoWriter ("test/"+ subject + "_" + conversation_name + ".avi", fourcc, fps, (frame_width, frame_height))

	nb_frames = -1
	time_series = []
	current_time = 0
	i = 0

	while True:
		while i < 1:
			ret, bgr_image = video_capture.read()
			current_time += 1.0 / float (fps)
			nb_frames += 1
			i += 1
		if ret == False:
		    break

		i = 0

		#cv2. namedWindow ("wind", cv2.WINDOW_NORMAL)
		bgr_image = cv2.resize (bgr_image, (1279, 1023))
		gray_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)

		faces = face_detector (gray_image, 0)
		x = eye_tracking_data[nb_frames, 1]
		y = eye_tracking_data[nb_frames, 2]
		row = [current_time, 0, 0, 0]

		# check if eyetracking coordinates are not nan
		if not np.isnan (x) and not np.isnan (y):
			x = int (x)
			y = int (y)
			# check if there is a face in the image
			if len (faces) > 0:
				face = faces [0]
				plot_face (bgr_image, face)
				cv2.circle(bgr_image, (x, y), 5, (0,0,255), -1)

				shape =  list (face_land_marks (gray_image, landMarksdetector, face))
				plot_landMarks (bgr_image, shape)
				mouth = landmark_to_rect (bgr_image, shape, "mouth")
				right_eye = landmark_to_rect (bgr_image, shape, "right_eye")
				left_eye = landmark_to_rect (bgr_image, shape, "left_eye")

				if isin_dlib_face (face, x, y):
					row [1] = 1
					if isin_rect (mouth, x, y):
						row [2] = 1
					if isin_rect (right_eye, x, y) or isin_rect (left_eye, x, y):
						row [3] = 1

		time_series. append (row)

		bgr_image = cv2.resize (bgr_image, (frame_width, frame_height))
		cv2.imshow('wind', bgr_image)
		out.write(bgr_image)

		if cv2.waitKey(20) & 0xFF == ord('q'):
			break

	video_capture.release()
	cv2.destroyAllWindows()
	out.release()

	df = pd.DataFrame (time_series, columns = ["Time", "Face", "Mouth", "Eyes"])
	df.to_csv (out_file + ".csv", sep = ';', index = False)
