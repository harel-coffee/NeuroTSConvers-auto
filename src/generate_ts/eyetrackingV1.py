# coding: utf8
import numpy as np
import pandas as pd

from imutils import face_utils
import matplotlib. pyplot as plt
import cv2
import dlib
import sys
import os


import argparse

import utils.tools as ts

#===========================================================

def face_land_marks (image, predictor, face):
	shape = predictor(image, face)
	shape = face_utils.shape_to_np(shape)
	return shape

#===========================================================#

def get_minmax (frame, land_marks, item_name):

	if item_name == "face":
		item = np.array (land_marks)

	else:
		begin, end = face_utils.FACIAL_LANDMARKS_IDXS [item_name]
		item = np.array (land_marks [begin : end])

	xmin = item [np.argmin (item [:,0])]
	xmax = item [np.argmax (item [:,0])]

	ymin = item [np.argmin (item [:,1])]
	ymax = item [np.argmax (item [:,1])]

	return xmin, xmax, ymin, ymax

#===========================================================#

def landmark_to_rect (frame, land_marks, item_name, scale = 50.0):

	xmin, xmax, ymin, ymax = get_minmax (frame, land_marks, item_name)

	if item_name == "right_eye":
		scale = 100
		xrbmin, xrbmax, yrbmin, yrbmax = get_minmax (frame, land_marks, "right_eyebrow")
		xscale =  int ((float (scale) / 200) * (xmax[0] - xmin[0]))
		yscale =   - yrbmax[1] + ymin[1]

	elif item_name == "left_eye":
		scale = 100
		xrbmin, xrbmax, yrbmin, yrbmax = get_minmax (frame, land_marks, "left_eyebrow")
		xscale =  int ((float (scale) / 200) * (xmax[0] - xmin[0]))
		yscale =  - yrbmax[1] + ymin[1]

	elif item_name == "face":
		xscale =  int ((float (scale) / 200) * (xmax[0] - xmin[0]))
		yscale =  int ((float (scale) / 200) * (ymax[0] - ymin[0]))
		x = xmin[0] - xscale
		w = xmax[0] - x + xscale

		y = ymin[1] - yscale
		h = ymax [1] - y + yscale
		y = y - int (0.333 * h)
		h = h + int (0.333 * h)

		cv2.rectangle(frame, (x, y, w, h), (0,165,255), 2)
		return (x, y, w, h)

	else:
		xscale =  int ((float (scale) / 200) * (xmax[0] - xmin[0]))
		yscale =  int ((float (scale) / 200) * (ymax[0] - ymin[0]))

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
			cv2.circle (image, (x, y), 3, colors[-1], -1)
		l = l + 1

#==================================================

def aggregate_ts (data, index):

	aggregated_time_series = []
	begin_time = data[0][0]

	rows_ts = []
	j = 0
	for i in range (len (data)):
		if j >= len (index):
		    break
		if (data[i][0] > index [j]):
		    aggregated_time_series. append ([index [j]] + np.mean (rows_ts, axis = 0). tolist ())
		    j += 1
		    rows_ts = []
		rows_ts. append (data [i][1:])

	if len (rows_ts) > 0 and j < len (index):
		aggregated_time_series. append ([index [j]] + np.mean (rows_ts, axis = 0). tolist ())

	return aggregated_time_series

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
	parser.add_argument("video", help = "the path of the video to process.")
	parser.add_argument("out_dir", help = "the path where to store the results.")
	parser.add_argument("--show",'-s', help = "Showing the video.", action = "store_true")
	args = parser.parse_args()

	if args. out_dir == 'None':
	    usage ()
	    exit ()

	if args. out_dir[-1] != '/':
	    args. out_dir += '/'

	# Input directory
	subject = args. video.split ('/')[-2]
	conversation_name = args. video.split ('/')[-1]. split ('.')[0]
	out_file = args. out_dir + conversation_name

	if os.path.isfile ("%s.pkl"%out_file):
		test_df = pd.read_pickle ("%s.pkl"%out_file)
		if test_df. shape [0] == 50:
			print ("Conversation already processed!")
			exit (1)

	# hyper-parameters for bounding boxes shape
	frame_window = 10

	# starting video streaming
	video_capture = cv2.VideoCapture(args.video)

	# Frames frequence : fps frame per seconde
	fps = video_capture.get(cv2.CAP_PROP_FPS)

	eye_tracking_file = "time_series/%s/eyetracking_ts/%s.pkl"%(subject, conversation_name)
	eye_tracking_data = pd. read_pickle (eye_tracking_file). values

	openface_file = "time_series/%s/facial_features_ts/%s/%s.csv"%(subject, conversation_name, conversation_name)
	openface_data = pd. read_csv (openface_file, sep = ',', header = 0)

	'''print (conversation_name)
	print (openface_file)
	print (eye_tracking_file)
	exit (1)'''

	LDs = []
	cols = [" timestamp", " success"]
	for i in range (68):
		cols = cols + [" x_%d"%i, " y_%d"%i]

	openface_data = openface_data [cols]. values

	frame_width = int( video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
	frame_height =int( video_capture.get( cv2.CAP_PROP_FRAME_HEIGHT))

	#print (frame_width, frame_height)
	fourcc = cv2.VideoWriter_fourcc(*'XVID')

	#out = cv2.VideoWriter ("test/"+ subject + "_" + conversation_name + ".avi", fourcc, fps, (frame_width, frame_height))

	nb_frames = 0
	time_series = []
	current_time = 0

	while True:
		ret, bgr_image = video_capture.read()
		current_time += 1.0 / float (fps)

		if ret == False:
		    break

		row = [current_time, 0, 0, 0]
		success = openface_data [nb_frames, 1]

		if success:
			lds = openface_data [nb_frames, 2: ]
			landmarks = []

			for j in range (0, 136, 2):
				landmarks. append ([ int (lds [j]), int (lds [j + 1]) ])

			face = landmark_to_rect (bgr_image, landmarks, "face", scale = 10)
			mouth = landmark_to_rect (bgr_image, landmarks, "mouth")
			right_eye = landmark_to_rect (bgr_image, landmarks, "right_eye")
			left_eye = landmark_to_rect (bgr_image, landmarks, "left_eye")

			for x, y in landmarks:
				cv2.circle(bgr_image, (x, y), 2, (255, 0, 0), -1)

			x = (eye_tracking_data[nb_frames, 1]  / float (1279)) * frame_width   #1279
			y = (eye_tracking_data[nb_frames, 2] / float (1023)) * frame_height   #1023

			# check if eyetracking coordinates are not nan
			if not np.isnan (x) and not np.isnan (y):
				x = int (x)
				y = int (y)

				#plot eyetracking point
				#bgr_image = cv2.resize (bgr_image, (1279, 1023))
				cv2.circle(bgr_image, (x, y), 3, (0,0,255), -1)
				#bgr_image = cv2.resize (bgr_image, (frame_width, frame_height))

				# fill the time series
				if isin_rect (face, x, y):
					row [1] = 1
					if isin_rect (mouth, x, y):
						row [2] = 1
					if isin_rect (right_eye, x, y) or isin_rect (left_eye, x, y):
						row [3] = 1

		time_series. append (row)

		#cv2.imshow('wind', bgr_image)
		#out.write(bgr_image)

		#if cv2.waitKey(30) & 0xFF == ord('q'):
			#break

		nb_frames += 1


	physio_index = [0.6025]
	for i in range (1, 50):
		physio_index. append (1.205 + physio_index [i - 1])

	time_series = aggregate_ts (time_series, physio_index)
	#print (pd.DataFrame (time_series))

	video_capture.release()
	cv2.destroyAllWindows()

	#out.release()
	#exit (1)

	df = pd.DataFrame (time_series, columns = ["Time", "Face", "Mouth", "Eyes"])
	df.to_pickle (out_file + ".pkl")
