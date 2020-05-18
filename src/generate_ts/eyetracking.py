"""
	Author: Youssef Hmamouche
	Year: 2019
	Using extracted feature detected (Openface) and eyetracking files to generate extra features
"""

import sys, os, inspect, argparse, importlib
import numpy as np
import pandas as pd

import cv2, dlib
from scipy.stats import mode as sc_mode
from imutils import face_utils

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
maindir = os.path.dirname(parentdir)

resampling_spec = importlib.util.spec_from_file_location("resampling", "%s/src/resampling.py"%maindir)
resampling = importlib.util.module_from_spec(resampling_spec)
resampling_spec.loader.exec_module(resampling)

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
# extract the location of an item as rectangle using ladmarks and the image
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

		return (x, y, w, h)

	else:
		xscale =  int ((float (scale) / 200) * (xmax[0] - xmin[0]))
		yscale =  int ((float (scale) / 200) * (ymax[0] - ymin[0]))

	x = xmin[0] - xscale
	w = xmax[0] - x + xscale

	y = ymin[1] - yscale
	h = ymax [1] - y + yscale

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
	parser.add_argument("--save",'-sv', help = "Save the new video.", action = "store_true")
	parser.add_argument("--demo",'-d', help = "If this script is using for demo.", action = "store_true")
	parser.add_argument("--eyetracking",'-eye', help = "the path of the eyetracking file.")
	parser.add_argument("--facial_features",'-faf', help = "the path of the facial features file (csv file).")
	args = parser.parse_args()

	if args. out_dir == 'None':
	    usage ()
	    exit ()

	if args. out_dir[-1] != '/':
	    args. out_dir += '/'

	# Input directory and Output file
	subject = args. video.split ('/')[-2]

	if args. demo:
		eye_tracking_file = args. eyetracking
		openface_file = args. facial_features
		conversation_name = "facial_features_eyetracking"
	else:
		eye_tracking_file = "time_series/%s/gaze_coordinates_ts/%s.pkl"%(subject, conversation_name)
		conversation_name = args. video.split ('/')[-1]. split ('.')[0]
		openface_file = "time_series/%s/facial_features_ts/%s/%s.csv"%(subject, conversation_name, conversation_name)

	out_file = args. out_dir + conversation_name
	if os.path.isfile ("%s.pkl"%out_file):
		test_df = pd.read_pickle ("%s.pkl"%out_file)
		if test_df. shape [0] == 50:
			print ("Conversation already processed!")
			exit (1)

	# read the video
	video_capture = cv2.VideoCapture(args.video)

	#  Video parameters
	frames_nb = int (video_capture. get (7))
	fps = video_capture.get (cv2.CAP_PROP_FPS)
	frame_width = int( video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
	frame_height = int( video_capture.get( cv2.CAP_PROP_FRAME_HEIGHT))

	# create output video if the save argument is specified
	if args. save:
		fourcc = cv2.VideoWriter_fourcc(*'XVID')
		out = cv2.VideoWriter (args.out_dir + "/" + conversation_name + ".avi", fourcc, fps, (frame_width, frame_height))

	# If demo, we use eyetracking and openface csv files as input,
	# else, we find them automatically based on the name of the video


	# read eyetracking and openface csv files
	eye_tracking_data = pd. read_pickle (eye_tracking_file) #. values. astype (float)
	saccades = eye_tracking_data . loc [:, ["Time (s)","saccades"]]. values. astype (float)
	eye_tracking_data = eye_tracking_data . loc [:, ["Time (s)", "x", "y"]]. values. astype (float)
	openface_data = pd. read_csv (openface_file, sep = ',', header = 0)

	# Construct the index of the video stream
	video_index = [1.0 / fps ]
	for i in range (1, frames_nb):
	    video_index. append (1.0 / 30.0 + video_index [i - 1])

	# resample gaze coordinates to the video frequency
	gaze_coordiantes = resampling. resample_ts (eye_tracking_data, video_index, mode = "mean")

	# extract time index and 2D landmarks columns from openface data
	cols = [" timestamp", " success"]
	for i in range (68):
		cols = cols + [" x_%d"%i, " y_%d"%i]

	openface_data = openface_data [cols]. values


	# start loop over video stream
	nb_frames = 0
	face_time_series = []
	current_time = 0
	lds = []

	while True:
		ret, bgr_image = video_capture.read()
		current_time += 1.0 / float (fps)

		if ret == False:
			break

		row = [current_time, 0, 0, 0]
		success = openface_data [nb_frames, 1]

		# check first if landmarks detection has not been failed
		if success:
			lds = openface_data [nb_frames, 2: ]
			landmarks = []

			for j in range (0, 136, 2):
				landmarks. append ([ int (lds [j]), int (lds [j + 1]) ])

			face = landmark_to_rect (bgr_image, landmarks, "face", scale = 10)
			mouth = landmark_to_rect (bgr_image, landmarks, "mouth")
			right_eye = landmark_to_rect (bgr_image, landmarks, "right_eye")
			left_eye = landmark_to_rect (bgr_image, landmarks, "left_eye")

			#for x, y in landmarks:
				#cv2.circle (bgr_image, (x, y), 2, (255, 0, 0), -1)

			# rescale coordinate according the screen of the experience
			x = (gaze_coordiantes [nb_frames, 1]  / float (1279)) * frame_width   #1279
			y = (gaze_coordiantes [nb_frames, 2] / float (1023)) * frame_height   #1023


			# check if eyetracking coordinates are not nan
			if not np.isnan (x) and not np.isnan (y):
				x = int (x)
				y = int (y)

				# plot eyetracking point
				# bgr_image = cv2.resize (bgr_image, (1279, 1023))
				cv2.circle(bgr_image, (x, y), 3, (0,0,255), -1)


				# Append observations to time series
				if isin_rect (face, x, y):
					row [1] = 1
					if isin_rect (mouth, x, y):
						row [2] = 1
					if isin_rect (right_eye, x, y) or isin_rect (left_eye, x, y):
						row [3] = 1

		face_time_series. append (row)

		if args. show:
			cv2.imshow('wind', bgr_image)

		if args. save:
			out.write(bgr_image)

		if cv2.waitKey(30) & 0xFF == ord('q'):
			break
		nb_frames += 1

	video_capture.release()
	cv2.destroyAllWindows()

	if args.save:
		out. release ()

	# compute the index of the BOLD signal frequency
	# TODO: make the expected frequency as input argument
	physio_index = [0.6025]
	for i in range (1, 50):
		physio_index. append (1.205 + physio_index [i - 1])

	# resampling data according the BOLD signal frequency
	saccades = resampling. resample_ts (saccades, physio_index, mode = "sum")[:, 1:]
	coordinates_resampled = resampling. resample_ts (eye_tracking_data, physio_index, mode = "mean")

	# compute and resample the gradient of the gaze coordinates
	coordinates_gradient = np. gradient (eye_tracking_data [:,1:3], eye_tracking_data [:,0], axis = 0)
	coordinates_gradient = np. concatenate ((np. reshape (eye_tracking_data [:,0], (-1, 1)), coordinates_gradient), axis = 1)
	coordinates_gradient = resampling. resample_ts (coordinates_gradient, physio_index, mode = "mean")[:, 1:]

	# resample facial-time-series
	face_time_series = resampling. resample_ts (face_time_series, physio_index, mode = "sum")[:, 1:]

	# Concatenate all columns in one dataframe
	output_time_series = pd.DataFrame (np. concatenate ((coordinates_resampled, coordinates_gradient, saccades, face_time_series), axis = 1),
										columns = ["Time (s)", "x", "y", "Vx", "Vy", "saccades", "Face", "Mouth", "Eyes"])
	output_time_series.to_pickle (out_file + ".pkl")
