# coding: utf8
import numpy as np
import pandas as pd

import cv2
import imutils
import os

import argparse

#===========================================================

def image_colorfulness(image):
	# split the image into its respective RGB components
	(B, G, R) = cv2.split(image.astype("float"))

	# compute rg = R - G
	rg = np.absolute(R - G)

	# compute yb = 0.5 * (R + G) - B
	yb = np.absolute(0.5 * (R + G) - B)

	# compute the mean and standard deviation of both `rg` and `yb`
	(rbMean, rbStd) = (np.mean(rg), np.std(rg))
	(ybMean, ybStd) = (np.mean(yb), np.std(yb))

	# combine the mean and standard deviations
	stdRoot = np.sqrt((rbStd ** 2) + (ybStd ** 2))
	meanRoot = np.sqrt((rbMean ** 2) + (ybMean ** 2))

	# derive the "colorfulness" metric and return it
	return stdRoot + (0.3 * meanRoot)

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
			aggregated_time_series. append ([index [j]] + np.nanmean (rows_ts, axis = 0). tolist ())
			j += 1
			rows_ts = []
		rows_ts. append (data [i][1:])

	if len (rows_ts) > 0 and j < len (index):
		aggregated_time_series. append ([index [j]] + np.nanmean (rows_ts, axis = 0). tolist ())

	return aggregated_time_series

#=============================================

if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument("video", help="the path of the video to process.")
	parser.add_argument("out_dir", help="the path where to store the results.")
	parser.add_argument("--show",'-s', help="Showing the video.", action="store_true")
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
			print ("Already processed")
			exit (1)

	print (conversation_name)
	# hyper-parameters for bounding boxes shape
	frame_window = 10

	# starting video streaming
	video_capture = cv2.VideoCapture(args.video)

	# Frames frequence : fps frame per seconde
	fps = video_capture.get(cv2.CAP_PROP_FPS)

	eye_tracking_file = "time_series/%s/eye_ts/%s.pkl"%(subject, conversation_name)
	eye_tracking_data = pd. read_pickle (eye_tracking_file). values

	LDs = []
	cols = [" timestamp", " success"]

	frame_width = int( video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
	frame_height =int( video_capture.get( cv2.CAP_PROP_FRAME_HEIGHT))

	fourcc = cv2.VideoWriter_fourcc(*'XVID')

	nb_frames = 0
	time_series = []
	current_time = 0

	limit = 20

	while True:
		ret, bgr_image = video_capture.read()
		current_time += 1.0 / float (fps)

		if ret == False:
		    break

		x = (eye_tracking_data[nb_frames, 1]  / float (1279)) * frame_width
		y = (eye_tracking_data[nb_frames, 2] / float (1023)) * frame_height

		# check if eyetracking coordinates are not nan
		if np.isnan (x) == False and np.isnan (y) == False:
			x = int (x)
			y = int (y)

			if x <= 0 or y <= 0:
				time_series. append ([current_time, np.nan, np.nan, np.nan])

			else:

				if x > limit and y > limit:
					w = limit
					h = limit
				else:
					if x < limit:
						w = limit - x
					else:
						w = limit
					if y < limit:
						h = limit - y
					else:
						h = limit

				small_image = bgr_image [x - w : x + w, y - h : y + h]

				try:
					partial_colorfulness = image_colorfulness (small_image)
				except:
					partial_colorfulness = 0

				rect = cv2.rectangle(bgr_image, (x - limit, y - limit, 2 * limit, 2 * limit), (0,165,255), 2)
				time_series. append ([current_time, image_colorfulness (bgr_image), partial_colorfulness, partial_colorfulness / image_colorfulness (bgr_image)])
		else:
			time_series. append ([current_time,  np.nan, np.nan, np.nan])

		nb_frames += 1

		#cv2.imshow('wind', bgr_image)

		"""if cv2.waitKey(20) & 0xFF == ord('q'):
			break"""

	physio_index = [1.205 / 2]
	for i in range (1, 50):
		physio_index. append (1.205 + physio_index [i - 1])

	time_series = aggregate_ts (time_series, physio_index)

	video_capture.release()
	cv2.destroyAllWindows()

	df = pd.DataFrame (time_series, columns = ["Time", "colorfulness", "part_colorfulness", "ratio"])

	if df.isnull().any().any():
		df. fillna (method = 'ffill' , inplace = True)
	if df.isnull().any().any():
		df. fillna (method = 'backfill' , inplace = True)

	df.to_pickle (out_file + ".pkl")
