# coding: utf8
import numpy as np
import pandas as pd
import seaborn

import matplotlib. pyplot as plt
import cv2
import dlib
import sys
import os

stderr = sys.stderr
sys.stderr = open(os.devnull, 'w') # hide keras messages
from keras.models import load_model
sys.stderr = stderr

from utils.face_classification.src.utils.datasets import get_labels
from utils.face_classification.src.utils.inference import draw_text
from utils.face_classification.src.utils.inference import draw_bounding_box
from utils.face_classification.src.utils.inference import apply_offsets
from utils.face_classification.src.utils.preprocessor import preprocess_input

import argparse

import utils.tools as ts

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def usage():
	print ("execute the script with -h for usage.")

if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument("video", help="the path of the video to process.")
	parser.add_argument("out_dir", help="the path where to store the results.")
	parser.add_argument("--show",'-s', help="Showing the video.", action="store_true")

	args = parser.parse_args()

	detection_model_path = 'src/utils/face_classification/trained_models/detection_models/haarcascade_frontalface_default.xml'
	emotion_model_path = 'src/utils/face_classification/trained_models/fer2013_mini_XCEPTION.119-0.65.hdf5'
	emotion_labels = get_labels('fer2013')

	if args. out_dir == 'None':
	    usage ()
	    exit ()

	if args. out_dir[-1] != '/':
	    args. out_dir += '/'

	# Input directory
	conversation_name = args. video.split ('/')[-1]. split ('.')[0]
	out_file = args. out_dir + conversation_name

	if os.path.isfile (out_file + ".pkl") and os.path.isfile (out_file + ".png"):
	    exit (1)

	# hyper-parameters for bounding boxes shape
	frame_window = 10

	# loading models
	face_detector = dlib.get_frontal_face_detector()

	# starting video streaming
	video_capture = cv2.VideoCapture(args.video)

	# Frames frequence : fps frame per seconde
	fps = video_capture.get(cv2.CAP_PROP_FPS)
	'''length = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
	print (fps)
	print (length)'''

	subject = args. video.split ('/')[2]
	eye_tracking_file = "time_series/%s/eye_ts/%s.pkl"%(subject, conversation_name)
	eye_tracking_data = pd. read_pickle (eye_tracking_file). values

	frame_width = int( video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
	frame_height =int( video_capture.get( cv2.CAP_PROP_FRAME_HEIGHT))
	fourcc = cv2.VideoWriter_fourcc(*'XVID')

	out = cv2.VideoWriter ("test.avi", fourcc, 20.0, (frame_width, frame_height))


	nb_frames = 0

	while True:
		ret, bgr_image = video_capture.read()
		#bgr_image = cv2.resize (bgr_image, (1280, 1024))
		#print (bgr_image. shape)
		#exit (1)

		if ret == False:
		    break

		# screen Resolution
		'''screen_res = 1279, 1023
		scale_width = screen_res[0] / bgr_image. shape[1]
		scale_height = screen_res[1] / bgr_image. shape[0]
		scale = min (scale_width, scale_height)'''

		# resize sizes
		wind_width = int (bgr_image. shape[1] * scale)
		wind_height = int (bgr_image. shape[0] * scale)

		cv2. namedWindow ("wind", cv2.WINDOW_NORMAL)
		#cv2. resizeWindow ("wind", wind_width, wind_height)
		rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)

		x = int (eye_tracking_data[nb_frames, 1] * 0.5)
		y = int (eye_tracking_data[nb_frames, 2] * 0.46875)

		cv2.circle(rgb_image, (x, y), 5, (0,0,255), -1)
		#cv2.circle(rgb_image, (400, 450), 10, (0,0,255), -1)

		#draw_text(face, rgb_image, emotion_text, color, 0, -45, 1, 1)

		bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
		#out.write(bgr_image)
		cv2.imshow('wind', bgr_image)

		if cv2.waitKey(30) & 0xFF == ord('q'):
			break

		nb_frames += 1

	video_capture.release()
	cv2.destroyAllWindows()
	#out.release()
