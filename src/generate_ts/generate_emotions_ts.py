# coding: utf8

import numpy as np
import pandas as pd
#import seaborn

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
#from utils.face_classification.src.utils.inference import draw_bounding_box
from utils.face_classification.src.utils.inference import apply_offsets
from utils.face_classification.src.utils.preprocessor import preprocess_input

import argparse

import utils.tools as ts

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

colors = ["black", "darkblue", "brown", "red", "slategrey", "darkorange", "grey","blue", "indigo", "darkgreen"]

#==================================================

def usage():
	print ("execute the script with -h for usage.")

#==================================================

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
	emotion_offsets = (20, 40)

	# loading models
	face_detector = dlib.get_frontal_face_detector()
	emotion_classifier = load_model(emotion_model_path, compile=False)

	# getting input model shapes for inference
	emotion_target_size = emotion_classifier.input_shape[1:3]

	# starting lists for calculating modes
	emotion_window = []

	# starting video streaming
	video_capture = cv2.VideoCapture(args.video)

	# Frames frequence : fps frame per seconde
	fps = video_capture.get(cv2.CAP_PROP_FPS)
	length = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))

	'''subject = args. video.split ('/')[2]
	eye_tracking_file = "time_series/%s/eye_ts/%s.pkl"%(subject, conversation_name)
	eye_tracking_data = pd. read_pickle (eye_tracking_file). values'''

	emotions_states = {'angry': 0, 'disgust':0, 'fear':0, 'happy':0,'sad':0, 'surprise':0, 'neutral':0}
	labels = ['angry', 'disgust', 'fear', 'happy','sad', 'surprise', 'neutral']

	# parameter initilization
	nb_frames = 0
	columns = ['Time (s)'] + labels
	time_series = []
	set_of_emotions = emotions_states. copy ()
	current_time = 0

	# Index of neurophysiological data
	index = [0.6025]
	for i in range (1, 50):
		index. append (1.205 + index [i - 1])

	# reading the video
	j = 0
	while True:
		ret, bgr_image = video_capture.read()

		if ret == False:
		    break

		gray_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
		rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
		faces = face_detector (gray_image, 1)

		# We suppose we have at most on face in each image
		if len (faces) > 0:
			face_cordinates = faces[0]
			x = face_cordinates.left()
			y = face_cordinates.top() #could be face.bottom() - not sure
			w = face_cordinates.right() - face_cordinates.left()
			h = face_cordinates.bottom() - face_cordinates.top()

			face = (x,y,w,h)

			x1, x2, y1, y2 = apply_offsets(face, emotion_offsets)
			gray_face = gray_image[y1:y2, x1:x2]

			try:
				gray_face = cv2.resize(gray_face, (emotion_target_size))
			except:
				continue

			gray_face = preprocess_input(gray_face, True)
			gray_face = np.expand_dims(gray_face, 0)
			gray_face = np.expand_dims(gray_face, -1)
			emotion_prediction = emotion_classifier.predict(gray_face)
			emotion_probability = np.max(emotion_prediction)
			emotion_label_arg = np.argmax(emotion_prediction)
			emotion_text = emotion_labels[emotion_label_arg]

			'''if emotion_text in emotions_states. keys ():
				emotions_states [emotion_text] = emotion_probability
			else:
				print ("Error! " + str (emotion_text) + " not in labels!")
				exit (1)'''

			#set_of_emotions. append ([emotion_text, emotion_probability])
			set_of_emotions[emotion_text] = max (emotion_probability, set_of_emotions[emotion_text])

		current_time += 1.0 / fps

		if j >= 50:
			break
		if current_time >= index [j]:
			time_series. append ([index [j]] + [set_of_emotions[emotion] for emotion in labels])
			set_of_emotions = emotions_states. copy ()
			#print ([index [j]] + [set_of_emotions[emotion] for emotion in labels])
			j += 1

		if args. show:
			draw_bounding_box(face, rgb_image, color)
			cv2.circle(rgb_image, (int (eye_tracking_data[nb_frames, 1]), int (eye_tracking_data[nb_frames, 2])), 30, (0,0,255), -1)

		if emotion_text == 'angry':
			color = emotion_probability * np.asarray((255, 0, 0))
		elif emotion_text == 'sad':
			color = emotion_probability * np.asarray((0, 0, 255))
		elif emotion_text == 'happy':
			color = emotion_probability * np.asarray((255, 255, 0))
		elif emotion_text == 'surprise':
			color = emotion_probability * np.asarray((0, 255, 255))
		else:
			color = emotion_probability * np.asarray((0, 255, 0))

		color = color.astype(int)
		color = color.tolist()

		draw_text(face, rgb_image, emotion_text, color, 0, -45, 1, 1)

		bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
		cv2.imshow('window_frame', bgr_image)

		if cv2.waitKey(30) & 0xFF == ord('q'):
			break

		#nb_frames += 1


	if j < 50:
		time_series. append ([index[j]] + [set_of_emotions[emotion] for emotion in labels])

	#df = pd.DataFrame (time_series, columns = columns)
	#print (df)
	#exit (1)
	#df. to_pickle (out_file + '.pkl')
	#seaborn. catplot (x = "Time (s)", y = "Emotions", data = df)
	#ts. plot_df (df, labels, figname= out_file + '.png', figsize=(12,9), y_lim = [0,1])
