# coding: utf8

# import the necessary packages
from imutils import face_utils
import numpy as np
import argparse
import imutils
import dlib
import cv2

import sys
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
import os

from colour import Color

#------------------------------------------#
#    Dimension reduction with PCA 		   #
#          on pandas Dataframe             #
#------------------------------------------#
def reduce_df_PCA (df):
	scaler = MinMaxScaler(feature_range=[0, 1])
	data_rescaled = scaler.fit_transform(df.values)
	pca = PCA (n_components = 'mle', svd_solver = 'full')
	pca. fit (data_rescaled)
	variances_vector = pca.explained_variance_ratio_

	cumul = 0
	header_ = []
	for ncomp, variance in zip (range (len (variances_vector)), variances_vector):
		cumul += variance
		header_. append ("Comp_"+str(ncomp+1))
		if cumul >= 0.99:
			break

	#print ("Nb of PCs that explain 0.99 of total variance: %s" % (ncomp+1))
	reduced_df = pd.DataFrame (pca. transform (data_rescaled)[:, 0:ncomp+1], columns = header_, index=df.index)
	reduced_df. reset_index (inplace=True)
	return reduced_df

#-----------------------------------------------------------------#
# Get eyes, eyebrows, jaw, mouths and noses positions in the face.
#-----------------------------------------------------------------#
def face_land_marks (image, detector, predictor):

	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	# detect Face (we suppose we have one face)

	rect = detector(gray, 1)[0]

	shape = predictor(gray, rect)
	shape = face_utils.shape_to_np(shape)


	return shape, rect

#-----------------------------------------------------------------#
# 					Plot landmarks on an image
#-----------------------------------------------------------------#

def plot_landMarks (image, shape, face):
	l = 0
	colors_names = ["red", "pink", "pink", "blue", "blue", "orange","black"]
	colors = [[255, 0, 0], [255, 128, 0], [255, 128, 0], [0,0,255], [0,0,255], [0,165,255], [0, 0, 0]]

	for (name, (i, j)) in face_utils.FACIAL_LANDMARKS_IDXS.items():
		for (x, y) in shape[i:j]:
			cv2.circle (image, (x, y), 1, colors[l], -1)
		l = l + 1
	x = face.left()
	y = face.top()
	w = face.right() - face.left()
	h = face.bottom() - face.top()
	#x1, y1, x2, y2, w, h = face.left(), face.top(), face.right() + 1, face.bottom() + 1, face.width(), face.height()
	#cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
	cv2.rectangle(image, (x, y, w, h), (255, 0, 0), 2)


#--------------------------------------#
# usage
def usage():
    print ("execute the script with -h for usage.")


#--------------------------------------#
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


	#print ("Facial landmarks detection ------")
	conversation_name = args.video.split ('/')[-1]. split ('.')[0]
	out_file = args.out_dir + conversation_name + "_reducedLandmarks.pkl"


	if os.path.isfile (out_file):
		#print ("facial landmarks already processed")
		exit (1)

	# Load face a,d landMarks detectors
	face_detector = dlib.get_frontal_face_detector ()
	landMarksdetector = dlib.shape_predictor ("src/utils/facial_landmarks_model/shape_predictor_68_face_landmarks.dat")

	#fourcc = cv2.VideoWriter_fourcc(*'XVID')
	video = cv2.VideoCapture(args.video)
	fps = video.get(cv2.CAP_PROP_FPS)

	#out = cv2.VideoWriter('output.avi',fourcc, 20.0, (640,480))
	#out = cv2.VideoWriter('output.avi',fourcc, fps, (640,480), True)

	time_series = []
	time_series = []
	time_series_compact = []
	first_positions = []
	current_time = 0

	dic = face_utils.FACIAL_LANDMARKS_IDXS.items()

	positions = ["" for i in range (68)]
	for item, (i,j) in dic:
		l=0
		for k in range (i, j):
			positions[k] =item+'_'+str(l)
			l += 1

	excluded_landmarks = ['nose']
	positions_reduced = [x for x in positions if x.split ('_')[0] not in excluded_landmarks]
	positions_reduced. insert (0, "Time")

	coordinates_cols = ["Time"]
	for x in  positions_reduced[1:]:
		coordinates_cols. append (x + '_x')
		coordinates_cols. append (x + '_y')

	#coordinates_cols. insert (0, "Time")

	#i = 0
	while(True):
		ret, frame = video.read()
		#i += 1

		if ret==False:
			break

		try:
			shape_, rect =  face_land_marks (frame, face_detector, landMarksdetector)
			shape_ = list (shape_)
			shape = shape_[0:27] + shape_[36:68]
			coordinates = []
			distances = []

			for j in range (len (shape)):
				# Compute the distance from each point to the central noise point (supposed fix)
				distances. append (np.sqrt((shape[j][0] - shape[30][0])**2 + (shape[j][1] - shape[30][0])**2 ))
				#coordinates. append (shape[j][0])
				#coordinates. append (shape[j][1])

			#time_series. append ([current_time] + coordinates)
			time_series_compact. append ([current_time] + distances)
			current_time = (1.0 / fps) + current_time
		except:
			continue

		if args. show:
			# Draw land marks on the image
			plot_landMarks (frame, shape_, rect)
			#out.write(frame)
			cv2.imshow("Output", frame)
			if cv2.waitKey(1) & 0xFF == ord('q'):
				break

    # Release the video
	video.release()
	cv2.destroyAllWindows()


	# Store time series to csv files
	#df = pd.DataFrame(time_series, columns=coordinates_cols)
	#df_compact = pd.DataFrame(time_series_compact, columns=positions_reduced)
	#df_compact. set_index ('Time', inplace=True)

	#df_compact.to_csv (args_dict['out_dir']+"landMarks_compact.csv", sep=';', header=True, index=False)
	#df.to_csv (args_dict['out_dir']+"landMarks.csv", sep=';', header=True, index=False)


	#reduced_df = reduce_df_PCA (df_compact)
	#reduced_df. to_pickle (out_file)
	#reduced_df. to_csv (out_file, sep=';', header = True, index=True, index_label = 'Time (s)')

	#os. system ("mv results/features_extraction/*pdf results/pdf/")
