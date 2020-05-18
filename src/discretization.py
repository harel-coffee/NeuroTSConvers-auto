import pandas as pd
import numpy as np
import os
import sys
from glob import glob
from scipy.signal import find_peaks
from sklearn.cluster import KMeans
from sklearn import preprocessing
from skimage import filters
import argparse

from skimage import filters

from sklearn.cluster import DBSCAN, AffinityPropagation, MeanShift, KMeans, SpectralClustering


#================================================
def block_normalization (np_matrix, nblocks):

	output = []

	data_blocks = np. vsplit (np_matrix, nblocks)

	for sub_data in data_blocks:
		normalize (sub_data)
		if len (output) == 0:
			output =  sub_data
		else:
			output = np . concatenate ((output, sub_data), axis = 0)

	return output

#================================================
def discretize_matrix (mat, min):

	M = mat. copy ()

	for i in range (M.shape [0]):
		for j in range (0, M.shape [1]):
				if M[i,j] <= min:
					M[i,j] = 0
				else:
					M[i,j] = 1
	return M

#================================================
def otsu (M, cols):

	output = []

	for j in range (0, M.shape [1]):
		disc_threshold = filters. threshold_otsu (M[:, j:j+1]) - 0.05
		print (cols[j], disc_threshold)
		for i in range (M.shape [0]):
			if M[i,j] < disc_threshold:
				M[i,j] = 0
			else:
				M[i,j] = 1

	return M

#================================================
def otsu_discretization (np_matrix, nblocks):

	output = []

	data_blocks = np. vsplit (np_matrix, nblocks)

	for sub_data in data_blocks:
		sub_result = []
		disc_threshold = filters. threshold_otsu (sub_data)
		sub_result = discretize_matrix (sub_data, disc_threshold)

		if len (output) == 0:
			output =  sub_result
		else:
			output = np . concatenate ((output, sub_result), axis = 0)

	return output

#===================================================
def my_discretization_vect (vector):
	discr_vect = []
	min = np.mean (vector)
	if vector[0] < min:
		discr_vect. append (0)
	else:
		discr_vect. append (1)

	for i in range (1, len (vector)):

		if discr_vect [i-1] == 1:
			if abs (vector [i] - vector[i-1]) > 0.2 and vector[i-1] < 0.55:
				discr_vect. append (0)
			else:
				discr_vect. append (1)

		else:
			if abs (vector [i] - vector[i-1]) > 0.2 and vector[i-1] > 0.45:
				discr_vect. append (1)
			else:
				discr_vect. append (0)


	return discr_vect


#===================================================
def my_discretization (X):
	mat = X.copy ()
	for j in range (mat.shape [1]):
		mat[:,j] = my_discretization_vect (mat[:,j])
	return mat

#================================================
def dbscan_clustering (M):

	for j in range (M.shape [1]):
		#clustering = SpectralClustering (n_clusters=2, assign_labels="discretize", random_state=0). fit (M [:,j:j+1])
		#clustering = AffinityPropagation (). fit (M [:,j:j+1])
		#clustering = DBSCAN ().fit (M [:,j:j+1])
		clustering = KMeans (n_clusters=2, algorithm = "elkan").fit (M [:,j:j+1])
		M[:,j] = clustering.labels_
		print (M[:,j])

		'''for i in range (1, M.shape [0] - 1):
			if M[i, j] == 0 and M[i - 1, j] == 1 and M[i + 1, j] == 1:
				M[i, j] = 1'''

	return M

#====================================================
def normalize_vect (x):
	max = np.max (x)
	min = np.min (x)

	if min < max:
		for i in range (len (x)):
			x[i]= (x[i] - min) / (max - min)
#=====================================================

def normalize (M):
	minMax = np.empty ([M.shape[1], 2])
	for i in range(M.shape[1]):
		#print (M[:,i])
		max = np.max(M[:,i])
		min = np.min(M[:,i])
		minMax[i,0] = min
		minMax[i,1] = max

		if min < max:
			for j in range(M.shape[0]):
				M[j,i] = (M[j,i] - min) / (max - min)

	return minMax

#=====================================================

def find_peaks_ (y, height = 0):
	x = []
	for i in range (len (y)):
		if y[i] > height:
			x.append (i)
	return x

################################################
def get_sliding_windows (n, size = 10):
	winds = []
	for j in range (n - size):
		winds. append (range (j, j + size))

	return winds


#============================================================
def discretize_vect_sliding (x, win_size = 10):
	mat = []
	winds = get_sliding_windows (len (x) + 1, 10)

	for i in range (len (x)):
		values_in_windows = []
		for j in range (len (winds)):
			if i in winds[j]:
				indice_i = winds[j]. index (i)
				row = [x[k] for k in winds[j]]
				normalize_vect (row)
				for k in range (len (row)):
					if row [k] >= 0.5:
						row [k] = 1
					else:
						row [k] = 0
				values_in_windows. append (row [indice_i])

		if (float (np. sum (values_in_windows)) / len (values_in_windows)) > 0.5:
			mat. append (1.0)
		else:
			mat. append (0.0)

	return mat

#===============================================================
def auto_discretize (M, columns):

	for j in range (0, M.shape [1]):

		if columns [j] in ["LeftMotor", "RightMotor"]:
			min = 0.42

		elif columns [j] in ["RightSTS", "LeftSTS"]:
			min = 0.42

		else:
			min = 0.45

		for i in range (M.shape [0]):

			if M[i,j] < min:
				M[i, j] = 0.0
			else:
				M[i, j] = 1.0

	return M
#===============================================================
def discretize_array (M, cols, min = 0.1, mean = False, peak = False, sliding = False):


	for j in range (0, M.shape [1]):
		if peak:
			peaks, _ = find_peaks (M[:,j], height=min)
			for i in range (M.shape [0]):
				M[i,j] = 0

			for x in peaks:
				M[x,j] = 1

		elif sliding:
			M[:,j] = discretize_vect_sliding (M[:,j], win_size = 10)

		else:
			if mean:
				min = np. mean (M[:,j])
				print (cols[j], min)
			for i in range (M.shape [0]):
				if M[i,j] < min:
					M[i, j] = 0.0
				else:
					M[i, j] = 1.0

			'''for i in range (1, M.shape [0] - 1):
				if M[i, j] == 0 and M[i - 1, j] == 1 and M[i + 1, j] == 1:
					M[i, j] = 1'''

			'''for i in range (1, M.shape [0] - 1):
				if M[i, j] == 1 and M[i - 1, j] == 0 and M[i + 1, j] == 0:
					M[i, j] = 0'''

	return M

#========================================================
def new_discretization (M, min = 0.1, min_peaks = 0):

	for j in range (0, M.shape [1]):
		peaks, _ = find_peaks (M[:,j], height=min_peaks)
		for i in range (M.shape [0]):
			if min <= M[i,j] or i in peaks:
				M[i, j] = 1
			else:
				M[i, j] = 0

	return M


#=====================================================

def discretize_df_kmeans (df, k = 2):
	for col in df. columns [1:]:
		clustering = KMeans (n_clusters = k, random_state = 1). fit (df. loc[:, col]. values. reshape (-1, 1))

		#clustering = DBSCAN (eps=3, min_samples=2).fit (df. loc[:, col]. values. reshape (-1, 1))
		df [col] = clustering. labels_

#=====================================================
if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	#parser.add_argument("data_type", help="data type")
	parser.add_argument("--nbins", "-k", default = 2, type = int)
	parser.add_argument("--mean", "-mean",  action="store_true")
	parser.add_argument("--peak", "-peak",  action="store_true")
	parser.add_argument("--otsu", "-otsu",  action="store_true")
	parser.add_argument("--dbscan", "-scan",  action="store_true")
	parser.add_argument("--sliding", "-sl",  action="store_true")
	parser.add_argument("--auto", "-auto",  action="store_true")
	parser.add_argument("--kmeans", "-kmeans",  action="store_true")
	parser.add_argument("--my", "-my",  action="store_true")
	parser.add_argument("--min", "-m", default = 0.0, type = float)
	parser.add_argument("--type", "-t", default = "raw")
	args = parser.parse_args()

	print (args)
	""" store the discretization parameters """
	f= open("disc_params.txt","w+")
	for item in vars (args). items ():
		f. write ("%s: %s\n"%(item[0], item [1]))
	f. close ()

	physio_hh_data = pd.read_pickle ("concat_time_series/bold_hh_data.pkl"). values
	physio_hr_data = pd.read_pickle ("concat_time_series/bold_hr_data.pkl"). values

	cols = pd.read_pickle ("concat_time_series/bold_hh_data.pkl"). columns

	# normalization
	#physio_hh_data = block_normalization (physio_hh_data, 96)
	#physio_hh_data = block_normalization (physio_hr_data, 96)
	#normalize (physio_hh_data)
	#normalize (physio_hr_data)

	print (cols)
	print (np.min (physio_hh_data), np.max (physio_hh_data), np.mean (physio_hh_data))

	if args. dbscan:
		print ("DBSCAN clustering")
		discr_physio_hh_data = dbscan_clustering (physio_hh_data)
		discr_physio_hr_data = dbscan_clustering (physio_hr_data)

	elif args.my:
		print ("my binarisation")
		discr_physio_hh_data = my_discretization (physio_hh_data)
		discr_physio_hr_data = my_discretization (physio_hr_data)

	elif args. otsu:
		print ("Otsu discretization")
		#discr_physio_hh_data = otsu_discretization (physio_hh_data, 24)
		#discr_physio_hr_data = otsu_discretization (physio_hr_data, 24)

		discr_physio_hh_data = otsu (physio_hh_data, cols)
		discr_physio_hr_data = otsu (physio_hr_data, cols)

	elif args. auto:
		discr_physio_hh_data = auto_discretize (physio_hh_data, cols)
		discr_physio_hr_data = auto_discretize (physio_hr_data, cols)

	else:
		discr_physio_hh_data = discretize_array (physio_hh_data,cols,  min = args. min, mean = args.mean, sliding = args.sliding)
		discr_physio_hr_data = discretize_array (physio_hr_data,cols,  min = args. min, mean = args.mean, sliding = args.sliding)


	discr_physio_hh_data = pd. DataFrame (discr_physio_hh_data, columns = cols)
	discr_physio_hr_data = pd. DataFrame (discr_physio_hr_data, columns = cols)


	# Save data in csv files
	discr_physio_hh_data. to_csv ("concat_time_series/discr_bold_hh_data.csv", sep = ';', index = False)
	discr_physio_hr_data. to_csv ("concat_time_series/discr_bold_hr_data.csv", sep = ';', index = False)

	discr_physio_hh_data. to_pickle ("concat_time_series/discr_bold_hh_data.pkl")
	discr_physio_hr_data. to_pickle ("concat_time_series/discr_bold_hr_data.pkl")
