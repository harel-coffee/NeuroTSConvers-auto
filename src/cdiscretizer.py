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
		sub_data = normalize (sub_data)
		if len (output) == 0:
			output =  sub_data.copy ()
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
		disc_threshold = filters. threshold_otsu (M[:, j:j+1]) - 0.06
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

#================================================
def dbscan_clustering (M):

	for j in range (M.shape [1]):
		clustering = KMeans (n_clusters=2, algorithm = "elkan").fit (M [:,j:j+1])
		M[:,j] = clustering.labels_
		print (M[:,j])

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

#===============================================================
def discretize_array (M_, cols, min, mean = False, peak = False):
	M = M_.copy ()
	for j in range (0, M.shape [1]):
		if peak:
			peaks, _ = find_peaks (M[:,j], height=0.0)

			for i in range (M.shape [0]):
				if mean:
					min = np. mean (M[:,j]) - 0.05

				if M[i,j] < min:
					M[i,j] = 0
				else:
					M[i,j] = 1

			for x in peaks:
				M[x,j] = 1

		else:
			if mean:
				min = np. mean (M[:,j]) - 0.05
				#print (cols[j], min)
			for i in range (M.shape [0]):
				if M[i,j] < min:
					M[i, j] = 0.0
				else:
					M[i, j] = 1.0

	return M


#=====================================================
if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument("--nbins", "-k", default = 2, type = int)
	parser.add_argument("--mean", "-mean",  action="store_true")
	parser.add_argument("--peak", "-peak",  action="store_true")
	parser.add_argument("--otsu", "-otsu",  action="store_true")
	parser.add_argument("--dbscan", "-scan",  action="store_true")
	parser.add_argument("--min", "-m", default = 0.0, type = float)
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
	#physio_hh_data = block_normalization (physio_hh_data, 24)
	#physio_hh_data = block_normalization (physio_hr_data, 24)
	#normalize (physio_hh_data)
	#normalize (physio_hr_data)

	if args. dbscan:
		print ("DBSCAN clustering")
		discr_physio_hh_data = dbscan_clustering (physio_hh_data)
		discr_physio_hr_data = dbscan_clustering (physio_hr_data)

	elif args. otsu:
		print ("Otsu discretization")
		#discr_physio_hh_data = otsu_discretization (physio_hh_data, 24)
		#discr_physio_hr_data = otsu_discretization (physio_hr_data, 24)

		discr_physio_hh_data = otsu (physio_hh_data, cols)
		discr_physio_hr_data = otsu (physio_hr_data, cols)

	else:
		discr_physio_hh_data = discretize_array (physio_hh_data,cols, args. min, args.mean, args.peak)
		discr_physio_hr_data = discretize_array (physio_hr_data,cols, args. min, args.mean, args.peak)

	discr_physio_hh_data = pd. DataFrame (discr_physio_hh_data, columns = cols)
	discr_physio_hr_data = pd. DataFrame (discr_physio_hr_data, columns = cols)

	# Save data in csv files
	discr_physio_hh_data. to_csv ("concat_time_series/discr_bold_hh_data.csv", sep = ';', index = False)
	discr_physio_hr_data. to_csv ("concat_time_series/discr_bold_hr_data.csv", sep = ';', index = False)

	discr_physio_hh_data. to_pickle ("concat_time_series/discr_bold_hh_data.pkl")
	discr_physio_hr_data. to_pickle ("concat_time_series/discr_bold_hr_data.pkl")
