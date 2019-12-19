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
	#print (winds)

	#wins = []

	for i in range (len (x)):
		values_in_windows = []
		for j in range (len (winds)):

			if i in winds[j]:
				indice_i = winds[j]. index (i)
				row = [x[k] for k in winds[j]]
				normalize_vect (row)
				for k in range (len (row)):
					if row [k] > 0:
						row [k] = 1
					else:
						row [k] = 0
				values_in_windows. append (row [indice_i])

		if (float (np. sum (values_in_windows)) / len (values_in_windows)) > 0.5:
			mat. append (1.0)
		else:
			mat. append (0.0)

		#mat. append (float (np. sum (values_in_windows)) / len (values_in_windows))
		#wins. append (values_in_windows)

	#print (wins)
	return mat

#===============================================================
def discretize_array (df, min = 0.1, mean = False, peak = False):

	cols = df. columns
	M = df. values

	for j in range (1, M.shape [1]):
		if peak:
			peaks, _ = find_peaks (M[:,j], height=min)
			for i in range (M.shape [0]):
				M[i,j] = 0

			for x in peaks:
				M[x,j] = 1
		else:
			for i in range (M.shape [0]):
				if mean:
					min = np. mean (M[:,j])
				if M[i,j] <= min:
					M[i, j] = 0.0
				else:
					M[i, j] = 1.0

	return pd.DataFrame (M, columns = cols)

#=====================================================

def discretize_df_kmeans (df, k = 3):
	for col in df. columns [1:]:
		clustering = KMeans (n_clusters = k, random_state = 1). fit (df. loc[:, col]. values. reshape (-1, 1))
		'''print (clustering. inertia_)
		exit (1)'''
		#clustering = DBSCAN (eps=3, min_samples=2).fit (df. loc[:, col]. values. reshape (-1, 1))
		df [col] = clustering. labels_

#=====================================================
if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument("data_type", help="data type")
	parser.add_argument("--nbins", "-k", default = 2, type = int)
	parser.add_argument("--mean", "-mean",  action="store_true")
	parser.add_argument("--peak", "-peak",  action="store_true")
	parser.add_argument("--kmeans", "-kmeans",  action="store_true")
	parser.add_argument("--threshold", "-min", default = 0.0, type = float)
	parser.add_argument("--type", "-t", default = "raw")
	args = parser.parse_args()

	print (args)

	""" store the discretization parameters """
	f= open("disc_params.txt","w+")
	for item in vars (args). items ():
		f. write ("%s: %s\n"%(item[0], item [1]))
	f. close ()

	subjects_in = glob ("time_series/*")
	subjects_out = glob ("time_series/*")

	#=====================================================#
	""" discretize physiological data """
	if args.data_type == "p":

		if args. type == "raw":
			in_data_type = "/physio_ts"
		elif args. type == "diff":
			in_data_type = "/physio_diff_ts"
		elif args. type == "smooth":
			in_data_type = "/physio_smooth_ts"

		for i in range (len (subjects_in)):

			if not os.path. exists ("%s/discretized_physio_ts"%subjects_in[i]):
				os.makedirs ("%s/discretized_physio_ts"%subjects_in[i])

			subjects_in[i] = subjects_in[i] + in_data_type
			subjects_out[i] = subjects_out[i] + "/discretized_physio_ts"

			pkl_files = glob ("%s/*pkl"%subjects_in[i])
			pkl_files. sort ()

			for filepath in pkl_files:
				df = pd. read_pickle (filepath)
				filename = filepath. split('/')[-1]

				if args.kmeans:
					cols = df. columns
					df = df. values
					min_max_scaler = preprocessing. MinMaxScaler ()
					df [:,1:] = min_max_scaler. fit_transform (df [:,1:])
					df = pd. DataFrame (df, columns = cols)
					discretize_df_kmeans (df, k = args. nbins)

				else:
					#discretize_df (df, float (args.threshold), n_classes = args. nbins)
					df = discretize_array (df, args.threshold, args. mean, args. peak)
				df.to_pickle ("%s/%s" %(subjects_out[i], filename))
