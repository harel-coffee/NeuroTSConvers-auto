import pandas as pd
import numpy as np
import os
import sys
from glob import glob
from scipy.signal import find_peaks
from sklearn.cluster import KMeans
from sklearn import preprocessing
from skimage import filters

from sklearn.cluster import DBSCAN, AffinityPropagation, MeanShift, KMeans, SpectralClustering

import argparse

#================================================
def otsu (M, cols):

	output = []

	for j in range (0, M.shape [1]):
		disc_threshold = filters. threshold_otsu (M[:, j:j+1])
		for i in range (M.shape [0]):
			if M[i,j] < disc_threshold:
				M[i,j] = 0
			else:
				M[i,j] = 1

	return M

#================================================
def dbscan_clustering (M):

	for j in range (M.shape [1]):
		clustering = KMeans (n_clusters=2, algorithm = "elkan"). fit (M [:,j:j+1])
		M[:,j] = clustering.labels_
		print (M[:,j])

	return M

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
			print (len (peaks))
		if mean:
			min = np. mean (M[:,j])
			#print (cols[j], min)
		for i in range (M.shape [0]):
			if M[i,j] < min:
				M[i, j] = 0.0
			else:
				M[i, j] = 1.0
		if peak:
			for i in range (M.shape [0]):
				if i in peaks:
					M[i,j] = 1

	return M

#=====================================================
if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument("--nbins", "-k", default = 2, type = int)
	parser.add_argument("--mean", "-mean",  action="store_true")
	parser.add_argument("--bold", "-bold",  action="store_true")
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

	os. system ("rm concat_time_series/discr_bold_hh_data.csv")
	os. system ("rm concat_time_series/discr_bold_hr_data.csv")
	os. system ("rm concat_time_series/discr_bold_hh_data.pkl")
	os. system ("rm concat_time_series/discr_bold_hr_data.pkl")

	physio_hh_data = pd.read_pickle ("concat_time_series/bold_hh_data.pkl"). values
	physio_hr_data = pd.read_pickle ("concat_time_series/bold_hr_data.pkl"). values

	cols = pd.read_pickle ("concat_time_series/bold_hh_data.pkl"). columns

	if args. dbscan:
		print ("DBSCAN clustering")
		discr_physio_hh_data = dbscan_clustering (physio_hh_data)
		discr_physio_hr_data = dbscan_clustering (physio_hr_data)

	elif args. bold:
		discr_physio_hh_data = disc_mean_bold (physio_hh_data, cols)
		discr_physio_hr_data = disc_mean_bold (physio_hr_data, cols)
	elif args. otsu:
		print ("Otsu discretization")
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
