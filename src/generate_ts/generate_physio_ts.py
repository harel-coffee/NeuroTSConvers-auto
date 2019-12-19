#import scipy.io as sio
from mat4py import loadmat #load matlab files
import pandas as pd
import numpy as np
from glob import glob
import os
import sys
import argparse
from sklearn.preprocessing import KBinsDiscretizer

from mat4py import loadmat # load matlab files

#============================================================
def normalize_vect (x):
	max = np.max (x)
	min = np.min (x)

	if min < max:
		for i in range (len (x)):
			x[i]= (x[i] - min) / (max - min)

#============================================================
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
					if row [k] > 0.5:
						row [k] = 1
					else:
						row [k] = 0
				values_in_windows. append (row [indice_i])

		if (float (np. sum (values_in_windows)) / len (values_in_windows)) >= 0.5:
			mat. append (1.0)
		else:
			mat. append (0.0)
	return mat


def correspondance (sujet_um):

    if sujet_um == "13":
        sujets_corresp = "12"

    elif sujet_um == "14":
        sujets_corresp = "13"

    elif sujet_um == "15":
        sujets_corresp = "14"

    elif sujet_um == "16":
        sujets_corresp = "15"

    elif sujet_um == "17":
        sujets_corresp = "16"

    elif sujet_um == "18":
        sujets_corresp = "17"

    elif sujet_um == "20":
        sujets_corresp = "18"

    elif sujet_um == "21":
        sujets_corresp = "19"

    elif sujet_um == "22":
        sujets_corresp = "20"

    elif sujet_um == "23":
        sujets_corresp = "21"

    elif sujet_um == "24":
        sujets_corresp = "22"

    elif sujet_um == "25":
        sujets_corresp = "23"

    else:
        sujets_corresp = sujet_um

    return sujets_corresp


#======================================================

def nearestPoint(vect, value):
    dist = abs(value - vect[0])
    pos = 0

    for i in range(1, len(vect)):
        if (vect[i] - value) < dist  and vect[i] > value :
            dist = vect[i] - value
            pos = i

    return pos

#======================================================

def add_duration(df):
    duration = []

    for i in range(df.shape[0]):
        if i == (df.shape[0] - 1):
            duration.append([df.iloc[i, 3] / 1000.0, df.iloc[i, 3] / 1000.0, 0])
        else:
            duration.append([df.iloc[i, 3] / 1000.0, df.iloc[i + 1, 3] / 1000.0,
                             df.iloc[i + 1, 3] / 1000.0 - df.iloc[i, 3] / 1000.0])

    df = pd.concat([df, pd.DataFrame(duration, columns=['begin', 'fin', 'Interval'])], axis=1)

    return df

#======================================================

def convers_to_df (data, discretized_data, smoothed_data, colnames, index, begin, end, type_conv, num_conv):
	index_normalized = index[begin:end]
	start_pt = index_normalized [0] # - 1.205 / 2

	#print (begin, end)
	for j in range(0, end - begin):
	   index_normalized[j] -= start_pt

	convers_data = pd. DataFrame (data [begin:end])
	convers_data_discr = pd. DataFrame (discretized_data [begin:end])
	#convers_data_smoothed = pd. DataFrame (smoothed_data [begin:end])

	if convers_data. shape[0]  == 49:
	    convers_data = convers_data. append (convers_data. iloc[48,:], ignore_index = True)
	    convers_data_discr = convers_data_discr. append (convers_data_discr. iloc[48,:], ignore_index = True)

	convers_data. reset_index (inplace = True)
	convers_data.columns = ['Time (s)'] +  colnames

	convers_data_discr. reset_index (inplace = True)
	convers_data_discr. columns = ['Time (s)'] +  colnames

	'''convers_data_smoothed. reset_index (inplace = True)
	convers_data_smoothed. columns = ['Time (s)'] +  colnames'''

	out_filename = "time_series/" + subject + "/physio_ts/convers-" + testBlock + "_" + type_conv + "_" + "%03d"%num_conv + ".pkl"
	out_disc_filename = "time_series/" + subject + "/discretized_physio_ts/convers-" + testBlock + "_" + type_conv + "_" + "%03d"%num_conv + ".pkl"
	#smooth_filename = "time_series/" + subject + "/physio_smooth_ts/convers-" + testBlock + "_" + type_conv + "_" + "%03d"%num_conv + ".pkl"

	convers_data.to_pickle (out_filename)
	convers_data_discr.to_pickle (out_disc_filename)
	#convers_data_smoothed.to_pickle (smooth_filename)

#======================================================

if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument("subject", help="the subject name (for example sub-01) or 'ALL' to process al the subjects.")

	args = parser.parse_args()

	_arg = args.subject

	subjects = []
	subjs = []

	if not os.path.exists("time_series"):
	    os.makedirs("time_series")

	for i in range(1, 26):
	    if i < 10:
	        subjects.append("sub-0%s" % str(i))
	    else:
	        subjects.append("sub-%s" % str(i))

	if _arg == 'ALL' or _arg == 'all':
	    subjs = subjects
	    subjs.remove('sub-12')
	    subjs.remove('sub-19')
	    print (subjs)
	elif _arg in subjects:
	    subjs = [_arg]
	else:
	    usage()
	    exit(1)

	for subject in subjs:
		for fname in ["physio_ts", "discretized_physio_ts", "physio_smooth_ts"]:
			if not os.path.exists ("time_series/%s/%s"%(subject, fname)):
			    os.makedirs("time_series/%s/%s"%(subject, fname))

		print (subject, 15 * '*', '\n')

		num_subj = subject.split('-')[-1]
		num_sujet_physio = '0' + correspondance (num_subj)

		#data_dat = sio.loadmat("data/physio_data/denoised/ROI_Subject" + num_subj + "_Condition000.mat")
		data_dat = loadmat("data/physio_data/ROIdata/ROI_Subject" +  num_sujet_physio + "_Condition000.mat")

		colnames = data_dat ['names']

		data = []
		for i in range (len (data_dat ['names'])):
			if len (data) == 0:
				data = np. array (data_dat ['data'][i])
			else:
				data = np. concatenate ((data, np. array (data_dat ['data'][i])), axis = 1)

		discr_data = np. empty (data. shape, dtype = float)
		for j in range (data. shape[1]):
			norm = data[:,j]. copy ()
			normalize_vect (norm)
			discretizer = KBinsDiscretizer (n_bins = 2, encode = 'ordinal', strategy = 'kmeans')
			discr_data[:,j] = discretizer. fit_transform (norm. reshape (-1, 1)). flatten ()

		'''discr_data = np. empty (data. shape, dtype=float)
		for j in range (data. shape[1]):
		    discr_data[:,j] = discretize_vect_sliding (data[:,j]. tolist (), win_size = 10)'''

		# smoothing the bold signal
		smoothed_signal = []

		index = [0.6025]
		for i in range (1, len (data)):
		    index. append (1.205 + index [i - 1])

		# ------------------ Analyse log files
		log_files = glob("data/physio_data/logfiles/*" + subject + "*")
		testBlocks = ["" for i in range(4)]
		for i in range(4):
		    for logfile in log_files:
		        if "TestBlocks" + str(i + 1) in logfile:
		            testBlocks[i] = "TestBlocks" + str(i + 1)
		            break

		# if there is no logfile, we continue to the next subject
		if len (testBlocks) == 0:
		    print ("subject %s has no logfiles" % subject)
		    continue

		indice_block = 0
		for testBlock in testBlocks:
		    log_block_file = glob ("data/physio_data/logfiles/*" + subject + "_task-convers-" + testBlock + "*") [0]
		    df = pd.read_csv (log_block_file, sep='\t', header=0)
		    df = df [['condition', 'image', 'duration', 'ONSETS_MS']]

		    df = add_duration(df)

		    hh_convers = df [df.condition.str.contains("CONV1")] [['condition', 'begin', 'fin']]
		    hr_convers = df [df.condition.str.contains("CONV2")] [['condition', 'begin', 'fin']]

		    nb_hh_convers = hh_convers. shape [0]
		    nb_hr_convers = hr_convers. shape [0]

		    hh = 1
		    hr = 2

		    for i in range(nb_hh_convers):
		        begin = nearestPoint (index, hh_convers.values[i][1]) + (385 * indice_block)
		        end = nearestPoint (index, hh_convers.values[i][2]) + (385 * indice_block) + 2  # add two observatiosn after the endof the conversation
		        convers_to_df (data, discr_data, smoothed_signal, colnames, index, begin, end, "CONV1", hh)
		        hh += 2


		    for i in range(nb_hr_convers):
		        begin = nearestPoint (index, hr_convers.values[i][1]) + (385 * indice_block)
		        end = nearestPoint (index, hr_convers.values[i][2]) + (385 * indice_block) + 2 # add two observatiosn after the endof the conversation
		        convers_to_df (data, discr_data, smoothed_signal, colnames, index, begin, end, "CONV2", hr)
		        hr += 2
		    indice_block += 1
