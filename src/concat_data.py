import os, glob, argparse, inspect, sys
import pandas as pd
import numpy as np
from sklearn import preprocessing

#============================================================
def get_lagged_colnames (behavioral_predictors, lag, dict = True):
	# get behavioral variables colnames with time lag
	columns = []
	lagged_columns = []

	if dict:
		for item in behavioral_predictors. keys ():
			items = behavioral_predictors[item]
			columns. extend (items)
	else:
		columns = behavioral_predictors

	for item in columns:
		lagged_columns. extend ([item + "_t%d"%(p) for p in range (lag, 2, -1)])
		#lagged_columns. extend ([item + "_sum"])
		#lagged_columns. extend ([item + "_mean", item + "_std"])

	return (lagged_columns)

#=============================================
# generate time series from transcriptions files
def process_transcriptions (subject, type = "speech_ts"):
    files = glob. glob ("time_series/%s/%s/*.pkl"%(subject, type))
    return sorted (files)

#=============================================
def list_files (subject, type = "speech_ts"):

    files = []
    if subject == 'sub-01':
        blocks = range (1, 5)
    else:
        blocks = range (1, 5)

    for block in blocks:
        for conv in [1, 2, 3, 4, 5, 6]:
            conv_type = conv % 2
            if conv_type == 0:
                conv_type = 2
            files. append ('time_series/%s/%s/convers-TestBlocks%d_CONV%d_00%d.pkl'%(subject, type, block, conv_type, conv))

    return sorted (files)

#===============================================
def reorganize_data (data_, lag, step = 1):
	"""
	    step: ratio between frequencies compared o bold ferquency (1.205 s)
	    lag: lag
	"""

	out_data = []

	# first point 0.6s correspont to the first BOLD acquisition
	real_lag = lag - 2

	for i in range (lag, len (data_), step):
		row = []
		for j in range (data_. shape [1]):
			row = row + list (data_ [i - lag : i -  lag + real_lag, j]. flatten ())
			#row = row +[np. mean (data_ [i - lag : i -  lag + real_lag, j]), np.std (data_ [i - lag : i -  lag + real_lag, j])]

		out_data. append (row)

	return np. array (out_data)

#===============================================
def get_unimodal_ts (subject, type, lag, bold = False):
	# do not consider the first columns (the time index)
	files = list_files (subject, type)
	# initilization
	HH_data = []
	HR_data = []

	# get colnames, and excluding first column, which is time
	colnames = list (pd. read_pickle (files[0]).iloc[:, 1:]. columns)
	lagged_columns = []

	#if type not in ["physio_ts", "discretized_physio_ts"]:
	if not bold:
		lagged_columns = get_lagged_colnames (colnames, lag, dict = False)
	else:
		lagged_columns = colnames

	# concatenate data of all conversations
	for filename in files:
		# Extract behavioral variables without the first column, which is time.
		data = pd. read_pickle (filename).iloc[:, 1:]. values

		# neuro_physio data must contain 52 observations for each conversation
		if type in ["physio_ts"]:
			if data. shape[0] < 53:
				print ("Not enouph bold observations in %s"%filename)
				exit (1)
			lagged_ts = data [lag:,]
			#lagged_ts = data

		else:
			if data. shape[0] < 50:
				print ("Not enouph bold observations in %s"%filename)
				exit (1)

			# apend last 4 obsevations in the end to have the same length as the bold signal
			data = np. append (data, data[-3:,:], axis = 0)

			'''zero_row = [0 for a in range (data.shape[1])]
			for i in range (lag):
				data = np.insert (data, 0, zero_row, axis = 0)'''

			lagged_ts = reorganize_data (data, lag = lag)

		if "CONV1" in filename:
			if len (HH_data) == 0:
			    HH_data = lagged_ts
			else:
			    HH_data = np. concatenate ((HH_data, lagged_ts), axis = 0)

		if "CONV2" in filename:
			if len (HR_data) == 0:
			    HR_data = lagged_ts
			else:
				if HR_data.shape[1] != lagged_ts. shape[1]:
					print ("Error in file %s"%filename)
					exit (1)
				HR_data = np. concatenate ((HR_data, lagged_ts), axis = 0)

	return [pd.DataFrame (HH_data, columns = lagged_columns), pd.DataFrame (HR_data, columns = lagged_columns)]

#=============================================
def get_behavior_ts_one_subject (subject, behaviours, lag):
    subj_behavioral = get_unimodal_ts (subject,  behaviours [0], lag)
    for type in behaviours[1:]:
        unimodal_data = get_unimodal_ts (subject, type, lag)
        subj_behavioral[0] = pd. concat ([subj_behavioral [0] , unimodal_data [0]], axis = 1)
        subj_behavioral[1] = pd. concat ([subj_behavioral [1], unimodal_data [1]], axis = 1)
    # return two outputs: for human-human and human-robot
    return subj_behavioral
#=============================================s'imprégner
if __name__ == '__main__':

	parser = argparse. ArgumentParser ()
	parser. add_argument ("--lag", "-p", help = "lag parameter", type = int, default = 6)
	args = parser.parse_args()

	if not os.path.exists ("concat_time_series"):
	    os.makedirs ("concat_time_series")

	# subjects to process
	subject_exceptions = ["sub-14"]
	#subjects = ["sub-02"]
	subjects = ["sub-%02d"%i for i in range (1, 26)]
	for sub in subject_exceptions:
	    if sub in subjects:
	        subjects. remove (sub)

	# 3 types of data to process
	behavioral_hh_data = pd. DataFrame ()
	behavioral_hr_data = pd. DataFrame ()

	bold_hh_data = pd. DataFrame ()
	bold_hr_data = pd. DataFrame ()

	disc_bold_hh_data = pd. DataFrame ()
	disc_bold_hr_data = pd. DataFrame ()

	#subjs_info = pd.read_csv ("data/participants_info.txt", sep = '\t')

	# Concatenate data of the subjects
	for subject in subjects:

	    print (subject, "\n", 18*'-')

	    subj_behavioral = get_behavior_ts_one_subject (subject, ["speech_left_ts", "speech_ts","smiles_ts", "dlib_smiles_ts", "facial_features_ts", "emotions_ts", "eyetracking_ts"], args.lag) #, ,
	    subj_bold = get_unimodal_ts (subject, "physio_ts", args.lag, True)
	    #subj_disc_bold = get_unimodal_ts (subject, "discretized_physio_ts", args.lag, True)

	    behavioral_hh_data = behavioral_hh_data. append (subj_behavioral[0], ignore_index=True, sort=False)
	    behavioral_hr_data = behavioral_hr_data. append (subj_behavioral[1], ignore_index=True, sort=False)

	    bold_hh_data = bold_hh_data. append (subj_bold [0], ignore_index=True, sort=False)
	    bold_hr_data = bold_hr_data. append (subj_bold [1], ignore_index=True, sort=False)

	# Replace nan and very small values with 0
	'''behavioral_hh_data. fillna (0, inplace = True)
	behavioral_hh_data [behavioral_hh_data < 0.00001] = 0

	behavioral_hr_data. fillna (0, inplace = True)
	behavioral_hr_data [behavioral_hr_data < 0.00001] = 0'''



	cols_hh = behavioral_hh_data. columns
	cols_hr = behavioral_hr_data. columns

	min_max_scaler = preprocessing. MinMaxScaler ()
	behavioral_hh_data = min_max_scaler. fit_transform (behavioral_hh_data. values)
	behavioral_hr_data = min_max_scaler. fit_transform (behavioral_hr_data. values)

	behavioral_hh_data = pd.DataFrame (behavioral_hh_data, columns = cols_hh)
	behavioral_hr_data = pd.DataFrame (behavioral_hr_data, columns = cols_hr)


	# Store data in csv files
	behavioral_hh_data. to_csv ("concat_time_series/behavioral_hh_data.csv", sep = ';', index = False)
	behavioral_hr_data. to_csv ("concat_time_series/behavioral_hr_data.csv", sep = ';', index = False)

	bold_hh_data. to_csv ("concat_time_series/bold_hh_data.csv", sep = ';', index = False)
	bold_hr_data. to_csv ("concat_time_series/bold_hr_data.csv", sep = ';', index = False)

	# Store data in pickle files
	behavioral_hh_data. to_pickle ("concat_time_series/behavioral_hh_data.pkl")
	behavioral_hr_data. to_pickle ("concat_time_series/behavioral_hr_data.pkl")

	bold_hh_data. to_pickle ("concat_time_series/bold_hh_data.pkl")
	bold_hr_data. to_pickle ("concat_time_series/bold_hr_data.pkl")


	print (behavioral_hh_data. shape)
	print (bold_hh_data. shape)

	print (behavioral_hr_data. shape)
	print (bold_hr_data. shape)
