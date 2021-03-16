import os, glob, argparse, inspect, sys
import pandas as pd
import numpy as np
from sklearn import preprocessing
import pickle

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(3,parentdir)

from src.normalizer import normalizer

#----------------------------------------------------------------#
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

	return (lagged_columns)

#=============================================
# generate time series from transcriptions files
def process_transcriptions (subject, type = "speech_ts"):
	files = glob. glob ("time_series/%s/%s/*.pkl"%(subject, type))
	return sorted (files)


#=============================================

def save_files (behavioral_hh_data, behavioral_hr_data, bold_hh_data, bold_hr_data):

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
def get_unimodal_ts (subject, type, lag, bold = False, add_time = False):
	files = list_files (subject, type)

	# initilization
	HH_data = []
	HR_data = []
	lagged_columns = []

	# include first colmun (normalized time index) as feature
	if add_time:
		colnames = list (pd. read_pickle (files[0]). columns)
	else:
		# get colnames, and excluding first column, which is time
		colnames = list (pd. read_pickle (files[0]).iloc[:, 1:]. columns)

	#if type not in ["physio_ts", "discretized_physio_ts"]:
	if not bold:
		lagged_columns = get_lagged_colnames (colnames, lag, dict = False)
	else:
		lagged_columns = colnames

	# concatenate data of all conversations
	for filename in files:
		if add_time:
			data = pd. read_pickle (filename). values
			max_time_conversation = np. max (data[:,0])
			min_time_conversation = np. min (data[:,0])

			for i in range (len (data)):
				data[i,0] = (data[i,0] - min_time_conversation) / float (max_time_conversation - min_time_conversation)

		else:
			data = pd. read_pickle (filename).iloc[:, 1:]. values

		# neuro_physio data must contain 52 observations for each conversation
		if type in ["physio_ts"]:
			if data. shape[0] < 53:
				print ("Not enough bold observations in %s"%filename)
				exit (1)
			lagged_ts = data [lag:,]

		else:
			if data. shape[0] < 50:
				print (filename)
				print ("Warning, Not enouph bold observations in %s, lines will be added"%filename)
				#exit (1)
				while  data. shape[0] < 50:
					data = np. insert (data, 0, data[-1,:], axis = 0)


			# apend last 3 obsevations in the end to have the same length as the bold signal
			#data = np.diff (data, axis = 0)
			#data = np. insert (data, 0, [0 for j in range (data. shape[1])], axis = 0)

			data = np. append (data, data[-3:,:], axis = 0)

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
	unimodal_data_first_conv = get_unimodal_ts (subject,  behaviours [0], lag, add_time = False)
	subj_behavioral = []
	subj_behavioral. append (unimodal_data_first_conv[0])
	subj_behavioral. append (unimodal_data_first_conv[1])
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
	#subjects = ["sub-10"]
	#subjects = ["sub-%02d"%i for i in range (9, 10)]
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
		subj_behavioral = get_behavior_ts_one_subject (subject, ["speech_left_ts", "speech_ts", \
		 														"dlib_smiles_ts", "facial_features_ts", "emotions_ts", "eyetracking_ts"], args.lag)
		#subj_behavioral = get_behavior_ts_one_subject (subject, ["openface_features_ts"], args.lag)
		#subj_behavioral = get_behavior_ts_one_subject (subject, ["facial_features_ts"], args.lag)
		#subj_behavioral = get_behavior_ts_one_subject (subject, ["mfcc_features_ts"], args.lag)

		subj_bold = get_unimodal_ts (subject, "physio_ts", args.lag, True)

		# Initialize the normaliers
		min_max_scaler_bold_hh = normalizer (subj_bold[0])
		min_max_scaler_bold_hr = normalizer (subj_bold[1])

		# Normalize the data
		subj_bold[0] = min_max_scaler_bold_hh. transform (subj_bold[0])
		subj_bold[1] = min_max_scaler_bold_hr. transform (subj_bold[1])

		behavioral_hh_data = behavioral_hh_data. append (subj_behavioral[0], ignore_index=True, sort=False)
		behavioral_hr_data = behavioral_hr_data. append (subj_behavioral[1], ignore_index=True, sort=False)

		bold_hh_data = bold_hh_data. append (subj_bold [0], ignore_index=True, sort=False)
		bold_hr_data = bold_hr_data. append (subj_bold [1], ignore_index=True, sort=False)

	# Replace nans with 0
	behavioral_hh_data. fillna (method = "ffill", inplace = True)
	behavioral_hr_data. fillna (method = "ffill", inplace = True)

	# removing columns with low standart deviation and low distinct values
	behavioral_hr_data = behavioral_hr_data. round (5)
	behavioral_hh_data = behavioral_hh_data. round (5)
	min_std = 0.000001
	max_perc_cst = 0.95

	'''to_remove = []
	for j in range (behavioral_hh_data. shape [1]):
		if behavioral_hh_data. iloc [:, j]. std() < min_std:
			to_remove. append (j)

		unique, counts = np.unique (behavioral_hh_data. iloc [:,j]. values, return_counts=True)
		perc_cst_values = max (counts) / len (behavioral_hh_data)
		if (perc_cst_values > max_perc_cst):
			to_remove. append (j)

	#print (behavioral_hh_data. columns [to_remove])
	behavioral_hh_data. drop (behavioral_hh_data. columns [to_remove], axis = 1, inplace = True)'''

	print (18 * '-')
	# removing columns with low standart deviation
	'''to_remove = []
	for j in range (behavioral_hr_data. shape [1]):
		if behavioral_hr_data. iloc [:, j]. std() < min_std:
			to_remove. append (j)

		unique, counts = np.unique (behavioral_hr_data. iloc [:,j]. values, return_counts=True)
		perc_cst_values = max (counts) / len (behavioral_hr_data)
		if (perc_cst_values > max_perc_cst):
			to_remove. append (j)

	#print (behavioral_hr_data. columns [to_remove])
	behavioral_hr_data. drop (behavioral_hr_data. columns [to_remove], axis = 1, inplace = True)'''

	# Initilize the normaliers
	min_max_scaler_hh = normalizer (behavioral_hh_data)
	min_max_scaler_hr = normalizer (behavioral_hr_data)

	# Normalize the data
	behavioral_hh_data = min_max_scaler_hh. transform (behavioral_hh_data)
	behavioral_hr_data = min_max_scaler_hr. transform (behavioral_hr_data)

	# save the models
	min_max_scaler_hh. save ("trained_models/min_max_scaler_hh")
	min_max_scaler_hr. save ("trained_models/min_max_scaler_hr")

	f = open('brain_areas.tsv', 'a')

	# Make brain areas names short
	short_bold_columns = list (bold_hh_data. columns)
	for i  in range (len (short_bold_columns)):
		f.write('%s\t'%short_bold_columns[i])
		short_bold_columns[i] = short_bold_columns[i]. replace ('atlas.', '')
		short_bold_columns[i] = short_bold_columns[i]. replace ('(', '')
		short_bold_columns[i] = short_bold_columns[i]. replace (')', '')
		short_bold_columns[i] = '_'.join (short_bold_columns[i]. split (' ')[0:-1])
		if short_bold_columns[i][-1] == '_':
			short_bold_columns[i] = short_bold_columns[i][:-1]
		f.write('%s\t'%short_bold_columns[i])
		f.write('%d\n'%(i+1))

	f.close()
	bold_hh_data. columns = short_bold_columns
	bold_hr_data. columns = short_bold_columns

	for a in short_bold_columns:
		print (a)
	save_files (behavioral_hh_data, behavioral_hr_data, bold_hh_data, bold_hr_data)
