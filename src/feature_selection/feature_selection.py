import numpy as np
import pandas as pd
import os
from src. feature_selection. reduction import rfe_reduction, manual_selection
from src.prediction.tools import *
from joblib import Parallel, delayed
from sklearn.model_selection import StratifiedShuffleSplit, ShuffleSplit

#============================================================

def feature_selection (subjects, target_column, set_of_behavioral_predictors, convers, lag, filename):

	if (int (convers[0]. split ('_')[-1]) % 2 == 1): convers_type = "HH"
	else : convers_type = "HR"

	for behavioral_predictors in set_of_behavioral_predictors:

		# concatenate data of all subjects  with regards to the behavioral variables
		all_data = concat_ (subjects[0], target_column, convers, lag, behavioral_predictors, add_target = False)
		for subject in subjects[1:]:
			subject_data = concat_ (subject, target_column, convers, lag, behavioral_predictors, add_target = False)
			all_data = np.concatenate ((all_data, subject_data), axis = 0)

		""" generate the random order to shuffle data for feature selection and prediction """
		#np.random.seed (5)
		#np.random.shuffle (all_data)

		""" split the data into test and training sets with stratified way """
		sss = ShuffleSplit (n_splits = 1, test_size = 0.2, random_state = 5)
		for train_index, test_index in sss.split (all_data[:, 1:], all_data [:, 0]):
			train_data = all_data [train_index]
			test_data = all_data [test_index]
			break

		lagged_names = get_lagged_colnames (behavioral_predictors, lag)

		# split the data in test and training sets
		train_data_, test_data_ = train_test_split (all_data, test_size = 0.2)

		if all_data. shape == 2:
			write_line (filename, [target_column, lagged_names, [i for i in range (len (lagged_names))], 1], mode = "a+")

		else:
			# Feature selection
			#train_data, test_data,
			features_indices, score = rfe_reduction (train_data_, test_data_, [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])

			write_line (filename, [target_column, lagged_names, features_indices, score], mode = "a+")

#====================================================================#

def process_region (target_column, convers, subjects, lag, filename):

	print ("	Region %s" %target_column)
	#num_region = int (target_column. split ('_')[-1])
	# Find the best parameters of the model using the cross-validation
	behavioral_predictors_ = manual_selection (target_column)
	feature_selection (subjects, target_column, behavioral_predictors_, convers, lag, filename)

#======================================================================

def  select_features (subjects, _regions, lag, remove):

	print ("\n	... FEATURE SELECTION")

	colnames = ["region", "features", "selected_features", "score"]

	subjects_str = "subject"
	for subj in subjects:
		subjects_str += "_%s"%subj

	filename_hh = "results/selection/selection_HH.csv"
	filename_hr = "results/selection/selection_HR.csv"

	for filename in [filename_hh, filename_hr]:
		# remove previous output files ir remove == true
		if remove:
			os. system ("rm %s"%filename)

		if not os.path.exists (filename):
			f = open (filename, "w+")
			f.write (';'. join (colnames))
			f. write ('\n')
			f. close ()

	#_regions = ["region_%d"%i for i in regions]
	subjects = ["sub-%02d"%i for i in subjects]

	# fulfill HH and HR conversations
	convers = list_convers ()
	hh_convers = []
	hr_convers = []

	for i in range (len (convers)):
		if i % 2 == 1:
			hr_convers. append (convers[i])
		else:
			hh_convers. append (convers[i])


	Parallel (n_jobs = 5) (delayed(process_region) (target_column, hh_convers, subjects, int (lag), filename_hh) for target_column in _regions)
	Parallel (n_jobs = 5) (delayed(process_region) (target_column, hr_convers, subjects, int (lag), filename_hr) for target_column in _regions)

	print ("\n 	DONE ...")
