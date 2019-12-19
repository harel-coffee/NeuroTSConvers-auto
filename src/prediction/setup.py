# -*- coding: utf-8 -*-
# Author: Youssef Hmamouche

import sys, os, inspect, glob
import itertools

from sklearn import preprocessing
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.utils import shuffle
from sklearn.svm import OneClassSVM
from sklearn.neural_network import MLPClassifier

from joblib import Parallel, delayed


from fylearn.garules import MultimodalEvolutionaryClassifier
from fylearn.nfpc import FuzzyPatternClassifier
from fylearn.fpt import FuzzyPatternTreeTopDownClassifier

# local files
from src.feature_selection. reduction import manual_selection, generic_reduction, feature_selection_modalities
#from clustering import *
from src.prediction.tools import *
from src.prediction.training import *
#from src.prediction.new_model import new_model

# for imbalanced class
from imblearn.over_sampling import RandomOverSampler,  SMOTE, ADASYN

from sklearn.model_selection import StratifiedShuffleSplit, ShuffleSplit, TimeSeriesSplit


import warnings
warnings.filterwarnings("ignore")

#=================================================================================================================
def train_pred_model (model, dm_method, train_data, target_column, lagged_names, lag, convers_type, find_params):
	# k-fold cross validation
	if find_params and model not in ["LSTM", "FUZZY", "MLP"]:
		valid_size = int (train_data. shape [0] * 0.2)
		# k_l_fold_cross_validation to find the best parameters
		best_model_params, pred_model = k_l_fold_cross_validation (train_data, model, lag = lag, n_splits = 1, block_size = valid_size)

	# exception for the lstm model: execute it without cross validation because of time ...
	elif model == 'LSTM':
		best_model_params =  {'epochs': [20],  'neurons' : [30]}
		pred_model = train_model (train_data, model, best_model_params, lag)

	elif model == 'MLP':
		nb_neurons = max (1, int (train_data. shape[1] / 2))
		pred_model = MLPClassifier (hidden_layer_sizes= [nb_neurons], shuffle = False, activation='logistic')
		pred_model. fit (train_data[:, 1:], train_data[:, 0])
		best_model_params = {}

	elif model == 'FUZZY':
		#pred_model = MultimodalEvolutionaryClassifier()
		pred_model = FuzzyPatternTreeTopDownClassifier ()
		pred_model. fit (train_data[:, 1:], train_data[:, 0])
		best_model_params = {}

	else:
		# extract model params from the previous k-fold-validation results
		models_params_file = glob. glob ("results/models_params/*%s_%s.csv" %(model, convers_type. upper ()))[0]
		best_model_params = extract_models_params_from_crossv (models_params_file, target_column, lagged_names, dm_method)
		# Train the model
		pred_model = train_model (train_data, model, best_model_params, lag)

	return pred_model, best_model_params

#============================================================
def predict_area (behavioral_variables, target_column, set_of_behavioral_predictors, model, lag, filename, find_params = False, method = "RFE", type = "hh"):
	"""
		- target_column:
		- set_of_behavioral_predictors:
		- lag: the lag parameter
		- model: the prediction model name
		- filename: where to put the results
		- find_params: if TRUE, a k-fold-cross-validation  is used to find the parameters of the models, else using the previous one stored.
		- method: the feature selection method. None for no feature selection, and rfe for recursive feature elimination.
	"""

	# load target data
	if model in ["RIDGE", "LASSO"]:
		y = pd. read_csv ("concat_time_series/bold_%s_data.csv"%str (type). lower (), sep = ';'). loc[:, [target_column]]

	else:
		y = pd. read_csv ("concat_time_series/discr_bold_%s_data.csv"%str (type). lower (), sep = ';'). loc[:, [target_column]]

	# Extract the selected features from features selection results
	if os.path.exists ("results/selection/selection_%s.csv" %(type)):
		selection_results = pd.read_csv ("results/selection/selection_%s.csv" %(type),  sep = ';', header = 0, na_filter = False, index_col=False)

	if find_params:
		numb_test = 1
	else:
		numb_test = 1

	# if the model the baseline (random), using multiple behavioral predictors has no effect
	if model == "baseline":
		set_of_behavioral_predictors = set_of_behavioral_predictors [0:1]


	for behavioral_predictors in set_of_behavioral_predictors:
		score = []
		lagged_names = get_lagged_colnames (behavioral_predictors, lag)
		selected_indices = [a for a in range (len (lagged_names))]

		X = behavioral_variables. loc[:, lagged_names]

		# Concatenate target and predictive features
		all_data = np. concatenate ((y, X), axis = 1)

		all_data = outlier_detection (all_data, alpha = 0.15)
		#print (all_data. shape)

		# Determine train and test data
		nb_obs = int (all_data. shape [0] * (0.8))
		# keep subjects just for test data
		stratified_indexes  = [[range (nb_obs), range (nb_obs + 1, len (all_data))]]



		# test multiple number of features for feature selection_
		if model == "baseline" or method == "None":
			method = "None"
			set_k = [int (all_data. shape [1] - 1)]

		#set_k = list (range (1, 12))
		else:
			set_k = [2]

		for n_comp in set_k:
			print ("%s K = %s ----------"%(method, n_comp))
			score = []

			if method == "None":
				dm_method = "None"
			else:
				dm_method = "%s_%s"%(method, str (n_comp))

			if n_comp >= all_data. shape [1]:
				break

			#sss = ShuffleSplit (n_splits = 1, test_size = 0.2, random_state = 5)
			#stratified_indexes = sss.split (all_data[:, 1:], all_data [:, 0:1])
			for train_index, test_index in stratified_indexes:
				train_data = all_data [train_index, :].copy ()
				test_data = all_data [test_index, :]. copy ()

				# normalization
				min_max_scaler = preprocessing. MinMaxScaler ()
				train_data [:,1:] = min_max_scaler. fit_transform (train_data [:,1:])
				test_data [:,1:] = min_max_scaler. transform (test_data [:,1:])


				# feature selection
				if method != "None" and model != "baseline" and n_comp < (train_data. shape [1] - 1):
					train_data_X, selected_indices, selector = generic_reduction (train_data[:, 1:], train_data[:, 0:1], method = method, n_comps = n_comp, estimator_name = "RF")
					train_data = np. concatenate ((train_data[:, 0:1], train_data_X), axis = 1)
					if method in ["PCA", "KPCA", "ICA"]:
						test_data_X = selector. transform (test_data[:,1:])
					else:
						test_data_X = test_data[:, [int(a + 1) for a in selected_indices]]
						#test_data_X = selector. predict (test_data[:,1:])
					test_data = np. concatenate ((test_data[:, 0:1], test_data_X), axis = 1)
				else:
					selected_indices = [a for a in range (len (lagged_names))]

				# Naive random and  ADASYN() over-sampling
				#ros = RandomOverSampler(random_state=5)
				#X_train, y_train =  RandomOverSampler().fit_resample(train_data [:,1:], train_data [:,0])
				X_test = test_data [:,1:]
				y_test = test_data [:,0]
				#X_test, y_test =  RandomOverSampler().fit_resample(X_test, y_test)

				# train the model and extract the best model parameters from a  k_fold_cross_validation
				#train_data = np. concatenate ((y_train. reshape ((-1,1)), X_train), axis = 1)
				pred_model, best_model_params = train_pred_model (model, method, train_data, target_column, lagged_names, lag, type, find_params)

				# Compute the score on test data
				score. append (test_model (X_test, y_test, pred_model, lag, model))


			row = [target_column, dm_method, lag, best_model_params,\
			 			str (dict(behavioral_predictors)), str (lagged_names),  str ([lagged_names [i] for i in selected_indices])] \
						+ np. mean (score, axis = 0). tolist () + np. std (score, axis = 0). tolist ()

			write_line (filename, row, mode = "a")

#=========================================================================
def predict_all (subjects, _regions, lag, k, model, remove, _find_params):

	print ("-- MODEL :", model)
	colnames = ["region", "dm_method", "lag", "models_params", "predictors_dict", "predictors_list", "selected_predictors",
				"recall. mean", "precision. mean", "fscore. mean", "accuracy. mean", "recall. std", "precision. std", "fscore. std", "accuracy. std"]

	if _find_params:
		filename_hh = "results/models_params/%s_HH.csv"%(model)
		filename_hr = "results/models_params/%s_HR.csv"%(model)
	else:
		filename_hh = "results/prediction/%s_HH.csv"%(model)
		filename_hr = "results/prediction/%s_HR.csv"%(model)

	if model == "baseline":
		_find_params = True

	for filename in [filename_hh, filename_hr]:
		# remove previous output files ir remove == true
		if remove:
			os. system ("rm %s"%filename)
		if not os.path.exists (filename):
			f = open (filename, "w+")
			f.write (';'. join (colnames))
			f. write ('\n')
			f. close ()

	behavioral_variables_hh = pd. read_csv ("concat_time_series/behavioral_hh_data.csv", sep = ';')
	behavioral_variables_hr = pd. read_csv ("concat_time_series/behavioral_hr_data.csv", sep = ';')

	if behavioral_variables_hh.isnull().any().any() or behavioral_variables_hr.isnull().any().any():
		print (" ERROR: data contains nan!")
		exit (1)

	# Predict HH  and HR conversations separetely
	Parallel (n_jobs=6) (delayed (predict_area)
	(behavioral_variables_hh, target_column, manual_selection (target_column), model = model, lag = lag, filename = filename_hh, find_params = _find_params)
									for target_column in _regions)


	Parallel (n_jobs=6) (delayed (predict_area)
	(behavioral_variables_hr, target_column, manual_selection (target_column), model = model, lag = lag, filename = filename_hr, find_params = _find_params)
									for target_column in _regions)

if __name__ == '__main__':
	print ("test")
