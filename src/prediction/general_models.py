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

'''from fylearn.garules import MultimodalEvolutionaryClassifier
from fylearn.nfpc import FuzzyPatternClassifier
from fylearn.fpt import FuzzyPatternTreeTopDownClassifier'''

# local files
from src.feature_selection. reduction import manual_selection, generic_reduction
from src.prediction.tools import *
from src.prediction.training import *

# for imbalanced class
from imblearn.over_sampling import RandomOverSampler,  SMOTE, ADASYN, SMOTENC

from sklearn.model_selection import StratifiedShuffleSplit, ShuffleSplit, TimeSeriesSplit
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings("ignore")


#=================================================================================================================
global_dict_models = {
	"BAG": BaggingClassifier (),
	"MLP":RandomForestClassifier (),
	"SGD": SGDClassifier (),
	"GB": GradientBoostingClassifier (),
	"RF": RandomForestClassifier (),
	"SVM": RandomForestClassifier (),
	"FUZZY": RandomForestClassifier (),
	"RIDGE": linear_model.Ridge (),
	"LASSO": linear_model.Lasso (),
	"baseline": DummyClassifier (),
	"LREG": linear_model.LogisticRegression (),
	"ada": AdaBoostClassifier (),
	"KNN": KNeighborsClassifier ()
	}

#=================================================================================================================
def get_best_features (brain_area, model, type):
	"""
		get best features in terms of fscore results (tsv file)
	"""
	filename = "results/prediction/%s_%s.tsv"%(model, type)
	df = pd. read_csv (filename, sep = '\t')

	best_features_results = df. loc [df["region"] == brain_area]
	best_features = literal_eval (best_features_results. loc[:,"selected_predictors"]. values[0])

	return best_features

#=================================================================================================================
def train_pred_model (model, dm_method, train_data, target_column, variables_list, lag, convers_type, find_params):
	# k-fold cross validation
	if find_params and model not in ["LSTM", "FUZZY", "MLP"]:
		valid_size = int (train_data. shape [0] * 0.2)
		# k_l_fold_cross_validation to find the best parameters
		best_model_params, mean_scores, std_errors = k_l_fold_cross_validation (train_data, model, lag = lag, n_splits = 1, block_size = valid_size)

		return best_model_params, mean_scores, std_errors

	# exception for the lstm model: execute it without cross validation because of time ...
	# TODO: std errors of the training step
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
		best_model_params, std_errors = extract_models_params_from_crossv (models_params_file, target_column, variables_list, dm_method)
		# Train the model
		pred_model = train_model (train_data, model, best_model_params, lag)

	return pred_model, best_model_params, std_errors

#============================================================
def predict_area (behavioral_variables, target_column, set_of_behavioral_predictors, model, lag, filename, find_params = False, method = "None", type = "HH", all = False, all_m = False):
	"""
		- target_column:
		- set_of_behavioral_predictors:
		- lag: the lag parameter
		- model: the prediction model name
		- filename: where to put the results
		- find_params: if TRUE, a k-fold-cross-validation  is used to find the parameters of the models, else using the previous one stored.
		- method: the feature selection method. None for no feature selection, and rfe for recursive feature elimination.
		- all: if True, we use all predictive variables as input.
	"""
	#print (target_column, "\n", 18*'-')

	# load target data
	if model in ["RIDGE", "LASSO"]:
		y = pd. read_csv ("concat_time_series/bold_%s_data.csv"%str (type). lower (), sep = ';'). loc[:, [target_column]]. values

	else:
		y = pd. read_csv ("concat_time_series/discr_bold_%s_data.csv"%str (type). lower (), sep = ';'). loc[:, [target_column]]. values

	if all_m:

		#best_features = get_best_features (target_column, model, type)
		# Add the set of all variables
		#if target_column == "Hypothalamus":
		set_of_behavioral_predictors =  [{"all_ts": ["x", "y", "Contempt", "Eyes"]}]
		#else:
		#set_of_behavioral_predictors =  [{"eyetracking_ts": ["Vx", "Vy", "saccades", "Face", "Mouth", "Eyes"] }]

	if all:
		# Add the set of all variables
		set_of_behavioral_predictors =  [{"speech_left_ts": ["IPU_left", "disc_IPU_left", "Overlap_left", "ReactionTime_left", "FilledBreaks_left", "Feedbacks_left",
										"Discourses_left", "Particles_left", "Laughters_left", "LexicalRichness1_left", "LexicalRichness2_left", "Polarity_left",
										"Subjectivity_left", "UnionSocioItems_left"], \

										"speech_ts": [ "IPU", "disc_IPU", "Overlap", "ReactionTime", "FilledBreaks", "Feedbacks", "Discourses",
													"Particles", "Laughters", "LexicalRichness1", "LexicalRichness2", "Polarity", "Subjectivity", "UnionSocioItems"], \

										"dlib_smiles_ts": ["dlib_smiles"], \

										"energy_ts": ["mouth_AU",  "eyes_AU", "total_AU", "head_rotation_energy", "head_translation_energy", "head_positions"], \

										"eyetracking_ts": ["Vx", "Vy", "saccades", "Face", "Mouth", "Eyes"] }]

	# if the model the baseline (random), using multiple behavioral predictors has no effect
	if model == "baseline":
		set_of_behavioral_predictors = set_of_behavioral_predictors [0:1]
		method = "None"

	results = []
	#for nb_test_exps in range (4, 5):
	nb_test_exps = 4
	nb_predictions = int (0.2 * len (behavioral_variables))
	test_index =  range (nb_test_exps * nb_predictions, (nb_test_exps + 1) * nb_predictions)

	train_index = []

	for i in range (len (behavioral_variables)):
		if i not in test_index:
			train_index. append (i)

	#print (list (set (train_index) & set (test_index)))

	X_train_all = behavioral_variables. iloc [train_index, :].values
	X_test_all =  behavioral_variables. iloc [test_index, :].values


	min_max_scaler = preprocessing. MinMaxScaler ()
	X_train_all = min_max_scaler. fit_transform (X_train_all)
	X_test_all = min_max_scaler. transform (X_test_all)


	y_train = y [train_index, :]. copy (). flatten ()
	y_test =  y [test_index, :].  copy (). flatten ()

	#n_original = len (X_train_all)
	#X_train_all, X_test_all, y_train, y_test = train_test_split (behavioral_variables. values, y, test_size=0.19, random_state=42)

	#ros = RandomOverSampler(random_state=5)
	if find_params == False:
		ros = ADASYN (random_state=5)
		#ros = SMOTE (random_state=5)
		# Handling oversampling
		try:
			X_train_all, y_train =  ros. fit_resample (X_train_all, y_train)
			#X_test_all, y_test   =   RandomOverSampler (random_state=5). fit_resample (X_test_all, y_test)

		except:
			print ("Data already balanced!")
			pass

	#print ("Ratio of synthetic data: ", len (X_train_all), n_original)
	X_train_all = pd. DataFrame (X_train_all, columns = behavioral_variables. columns)
	X_test_all = pd. DataFrame (X_test_all, columns = behavioral_variables. columns)

	if model == "baseline":
		set_of_behavioral_predictors = set_of_behavioral_predictors [0:1]
	# A loop to test each set of predictive features
	for behavioral_predictors in set_of_behavioral_predictors:

		score = []
		lagged_names = get_lagged_colnames (behavioral_predictors, lag)

		'''if all_m:
			for x in lagged_names:
				if x[0:-3] in str (best_features):
					lagged_names. remove (x)'''

		selected_indices = [a for a in range (len (lagged_names))]

		X_train = X_train_all. loc[:, lagged_names]. values
		X_test = X_test_all. loc[:, lagged_names]. values

		# test multiple number of features for feature selection_
		if model == "baseline" or method in ["None", "", "TREE_ALL"]:
			set_k = [X_train. shape [1]]

		else:
			set_k = list (range (2, 21, 2))
			#set_k = [12, 16, 20]

		for n_comp in set_k:
			print ("%s K = %s ----------"%(method, n_comp))
			score = []

			if method in ["None", ""]:
				dm_method = method
			else:
				dm_method = "%s_%s"%(method, str (n_comp))

			if n_comp > X_train. shape [1]:
				continue

			if method == "TREE_ALL" and model != "baseline":
				selected_indices = []
				if model == "KNN":
					clf = global_dict_models ["RF"]
				else:
					clf = global_dict_models [model]
				clf = clf.fit (X_train, y_train)
				select_model = SelectFromModel (clf, prefit = True)
				support = select_model. get_support()

				for i in range (len (support)):
					if support [i]:
						selected_indices. append (i)

				# transform data
				X_train_select = select_model.transform (X_train)
				X_test_select = select_model.transform (X_test)
			# feature selection
			elif method not in ["None", ""] and model != "baseline" and n_comp < (X_train. shape [1]):
				X_train_select, selected_indices, selector = generic_reduction (X_train, y_train, method = method, n_comps = n_comp, estimator_name = "RF")
				if method in ["PCA", "KPCA", "ICA", "TREE"]:
					X_test_select = selector. transform (X_test)
				else:
					X_test_select = X_test[:, [int(a) for a in selected_indices]]

			elif model == "baseline" or method == "None":
				selected_indices = [a for a in range (len (lagged_names))]
				X_train_select = X_train [:]
				X_test_select = X_test [:]

			# train the model and extract the best model parameters from the  k_fold_cross_validation experiment
			train_data = np. concatenate ((y_train. reshape ((-1,1)), X_train_select), axis = 1)

			if find_params:
				best_model_params, evaluations, std_errors = train_pred_model (model, method, train_data, target_column, lagged_names, lag, type, find_params)

			else:
				pred_model, best_model_params, std_errors = train_pred_model (model, method, train_data, target_column, lagged_names, lag, type, find_params)
				evaluations = test_model (X_test_select, y_test, pred_model, lag, model)

			#print (best_model_params)
			#results. append ([target_column,  dm_method, lag, str (best_model_params), str (dict (behavioral_predictors)), lagged_names, str ([lagged_names [i] for i in selected_indices])] + evaluations + std_errors)

			if all:
				str_dict = 'all'

			elif all_m:
				str_dict = "all_minus_best"
			else:
				str_dict = str (dict (behavioral_predictors))

			selected_list = str ([lagged_names [i] for i in selected_indices])

			row = [target_column,  dm_method, lag, str (best_model_params), str_dict, lagged_names, selected_list] + evaluations + std_errors
			write_line (filename, row, mode = "a")


#=========================================================================
def predict_all (_regions, lag, k, model, remove, _find_params, all, all_m):

	print ("-- MODEL :", model)
	colnames = ["region", "dm_method", "lag", "models_params", "predictors_dict",   "predictors_list", "selected_predictors",
				"recall. mean",  "precision. mean", "fscore. mean",  "kappa. mean",  "recall. std",  "precision. std",  "fscore. std",  "kappa. std"]

	if _find_params:
		filename_hh = "results/models_params/%s_HH.csv"%(model)
		filename_hr = "results/models_params/%s_HR.csv"%(model)
	else:
		filename_hh = "results/prediction/%s_HH.csv"%(model)
		filename_hr = "results/prediction/%s_HR.csv"%(model)

	'''if model == "baseline":
		_find_params = True'''

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


	if behavioral_variables_hh. isnull (). any (). any () or behavioral_variables_hr. isnull (). any (). any ():
		print (" ERROR: data still contains nan after nan values elimination!")

	# Predict HH  and HR conversations separetely
	Parallel (n_jobs = 1) (delayed (predict_area)
	(behavioral_variables_hh, target_column, manual_selection (target_column), model = model, lag = lag, filename = filename_hh, find_params = _find_params, type = "HH", all = all, all_m = all_m)
									for target_column in _regions)


	Parallel (n_jobs = 1) (delayed (predict_area)
	(behavioral_variables_hr, target_column, manual_selection (target_column), model = model, lag = lag, filename = filename_hr, find_params = _find_params, type = "HR", all = all, all_m = all_m)
									for target_column in _regions)

if __name__ == '__main__':
	print ("test")
