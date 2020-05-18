import itertools, glob

import numpy as np
import pandas as pd
import random as rd

from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier, GradientBoostingClassifier
from sklearn import linear_model
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold
from ast import literal_eval

from sklearn.metrics import recall_score, precision_score, f1_score, average_precision_score, accuracy_score

from scipy.signal import find_peaks
from src. prediction. lstm import *
from imblearn.over_sampling import ADASYN

#===========================================
global_dict_models = {
		"BAG": BaggingClassifier (),
		"SGD": SGDClassifier (),
		"GB": GradientBoostingClassifier (),
		"RF": RandomForestClassifier (),
		"SVM": SVC (),
		"RIDGE": linear_model.Ridge (),
		"LASSO": linear_model.Lasso (),
		"baseline": DummyClassifier (),
		"LREG": linear_model.LogisticRegression (),
		"ada": AdaBoostClassifier (),
		"KNN": KNeighborsClassifier ()
		}

#===========================================
def find_peaks_ (y, height = 0):
	x = []
	for i in range (len (y)):
		if y[i] >= height:
			x.append (i)
	return x

#===========================================
def discretize_preds (x):

	disc_file = pd. read_csv ("disc_params.txt", sep = ':', header = None, index_col = 0)
	height = float (disc_file. loc ["threshold"]. values [0])

	result = [i for i in x]
	for i in range (len (result)):
		if result [i] >= height:
			result [i] = 1
		else:
			result [i] = 0

	return result

#===========================================
# generate tuples from two lists
def generate_tuples (a, b):
	c = [[x , y] for x in a for y in b]
	return c

#===========================================
def generate_models (params):

	combinations = itertools. product (* (params[Name] for Name in params. keys ()))
	keys = list (params. keys ())
	res = []
	for combin in list (combinations):
		dict = {}
		for i in range (len (keys)):
			dict[keys[i]] = combin [i]
		res. append (dict)

	return res

#===========================================
#---- get items from dict as string
def get_items (predictors, external_predictors):
	if predictors != None:
		external_variables = []
		for key in external_predictors. keys ():
			external_variables += external_predictors[key]
		variables = '+'.join (predictors + external_variables)
	else:
		variables = '+'.join (predictors)
	return (variables)

#===========================================

def get_max_of_list (data):
	best_line = data [0]
	best_index = 0
	i = 1
	for line in data[1:]:
		""" select best model based on the fscore """
		if line[1][2] > best_line [1][2]:
			best_index = i
		i += 1

	return best_index

#===========================================
def k_fold_cross_validation (data, model, lag, params, block_size):
	scores = []

	kf_obj = KFold (n_splits = 10, shuffle = False)
	splits = kf_obj. split (data)

	for train_index, test_index in splits:

		X_test = data [test_index,1:]. copy ()
		y_test = data [test_index,0]. copy ()

		X_train =  data [train_index, 1:]. copy ()
		y_train =  data [train_index, 0]. copy ()

		# Handling oversampling
		ros = ADASYN (random_state=5)
		try:
			X_train, y_train =  ros. fit_resample (X_train, y_train)
		except:
			pass

		pred_model = train_model (np. concatenate ((y_train. reshape (-1,1), X_train), axis = 1), model, params, lag)
		scores. append (test_model (X_test, y_test, pred_model))

	return (np. mean (np. array (scores), axis = 0). tolist (), np. std (np. array (scores), axis = 0). tolist (), np. array (scores)[:,0]. flatten ())

#===========================================

def k_l_fold_cross_validation (data, model, lag, n_splits, block_size):

	if model == "BAG":
		models_params = generate_models ({'bootstrap': [False], 'n_estimators': [10, 50, 100, 200, 300], 'random_state': [5]})

	elif model == "GB":
		models_params = generate_models ({'learning_rate': [0.1, 0.2, 0.3, 0.4], 'n_estimators': [10, 50, 100], 'max_depth' : [5, 10, 50, 100]})

	elif model == "KNN":
		models_params = generate_models ({'n_neighbors':[3, 4, 5]})

	elif model == "ada":
		models_params = generate_models ({'n_estimators': [10, 50, 100, 500, 1000], 'learning_rate': [1.0, 0.9, 0.8]})

	elif model == "LREG":
		models_params = generate_models ({'C': [1, 0.9, 0.8, 0.7], 'solver': ['lbfgs', 'liblinear']})

	elif model == "SVM":
		models_params = generate_models ({'C': [1, 0.9, 0.8, 0.7], 'kernel': ['linear', "rbf", "sigmoid"], 'random_state': [5]})

	elif model == "LSTM":
		models_params = generate_models ({'epochs': [20],  'neurons' : [30]})

	elif model in ["RIDGE", "Ridge", "LASSO", "Lasso"]:
		models_params = generate_models ({'alpha': [0.01, 0.1, 0.2, 0.3, 0.4, 0.5]})

	elif model == "SGD":
		models_params = generate_models ({'alpha': [0.001, 0.01, 0.05, 0.1, 0.2], 'loss': ["hinge"], 'penalty': ["l2", "l1"], 'max_iter': [20]})

	elif model == "RF":
		models_params = generate_models ({'bootstrap': [True], 'max_depth': [10, 50, 100, 500], 'max_features': ['auto'], 'n_estimators': [10, 50, 100, 200, 300], 'random_state': [5]})

	elif model == "baseline":
		models_params = generate_models ({'strategy': ['uniform', 'stratified', "most_frequent"], 'random_state': [5]})

	'''if model in  ["LSTM"]:
		# Train
		pred_model = train_model (data, model, models_params [0], lag)
		return str (models_params [0]), pred_model'''

	models_results = []

	# fing the best model parameters
	for params in models_params:
		results_mean, results_std, k_fscores = k_fold_cross_validation (data, model = model, lag = lag, params = params, block_size = block_size)
		models_results. append ([str (params), results_mean, results_std, k_fscores])

	# Get the best model parameters
	best_index = get_max_of_list (models_results)

	return models_results [best_index]

#======================================================
def  train_model (data, model, params, lag):

	if model == "LSTM":
		pred_model = LSTM_MODEL (4)
		pred_model. fit (data [:, 1:], data[:, 0])
	else:
		pred_model = global_dict_models [model]
		pred_model. set_params (**params )
		pred_model. fit (data[:,1:], data[:,0])

	return pred_model

#==============================================================#
def  test_model (X, Y, model):

	pred = model. predict (X)
	real = Y

	recall_ 	= recall_score (real, pred, average = 'weighted')
	precision_ 	= precision_score (real, pred, average = 'weighted')
	fscore_ 	= f1_score (real, pred, average = 'weighted')

	return [recall_, precision_, fscore_]

#===========================================================
def extract_models_params_from_crossv (crossv_results_filename, brain_area, features, reduction_method):
	"""
		- extract  parameters of the mode from cross-validation results

		- crossv_results_filename: the filename of the model where the results are saved.
		- brain_area: brain region name
		- features: the set of predictive features (lagged)
		- dictionary containing the parameter of the model
	"""

	features_exist_in_models_params = False
	models_params_ = pd.read_csv (crossv_results_filename, sep = ';', header = 0, na_filter = False, index_col = False)
	models_params = models_params_. loc [(models_params_ ["region"] ==  brain_area)]

	# if the brain_area was not processed with cross-validation
	if models_params. shape [0] == 0:
		best_model_params_index = models_params_ ["fscore. mean"].idxmax ()
		best_model_params = models_params_. loc [best_model_params_index, "models_params"]
		std_errors =  models_params. loc [best_model_params_index, ["recall. std",  "precision. std",  "fscore. std"]]
		return literal_eval (best_model_params), std_errorss

	# find the models_paras associated to each predictors_list with dimension reduction method
	for i in list (models_params. index):
		if set (literal_eval(models_params. loc [i, "predictors_list"])) == set (features) and models_params. loc [i, "dm_method"] == reduction_method:
			features_exist_in_models_params = True
			best_model_params_index = i
			break

	# find the models_paras associated to each predictors_list without dimension reduction method
	if not features_exist_in_models_params:
		for i in list (models_params. index):
			if set (literal_eval(models_params. loc [i, "predictors_list"])) == set (features):
				features_exist_in_models_params = True
				best_model_params_index = i
				break

	# else, choose the best model_params without considering features
	if not features_exist_in_models_params:
		best_model_params_index = models_params ["fscore. mean"].idxmax ()


	best_model_params = models_params. loc [best_model_params_index, "models_params"]
	std_errors =  models_params. loc [best_model_params_index, ["recall. std",  "precision. std",  "fscore. std"]]. values

	return literal_eval (best_model_params), std_errors. tolist ()

#=================================================================================================================
def train_pred_model (model, dm_method, train_data, target_column, variables_list, lag, convers_type, find_params):
	# k-fold cross validation
	if find_params and model not in ["LSTM", "FUZZY", "MLP"]:
		valid_size = int (train_data. shape [0] * 0.2)
		# k_l_fold_cross_validation to find the best parameters
		best_model_params, mean_scores, std_errors, k_fscores = k_l_fold_cross_validation (train_data, model, lag = lag, n_splits = 1, block_size = valid_size)

		return best_model_params, mean_scores, std_errors, k_fscores, []

	# exception for the lstm model: execute it without cross validation because of time ...
	# TODO: std errors of the training step
	elif model == 'LSTM':
		best_model_params =  {'epochs': [20],  'neurons' : [30]}
		pred_model = train_model (train_data, model, best_model_params, lag)
		std_errors = [0, 0, 0]
		features_importance = []

	elif model == 'MLP':
		nb_neurons = max (1, int (train_data. shape[1] / 2))
		pred_model = MLPClassifier (hidden_layer_sizes= [nb_neurons], shuffle = False, activation='logistic')
		pred_model. fit (train_data[:, 1:], train_data[:, 0])
		best_model_params = {}
		std_errors = [0, 0, 0]
		features_importance = []

	elif model == 'FUZZY':
		#pred_model = MultimodalEvolutionaryClassifier()
		pred_model = FuzzyPatternTreeTopDownClassifier ()
		pred_model. fit (train_data[:, 1:], train_data[:, 0])
		best_model_params = {}
		features_importance = []

	else:
		# extract model params from the previous k-fold-validation results
		models_params_file = glob. glob ("results/models_params/*%s_%s.csv" %(model, convers_type. upper ()))[0]
		best_model_params, std_errors = extract_models_params_from_crossv (models_params_file, target_column, variables_list, dm_method)
		# Train the model
		pred_model = train_model (train_data, model, best_model_params, lag)
		features_importance = pred_model.feature_importances_

	return pred_model, best_model_params, std_errors, features_importance
