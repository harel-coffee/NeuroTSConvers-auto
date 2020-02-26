import sys
import os

from sklearn.preprocessing import MinMaxScaler
from sklearn import linear_model
from sklearn.metrics import recall_score, precision_score, f1_score, average_precision_score, accuracy_score, cohen_kappa_score

from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectPercentile, chi2
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import linear_model
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold
import numpy as np
import pandas as pd

import random as rd
from scipy.signal import find_peaks
from src. prediction. tools import get_behavioral_data
from src. prediction. lstm import *
import itertools as it

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

	combinations = it. product (* (params[Name] for Name in params. keys ()))
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
		""" select the best fscore based on the mean if recall, precision, and fscore """
		#if np.mean (line[1]) > np. mean (best_model [1]):
		""" select best model based on the fscore """
		if line[1][2] > best_line [1][2]:
			best_index = i
		i += 1

	return best_index

#===========================================
def k_fold_cross_validation (data, model, lag, params, block_size):
	scores = []

	kf_obj = KFold (n_splits = 10)
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
		scores. append (test_model (X_test, y_test, pred_model, lag, model))

	return (np. mean (np. array (scores), axis = 0), np. std (np. array (scores), axis = 0))

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
		models_params = generate_models ({'C': [1, 0.9, 0.8, 0.7], 'kernel': ['linear', "rbf"]})

	elif model == "LSTM":
		models_params = generate_models ({'epochs': [20],  'neurons' : [30]})

	elif model in ["RIDGE", "Ridge", "LASSO", "Lasso"]:
		models_params = generate_models ({'alpha': [0.01, 0.1, 0.2, 0.3, 0.4, 0.5]})

	elif model == "SGD":
		models_params = generate_models ({'alpha': [0.001, 0.01, 0.05, 0.1, 0.2], 'loss': ["hinge"], 'penalty': ["l2", "l1"], 'max_iter': [20]})

	elif model == "RF":
		models_params = generate_models ({'bootstrap': [True], 'max_depth': [10, 50, 100, 500], 'max_features': ['auto'], 'n_estimators': [10, 50, 100, 200, 300], 'random_state': [5]})

	elif model == "baseline":
		models_params = generate_models ({'strategy': ['uniform', 'stratified', "most_frequent"], 'random_state': [None]})

	if model in  ["LSTM"]:
		# Train
		pred_model = train_model (data, model, models_params [0], lag)
		return str (models_params [0]), pred_model

	models_results = []

	# Split the data into k sets
	if data. shape[0] % n_splits == 0:
		splits = np.split (data, n_splits)
	else:
		while data. shape[0] % n_splits != 0:
			n_splits -= 1

		splits = np.split (data, n_splits)

	# fing the best model parameters
	for params in models_params:
		models_results. append ([str (params)] + [[0, 0, 0, 0]] + [[0, 0, 0, 0]])

	# k_fold_cross_validation in each split
	for split in splits:
		for i in range (len (models_params)):
			results_mean, results_std = k_fold_cross_validation (split, model = model, lag = lag, params = models_params [i], block_size = block_size)
			for l in range (3):
				models_results[i][1][l] += results_mean [l] / n_splits
				models_results[i][2][l] += results_std [l] / n_splits

	# Get the best model parameters
	best_index = get_max_of_list (models_results)

	std_errors = models_results [best_index][2]

	mean_measures = models_results [best_index][1]
	# Evaluate the best model on test data
	#pred_model = train_model (data, model, models_params [best_index], lag)

	return str (models_params [best_index]), mean_measures, std_errors

#======================================================

def  train_model (data, model, params, lag):

	if model == "LSTM":
		pred_model = fit_lstm (data [:, 1:], data[:, 0], lag = lag,  params = params)

	else:
		pred_model = global_dict_models [model]
		pred_model. set_params (**params )
		pred_model. fit (data[:,1:], data[:,0])

	return pred_model

#==============================================================#

def  test_model (X, Y, model, lag, model_type = "sickit"):

	if model_type == "LSTM":
		pred = lstm_predict (X, model, lag)
	else:
		pred = model. predict (X)

	real = Y

	if type(model).__name__ in ["Ridge", "RIDGE", "Lasso", "LinearRegression"]:
		pred = discretize_preds (pred)
		real = discretize_preds (real)

	recall_ 	= recall_score (real, pred, average = 'weighted')
	precision_ 	= precision_score (real, pred, average = 'weighted')
	fscore_ 	= f1_score (real, pred, average = 'weighted')


	'''recall_ 	= recall_score (real, pred)
	precision_ 	= precision_score (real, pred)
	fscore_ 	= f1_score (real, pred)'''

	accuray_ = cohen_kappa_score (real, pred)

	return [recall_, precision_, fscore_, accuray_]
