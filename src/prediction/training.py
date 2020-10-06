import itertools, glob

import numpy as np
import pandas as pd
import random as rd

from sklearn.svm import SVC
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier, GradientBoostingClassifier
from sklearn import linear_model
from sklearn.model_selection import KFold, RandomizedSearchCV

from ast import literal_eval

from sklearn.metrics import recall_score, f1_score, balanced_accuracy_score

from src. prediction. lstm import *
from src. prediction. mlp import *

#===========================================
global_dict_models = {
		"BAG": BaggingClassifier (),
		"GB": GradientBoostingClassifier (),
		"RF": RandomForestClassifier (),
		"SVM": SVC (),
		"baseline": DummyClassifier (),
		"LREG": linear_model.LogisticRegression (),
		"ada": AdaBoostClassifier ()
		}

#===========================================
dict_params = {
	"BAG": {'bootstrap': [False], 'n_estimators': [10, 50, 100, 200, 300], 'random_state': [5]},
	"GB": {'learning_rate': [0.1, 0.2, 0.3, 0.4], 'n_estimators': [10, 50, 100], 'max_depth' : [5, 10, 50, 100]},
	"ada": {'n_estimators': [10, 50, 100, 500, 1000], 'learning_rate': [1.0, 0.9, 0.8]},
	"LREG": {'C': [1, 0.9, 0.8, 0.7, 0.6], 'solver': ['lbfgs', 'liblinear']},
	"SVM": {'C': [1, 0.9, 0.8, 0.7, 0.6], 'kernel': ['linear'], 'random_state': [5], 'class_weight': ["balanced"]},
	"LSTM": {'epochs': [20],  'neurons' : [30]},
	"MLP": {'epochs': [20],  'neurons' : [30]},
	"RF": {'bootstrap': [True], 'max_depth': [5, 10, 50, 100, 500, 1000], 'max_features': ['auto'],\
		  'n_estimators': [5, 10, 50, 100, 200, 300], 'random_state': [5], 'class_weight': ["balanced_subsample"]},
	"baseline":{'strategy': ['uniform', 'stratified', "most_frequent"], 'random_state': [5]}
}
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

#===========================================#
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

#===========================================#
def k_fold_cv (X, y, n_folds, classifier):

	"""
	    make a k-fold-cross-validation with a random search strategy
		classifier: name of the estimator to use
	"""

	kf_obj = KFold (n_splits = n_folds, shuffle = False)
	splits = kf_obj. split (X)

	params_search =  dict_params [classifier]
	estimator = global_dict_models[classifier]

	rf_random = RandomizedSearchCV (estimator, param_distributions = params_search, n_iter = 30, cv = splits, verbose=0, random_state=5,\
									scoring = ['balanced_accuracy', 'recall', 'f1_weighted'], refit = 'f1_weighted', n_jobs = 1)

	try:
		search = rf_random. fit (X, y. ravel ())
	except ValueError as e:
		print (e)
		exit (1)

	try:
		std_accuracy = search. cv_results_['std_test_balanced_accuracy'][search.best_index_]
		std_recall = search. cv_results_['std_test_recall'][search.best_index_]
		std_fscore = search. cv_results_['std_test_f1_weighted'][search.best_index_]

		mean_accuracy = search. cv_results_['mean_test_balanced_accuracy'][search.best_index_]
		mean_recall = search. cv_results_['mean_test_recall'][search.best_index_]
		mean_fscore = search. cv_results_['mean_test_f1_weighted'][search.best_index_]

		best_mean_scores = [mean_accuracy, mean_recall, mean_fscore]
		best_std_scores = [std_accuracy, std_recall, std_fscore]

		k_cv_scores = [search. cv_results_['split%d_test_f1_weighted'%i][search.best_index_] for i in range (n_folds)]
		best_model_params = search. best_params_
		best_model = search. best_estimator_
	except ValueError as e:
		print (e)
		exit (1)

	return best_model, best_model_params, best_mean_scores, best_std_scores, k_cv_scores

#======================================================
def  train_model (data, model, params, lag):
	try:
		if model == "LSTM":
			pred_model = LSTM_MODEL (lag - 2)
			pred_model. fit (data [:, 1:], data[:, 0])

		elif model == "CMLP":
			pred_model = CMLP (lag - 2)
			pred_model. fit (data [:, 1:], data[:, 0])

		elif model == "MLP":
			pred_model = MLP (lag - 2)
			pred_model. fit (data [:, 1:], data[:, 0])

		else:
			pred_model = global_dict_models [model]
			pred_model. set_params (**params )
			pred_model. fit (data[:,1:], data[:,0])
	except ValueError as e:
		print (" %s \n Error in train_model (training.py)."%e)
		exit (1)

	return pred_model

#==============================================================#
def  test_model (X, y, model):
	try:
		preds = model. predict (X)
		accuracy_ 	= balanced_accuracy_score (y, preds)
		recall_ 	= recall_score (y, preds, average = 'weighted')
		fscore_ 	= f1_score (y, preds, average = 'weighted')
	except ValueError as e:
		print (" %s \n Error in test_model (training.py)."%e)
		exit (1)

	return [accuracy_, recall_, fscore_]

#=================================================================================================================
def train_pred_model (model, dm_method, train_data, target_column, variables_list, lag, convers_type, find_params):
	try:
		if model in ['LSTM', 'CMLP', "MLP",]:
			best_model_params =  {'epochs': [],  'neurons' : []}
			pred_model = train_model (train_data, model, best_model_params, lag)
			std_errors = [0, 0, 0]
			features_importance = []

		# k-fold cross validation: find hyperparameters if the model
		elif find_params:
			pred_model, best_model_params, mean_scores, std_errors, k_fscores = k_fold_cv (train_data[:,1:], train_data[:,0], n_folds = 10, classifier = model)
			return best_model_params, mean_scores, std_errors, k_fscores, []

		else:
			# extract model params from the previous k-fold-validation results
			models_params_file = glob. glob ("results/models_params/*%s_%s.csv" %(model, convers_type. upper ()))[0]
			best_model_params, std_errors = extract_models_params_from_crossv (models_params_file, target_column, variables_list, dm_method)

			# Train the model
			pred_model = train_model (train_data, model, best_model_params, lag)
			if model == "baseline":
				features_importance = []
			else:
				if model == "RF":
					features_importance = pred_model.feature_importances_
				elif model in ["SVM", "LREG"]:
					try:
						features_importance = pred_model.coef_[0]
						features_importance = [abs (a) for a in features_importance]
						sum_coef = sum (features_importance)
						features_importance = [a / sum_coef for a in features_importance]
					except:
						features_importance = []
				else:
					features_importance = []
	except:
		print ("Error in train_pred_model")
		exit (1)

	return pred_model, best_model_params, std_errors, features_importance
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
		std_errors =  models_params. loc [best_model_params_index, ["accuracy. std",  "recall. std",  "fscore. std"]]
		return literal_eval (best_model_params), std_errorss

	# find the models_paras associated to each predictors_list with dimension reduction method
	for i in list (models_params. index):
		if set (literal_eval(models_params. loc [i, "predictors_list"])) == set (features) and models_params. loc [i, "dm_method"] == reduction_method:
			features_exist_in_models_params = True
			best_model_params_index = i
			break

	# find the models_params associated to each predictors_list without dimension reduction method
	if not features_exist_in_models_params:
		for i in list (models_params. index):
			if set (literal_eval(models_params. loc [i, "predictors_list"])) == set (features):
				features_exist_in_models_params = True
				best_model_params_index = i
				break

	# find the models_params associated to each predictors_list with equal number of features
	if not features_exist_in_models_params:
		for i in list (models_params. index):
			if len (set (literal_eval(models_params. loc [i, "predictors_list"]))) == len (set (features)):
				features_exist_in_models_params = True
				best_model_params_index = i
				break

	# else, choose the best model_params without considering features
	if not features_exist_in_models_params:
		best_model_params_index = models_params ["fscore. mean"].idxmax ()


	best_model_params = models_params. loc [best_model_params_index, "models_params"]
	std_errors =  models_params. loc [best_model_params_index, ["accuracy. std",  "recall. std",  "fscore. std"]]. values

	return literal_eval (best_model_params), std_errors. tolist ()
