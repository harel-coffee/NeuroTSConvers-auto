# -*- coding: utf-8 -*-
# Author: Youssef Hmamouche
import sys
import os
import glob

from prediction. tools import *
from sklearn import preprocessing
from ast import literal_eval

from joblib import Parallel, delayed

import warnings
warnings.filterwarnings("ignore")

from sklearn.externals import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier

import argparse

#============================================================
def  train_model (data, model, params, lag):

	if model == "LSTM":
		pred_model = fit_lstm (data [:, 1:], data[:, 0], lag = lag,  params = params)

	else:
		if model == "SGD":
			pred_model = SGDClassifier (**params)

		elif model == "GB":
			pred_model = GradientBoostingClassifier (**params)

		elif model == "RF":
			pred_model = RandomForestClassifier (**params)

		elif model == "SVM":
			pred_model = SVC (**params)

		elif model in ["RIDGE", "Ridge"]:
			pred_model = linear_model.Ridge (**params)

		elif model in ["LASSO", "Lasso"]:
			pred_model = linear_model.Lasso (**params)

		elif model in ["random", "baseline"]:
			pred_model = DummyClassifier (**params)

		pred_model. fit (data[:,1:], data[:,0])

	return pred_model

#============================================================
def train_model_area (subjects, target_column, convers, lag):

	"""
	Predict the BOLD signal of a given breain area using
	data of multiple subjects
	"""

	print (target_column, "\n ------------")

	if (int (convers[0]. split ('_')[-1]) % 2 == 1): convers_type = "HH"
	else : convers_type = "HR"

	""" extract best model, and the best parameters founded based on fscore measure """
	prediction_files = glob. glob ("results/prediction/*%s.tsv"%convers_type)
	best_score = 0

	for filename in prediction_files:

		model_name = filename. split ('/')[-1]. split ('_') [0]
		pred_data = pd.read_csv (filename,  sep = '\t', header = 0, na_filter = False, index_col = False)

		pred_data =  pred_data. loc [pred_data ["region"] == target_column]

		if (pred_data. shape [0] == 0):
			continue

		pred_results = pred_data. loc [pred_data ["fscore. mean"]. idxmax()]
		score =  pred_results ["fscore. mean"]

		if score > best_score:
			best_score = score
			best_model = model_name
			best_results = pred_results

	#selected_features =  best_results ["selected_indices"]
	selected_features  = literal_eval (best_results ["selected_predictors"])
	best_behavioral_predictors = literal_eval (best_results ["predictors_dict"])
	best_model_params = literal_eval (best_results ["models_params"]. replace("'", '"'))

	lagged_names = get_lagged_colnames (best_behavioral_predictors, args. lag)

	selected_indices = [lagged_names. index (col) for col in selected_features]

	# concatenate data of all subjects  with regards to the behavioral variables
	train_data = concat_ (subjects[0], target_column, convers, lag, best_behavioral_predictors, False)

	# use only best features founded using feature selection
	for subject in subjects[1:]:
		subject_data = concat_ (subject, target_column, convers, lag, best_behavioral_predictors, False)
		#train_data = np.concatenate ((train_data, subject_data), axis = 0)
		train_data = np. concatenate ((train_data, subject_data), axis = 0)


	# feature selection using the founded indices
	train_data = train_data [:, [0] + [int (a + 1) for a in selected_indices]]

	# Train the model
	print (best_model)
	pred_model = train_model (train_data, best_model, best_model_params, lag)

	# save the model
	joblib. dump (pred_model, "trained_models/%s_%s_%s.pkl"%(best_model, target_column, convers_type))

#====================================================================#

def train_all (subjects, regions, lag):

	subjects_str = "subject"
	for subj in subjects:
		subjects_str += "_%s"%subj

	subjects = ["sub-%02d"%i for i in subjects]

	# fill HH and HR conversations
	convers = list_convers ()
	hh_convers = []
	hr_convers = []

	for i in range (len (convers)):
		if i % 2 == 1:
			hr_convers. append (convers [i])
		else:
			hh_convers. append (convers [i])

	# Predict HH  and HR conversations separetely
	Parallel (n_jobs = 1) (delayed (train_model_area) (subjects, target_column,  convers = hh_convers, lag = int (lag))
									for target_column in regions)

	Parallel (n_jobs = 1) (delayed (train_model_area) (subjects, target_column, convers = hr_convers, lag = int (lag))
									for target_column in regions)

if __name__=='__main__':
	# read arguments
	parser = argparse. ArgumentParser ()
	parser. add_argument ('--subjects', '-s', nargs = '+', type = int)
	parser. add_argument ("--remove", "-rm", help = "remove previous files", action = "store_true")
	parser. add_argument ("--lag", "-p", default = 5, type = int)
	parser. add_argument ('--regions','-rg', nargs = '+', type=int)

	if not os. path. exists ("trained_models"):
		os. makedirs ("trained_models")

	args = parser.parse_args()
	print (args)

	if args. remove:
		os. system ("rm trained_models/*")

	# get regions names for their codes
	brain_areas_desc = pd. read_csv ("brain_areas.tsv", sep = '\t', header = 0)
	brain_areas = []

	for num_region in args. regions:
		brain_areas. append (brain_areas_desc . loc [brain_areas_desc ["Code"] == num_region, "Name"]. values [0])

	train_all (args. subjects, brain_areas, args. lag)
