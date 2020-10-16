# -*- coding: utf-8 -*-
# Author: Youssef Hmamouche
import sys
import os
import glob

from prediction. tools import *
from sklearn import preprocessing
from ast import literal_eval

import joblib
#from joblib import Parallel, delayed

import warnings
warnings.filterwarnings("ignore")

#from sklearn.externals import joblib

from sklearn import linear_model
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier
from prediction. lstm import *

#from imblearn.over_sampling import ADASYN

import argparse


def get_features_from_lagged (lagged_variables):
	features = set (['_'. join (a.split ('_')[0:-1]) for a in lagged_variables])
	return ','. join (map (str, list (features)))


#============================================================
def  train_model (X, y, model, params):

	if model == "LSTM":
		pred_model = fit_lstm (X, y, lag = 4,  params = params)

	else:
		if model == "SGD":
			pred_model = SGDClassifier (**params)

		elif model == "GB":
			pred_model = GradientBoostingClassifier (**params)

		elif model == "LREG":
			pred_model = linear_model.LogisticRegression ()

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

		pred_model. fit (X, y)

	return pred_model

#============================================================
def train_model_area (X, y, target_column, convers_type):

	""" X: dataframe with columns
		y: dataframe one column, the target variable
	"""

	print ("\n\nProcessing ROI: %s ...."%target_column)
	""" extract best model, and the best parameters founded based on fscore measure"""
	prediction_files = glob. glob ("results/prediction/*%s.tsv"%convers_type)
	best_score = 0

	for filename in prediction_files:
		if "LSTM" in filename or "MLP" in filename  or "baseline" in filename:
			continue

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
	best_behavioral_predictors = get_features_from_lagged (selected_features)
	best_model_params = literal_eval (best_results ["models_params"]. replace("'", '"'))

	# feature selection using the founded indices
	X_train = X. loc[:, selected_features]

	# balance the data with ADASYN
	'''try:
		X_train, y =  ADASYN (random_state=5). fit_resample (X_train.values, y. values. flatten ())
	except:
		X_train, y = X_train.values, y. values. flatten ()
		pass'''

	# Train the model
	print ("Best model: %s"%best_model)
	print ("... Done\n")

	pred_model = train_model (X_train, y, best_model, best_model_params)

	# save the model
	joblib. dump (pred_model, "trained_models/%s_%s_%s.pkl"%(best_model, target_column, convers_type), compress=3)

	return best_model, best_behavioral_predictors

#====================================================================#

def train_all (X, Y, regions, short_regions, type = "HH"):

	output = []
	for target_column, short_target_column in zip (regions, short_regions):
		best_model, features = train_model_area (X, Y. loc [:, target_column], short_target_column, type)
		output. append ([short_target_column, best_model, features])

	return pd.DataFrame (output, columns = ["ROI", "Prediction model", "Predictive Features"])

if __name__=='__main__':
	# read arguments
	parser = argparse. ArgumentParser ()
	parser. add_argument ("--remove", "-rm", help = "remove previous files", action = "store_true")
	parser. add_argument ('--regions','-rg', nargs = '+', type=int)

	args = parser.parse_args()
	print (args)

	# create output folder if not exist
	if not os. path. exists ("trained_models"):
		os. makedirs ("trained_models")
	else:
		# remove old results if specified in argument
		if args. remove:
			os. system ("rm trained_models/*")

	# get regions names for their codes
	brain_areas_desc = pd. read_csv ("brain_areas.tsv", sep = '\t', header = 0)
	brain_areas = []
	short_brain_areas = []

	for num_region in args. regions:
		brain_areas. append (brain_areas_desc . loc [brain_areas_desc ["Code"] == num_region, "Name"]. values [0])
		short_brain_areas. append (brain_areas_desc . loc [brain_areas_desc ["Code"] == num_region, "ShortName"]. values [0])

	# Read training data for human-human and human-robot
	X_hh = pd. read_pickle ("concat_time_series/behavioral_hh_data.pkl")
	X_hr = pd. read_pickle ("concat_time_series/behavioral_hr_data.pkl")

	Y_hh = pd. read_pickle ("concat_time_series/discr_bold_hh_data.pkl")
	Y_hr = pd. read_pickle ("concat_time_series/discr_bold_hr_data.pkl")

	# Train the models for each brain area
	results_hh = train_all (X_hh, Y_hh, brain_areas, short_brain_areas, "HH")
	results_hr = train_all (X_hr, Y_hr, brain_areas, short_brain_areas, "HR")

	#print (results_hh)
	#print (results_hr)

	'''df = pd. concat ([results_hh, results_hr. iloc[:,1:]], axis = 1)
	header1 = ["ROIS"] + ["Human-Human", "Human-Machine"] +  ["Human-Human", "Human-Machine"]
	header2 = ["ROIS"] + ["Best model", "Features"] +  ["Best model", "Features"]


	df. columns = [np. array (header1), np. array (header2)]
	print (df)'''
	#print(df.to_latex(index=False, multirow = True))

	'''latex = df.to_latex(index=False, multirow = True)
	file = open ("table.txt", mode = 'w')
	file. write (latex)
	file. close ()'''
