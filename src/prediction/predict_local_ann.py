import pandas as pd
import argparse, os, inspect, sys
import numpy as np
from sklearn.metrics import precision_score, f1_score, balanced_accuracy_score, recall_score
from ast import literal_eval
from network_export import *
from py_net_module import pyNet
from joblib import Parallel, delayed

# global variables

brain_areas_desc = pd. read_csv ("brain_areas.tsv", sep = '\t', header = 0)

#===========================================================
def write_line (filename, row, mode = "a+", sep = ';'):
	f = open(filename, mode)
	for i in range (len (row)):
		row[i] = str (row[i])
	f.write (sep. join (row))
	f. write ('\n')
	f. close ()
	return

#===========================================================
def count_feat_in_all (feature, all_fatures):
	count = 0
	for var in all_fatures:
		if feature in var:
			count += 1
	return count
#===========================================================
def summarize_features_importance (features, importance, lag = True):

	if lag:
		reduce_features = [('_'). join (a. split ('_')[:-1]) for a in features]
	else:
		reduce_features = features

	df = pd.DataFrame ()

	if len (importance) == 0:
		return list (set (reduce_features)), importance

	df ["feats"] = reduce_features
	df ["scores"] = importance

	df = df. groupby ('feats')["scores"].sum (). reset_index ()

	df = df. sort_values (by = "scores", ascending = False). values

	feats = df[:,0]
	scores = df[:,1]
	return df[:,0]. tolist (), df[:,1]. tolist ()

#===========================================================
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

#===========================================================
def split_train_test (X, y, perc, pos, normalize = False, resample = False):
	"""
		X: np array (predictive variables)
		y: numpy array (the target variable)
		Split data into train and test set
		perc: the persentage of test set (between 0 and 1)
		pos: the position of the test data. For example, if 1, then take the first perc from the data
		normalize: if Truen normlize train and test set, with model fitted only on train data.
	"""

	nb_predictions = int (perc * X.shape[0])
	test_index =  range (pos * nb_predictions, (pos + 1) * nb_predictions)

	train_index = []

	for i in range (len (X)):
		if i not in test_index:
			train_index. append (i)

	train_set = X [train_index, :]
	test_set =  X [test_index,  :]

	# Target variable of train and test sets resp.
	y_train_ = y [train_index, :]. flatten ()
	y_test_ =  y [test_index, :].  flatten ()

	return train_set, test_set, y_train_, y_test_

#===========================================================
def predict_one_roi (target_name, features, dm_method, results_file, model, bold, behaviors, lag):

	bold_target_name = brain_areas_desc . loc [brain_areas_desc ["ShortName"] == target_name, "Name"]. values [0]
	available_features = list (behaviors. columns)

	dm_method = dm_method + "_%d"%len (literal_eval (features))
	features = literal_eval (features)

	set_of_behavioral_predictors = list (set ([('_'). join (a. split ('_')[:-1]) for a in features]))
	compact_features =  set_of_behavioral_predictors [:]

	if model == "LSTM":
		# eliminate variables where we don't have sequences of size 4
		for feature in set_of_behavioral_predictors:
			if count_feat_in_all (feature, available_features) < (lag - 2):
				set_of_behavioral_predictors. remove (feature)
				compact_features. remove (feature)

	set_of_behavioral_predictors = get_lagged_colnames (set_of_behavioral_predictors, lag, False)
	for a in set_of_behavioral_predictors:
		if a not in available_features:
			set_of_behavioral_predictors. remove (a)

	# load data and prepare training/test sets
	Y = pd.read_pickle ("concat_time_series/discr_bold_%s_data.pkl"%args.type). loc [:, [bold_target_name]]. values
	X = behaviors. loc [:,set_of_behavioral_predictors]. values
	X_train, X_test, y_train, y_test = split_train_test (X, Y, 0.25, 3)
	Y_train = y_train. reshape ((-1,1))

	# reshape the data
	if model == "MLP":
		opt_algos = ["sgd"]
		nb_layers = [1, 2]
		X_train = X_train. reshape ([X_train.shape[0],1, X_train.shape[1]])
		X_test = X_test. reshape ([X_test. shape[0], 1, X_test. shape[1]])
	elif model == "LSTM":
		opt_algos = ["sgd"]
		nb_layers = [1]
		n_temp_features =  int (X_train.shape[1] / 4)
		X_train = np. reshape (X_train, [X_train.shape[0], 4, n_temp_features], order = 'F')
		X_test = np. reshape (X_test, [X_test.shape[0], 4, n_temp_features], order = 'F')

	for num_layers in nb_layers:
		for algo in opt_algos:
			# learning rate
			if algo == "sgd":
				learning_rate = 0.01
			else:
				learning_rate = 0.1

			if model == "MLP":
				Net = pyNet (X_train.shape[2], "binary_cross_entropy")
				Net. push_back_dense (Dense (int ((X_train.shape[2] + 1) * 2 / 3), "relu", learning_rate, 1, algo))
				if nb_layers == 2:
					Net. push_back_dense (Dense (2, "relu", learning_rate, 1, algo))
				Net. push_back_dense ( Dense (1, "sigmoid", learning_rate, 1, algo))

			elif model == "LSTM":
				Net = pyNet ((lag - 2, n_temp_features), "binary_cross_entropy")
				l1 = LSTM (n_temp_features, learning_rate, 1, algo);
				l2 = Dense (1, "sigmoid", learning_rate, 1, algo);
				Net. push_back_lstm (l1)
				Net. push_back_dense (l2)

			Net. fit (X_train, Y_train, 100, True)
			pred = Net. predict (X_test)[:,0]

			features_importance = Net. input_features_scores ()

			if model == "MLP":
				compact_features, features_importance = summarize_features_importance (set_of_behavioral_predictors, features_importance)
			else:
				compact_features, features_importance = summarize_features_importance (compact_features, features_importance, lag = False)

			for i in range (len (pred)):
				if pred[i] < 0.5:
					pred[i] = 0
				else:
					pred[i] = 1

			fscore 	= np. round (f1_score (y_test, pred, average = 'weighted'), 2)
			accuracy = balanced_accuracy_score (y_test, pred)
			recall 	= np. round (recall_score (y_test, pred, average = 'weighted'), 2)
			row = [bold_target_name, dm_method, 6, "%d_%s"%(num_layers, algo), compact_features, set_of_behavioral_predictors,\
							features_importance, accuracy, recall, fscore, 0, 0, 0, len (compact_features)]
			write_line (results_file, row, mode = "a")


#===========================================================
if __name__ == '__main__':

	parser = argparse. ArgumentParser ()
	parser. add_argument ('--regions','-rg', nargs = '+', type=int)
	parser. add_argument ('--type','-t', type=str,  default="hh")
	parser. add_argument ('--lag','-lag', type=int,  default=6)
	parser. add_argument ('--model','-m', type=str, default="MLP")
	parser. add_argument ("--remove", "-rm", help = "remove previous files", action="store_true")
	args = parser.parse_args()

	# get bold signal
	bold = pd.read_pickle ("concat_time_series/discr_bold_%s_data.pkl"%(args.type))

	cols = ["region", "dm_method", "lag", "models_params", "predictors_dict",  "selected_predictors",  "features_importance",\
	 		"accuracy. mean", "recall. mean", "fscore. mean", "accuracy. std", "recall. std", "fscore. std", "nb_predictors"]

	filename = "results/prediction/%s_%s.csv"%(args.model, args. type. upper ())
	if args.remove:
		os. system ("rm results/prediction/%s_%s.csv"%(args.model, args. type. upper ()))

	if not os.path.exists (filename):
		results_file = open (filename, "a")
		results_file.write (';'. join (cols))
		results_file. write ('\n')
		results_file. close ()

	# get behavioral features
	behaviors = pd.read_pickle ("concat_time_series/behavioral_%s_data.pkl"%args.type)

	# Read brain areas
	brain_areas = []
	for num_region in args. regions:
		brain_areas. append (brain_areas_desc . loc [brain_areas_desc ["Code"] == num_region, "ShortName"]. values [0])

	# Get best results obtained with other classfiers
	best_model_results = pd. read_csv ("results/best_results/bestModel_%s.tsv"%(args.type). upper (), sep = "\t")
	best_model_results = best_model_results. loc [(best_model_results. model.isin (["MLP", "mlp", "lstm", "LSTM", "LSTM_MODEL", "MLP_MODEL"]) == False),["region", "selected_predictors", "dm_method"]]
	best_model_results =  best_model_results. loc [best_model_results. region.isin (brain_areas)]

	Parallel (n_jobs = 2) (delayed (predict_one_roi)(target_name, features, dm_method, filename, args. model, bold, behaviors, args. lag)
							for target_name, features, dm_method in best_model_results. values)

	print (".. Done.")
