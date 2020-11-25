import sys, os, inspect
import numpy as np
import pandas as pd
import random as rd
from ast import literal_eval
from imblearn.over_sampling import ADASYN # for imbalanced data
from sklearn.feature_selection import SelectFromModel
from sklearn import preprocessing

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
maindir = os.path.dirname(parentdir)

sys.path.insert(3,maindir)

from src.feature_selection.reduction import generic_reduction

#===========================================================
def features_in_select_results (selec_results, region, features):
	"""
		- check if a set of predictive variables has been processed in the feature selection step
			if so, the reduced form of this set is used
		- selec_results: the dataframe containing the feature selection results
			region: brain region
		- features: the set of predictive features
		- returns the indices (in the list features) of the selected variables
	"""
	features_exist = False
	results_region = selec_results. loc [(selec_results ["region"] ==  region)]
	selected_indices = [i for i in range (len (features))]

	# TODO: groupe by [region, features]

	''' find the models_paras associated to each predictors_list '''
	for i in list (results_region. index):
		if set (literal_eval(results_region. loc [i, "features"])) == set (features):
			features_exist = True
			selected_indices = literal_eval (results_region. ix [i, "selected_features"])
			break
	return selected_indices
#=======================================================

def write_line (filename, row, mode = "a+", sep = ';'):
	f = open(filename, mode)
	for i in range (len (row)):
		row[i] = str (row[i])
	f.write (sep. join (row))
	f. write ('\n')
	f. close ()
	return


#===============================================================
def get_names_from_laggedNames (lagged_n):
	vars = []
	for a in lagged_n:
		vars.append ('_'. join (a. split ('_')[:-1]))

	return list (set (vars))
#=================================================================================================================
def get_indices (small_list, big_list):
	"""
		get index of samll_list in big_list
		supposes that big_list contains small_list and both contain unique values
	"""
	indices = []
	for j in range (len (big_list)):
		if big_list[j] in small_list:
			indices. append (j)
	return indices

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

#============================================================
def train_test_from_df (df, y, perc, pos, normalize = False, resample = False):
	"""
		df: dataframe (predictive variables)
		y: numpy array (the target variable)
		Split data into train and test set
		perc: the persentage of test set (between 0 and 1)
		pos: the position of the test data. For example, if 1, then take the first perc from the data
		normalize: if Truen normlize train and test set, with model fitted only on train data.
	"""

	nb_predictions = int (perc * df.shape[0])

	begin_index = pos * nb_predictions
	end_index = min (df.shape[0], int ((pos + 1) * nb_predictions))

	#begin_index = df.shape[0] - nb_predictions
	#end_index = df.shape[0]

	test_index =  range (begin_index, end_index)
	train_index = []

	for i in range (df.shape[0]):
		if i not in test_index:
			train_index. append (i)

	train_set = df. iloc [train_index, :].values
	test_set =  df. iloc [test_index, :].values

	# Target variable of train and test sets resp.
	y_train_ = y [train_index, :]. copy (). flatten ()
	y_test_ =  y [test_index, :].  copy (). flatten ()

	if normalize:
		min_max_scaler = preprocessing. MinMaxScaler ()
		train_set = min_max_scaler. fit_transform (train_set)
		test_set = min_max_scaler. transform (test_set)

	# Handling  the oversampling in training data (if they are imbalanced) with the ADASYN algorithm
	if resample:
		res_model = ADASYN (random_state=5)
		try:
			train_set, y_train_ =  res_model. fit_resample (train_set, y_train_)
			#test_set, y_test_ =  RandomOverSampler (random_state=5). fit_resample (test_set, y_test_)
		except:
			print ("Data already balanced!")
			pass

	return train_set, test_set, y_train_, y_test_

#===============================================================
def specific_feature_selection (train_set, test_set, y_train,  k, method, model):

	# No feature selection in this case
	if method == "None" or model == "baseline" or k == (train_set. shape [1]):
		select_indices = [a for a in range (train_set. shape [1])]
		train_set_select = train_set [:]
		test_set_select = test_set [:]

	# Feature selection with the TREE method on all features (without specifing the number of features to select)
	elif method == "TREE_ALL":
		select_indices = []
		if model == "KNN":
			clf = global_dict_models ["RF"]
		else:
			clf = global_dict_models [model]
		clf = clf.fit (train_set, y_train)
		select_model = SelectFromModel (clf, prefit = True)
		support = select_model. get_support()

		for i in range (len (support)):
			if support [i]:
				select_indices. append (i)

		# transform data
		train_set_select = select_model.transform (train_set)
		test_set_select = select_model.transform (test_set)

	# Normal feature selection by specifing the number of features to select
	else:
		train_set_select, select_indices, selector = generic_reduction (train_set, y_train, method = method, n_comps = k, estimator_name = "RF")
		if method in ["PCA", "KPCA", "ICA", "TREE"]:
			test_set_select = selector. transform (test_set)
		else:
			test_set_select = test_set [:, [int(a) for a in select_indices]]


	return train_set_select, test_set_select, select_indices

#============================================================
def get_predictive_features_set (target_column, type):
	"""
		target_column: name of the ROI
		model: test all features as input
	"""

	brain_areas_desc = pd. read_csv ("brain_areas.tsv", sep = '\t', header = 0)
	short_target_name = brain_areas_desc . loc [brain_areas_desc ["Name"] == target_column, "ShortName"]. values [0]

	if target_column in ["GreyMatter", "WhiteMatter"]:
		target_name = "lSTS"
		short_target_name = target_name
	else:
		target_name = target_column

	if not os.path.exists ("results/last_best_results/bestModel_%s.tsv"%(type)):
		print ("path results/last_best_results/bestModel_%s.tsv does not exist!"%(type))
		exit (1)

	try:
		results = pd. read_csv ("results/last_best_results/bestModel_%s.tsv"%(type), sep = "\t")
		set_of_behavioral_predictors = literal_eval (results. loc [results. region == short_target_name, "selected_predictors"]. values[0])
		set_of_behavioral_predictors  = [str (x) for x in set_of_behavioral_predictors]
	except:
		print ("Error when getting  selected features from existing results!")
		exit (1)

	return set_of_behavioral_predictors
