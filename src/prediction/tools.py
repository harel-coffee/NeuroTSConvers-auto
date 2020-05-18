import sys, os, inspect
import numpy as np
import pandas as pd
import random as rd
from ast import literal_eval
from sklearn.ensemble import IsolationForest
from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN # for imbalanced class
from sklearn.feature_selection import SelectFromModel
from sklearn import preprocessing

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
maindir = os.path.dirname(parentdir)

sys.path.insert(3,maindir)

from src.feature_selection.reduction import manual_selection, generic_reduction
#===========================================================
def outlier_detection (data_, alpha = 0.1):
	data = data_.copy ()

	outlier_model = IsolationForest (n_estimators = 100, contamination = alpha)
	outlier_model. fit (data)
	scores = outlier_model. predict (data)

	delt = []
	for m in range (len (scores)):
	    if scores [m] == -1:
	        delt. append (m)

	data = np. delete (data, delt, axis = 0)
	return data, delt

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

#=======================================================
def list_convers (n_blocks = 4, n_convs = 6):
	# Return the list of conversations names in the format like : CONV2_002
	convs = []
	for t in range (1, n_blocks + 1):
		for j in range (1, n_convs + 1):
			if j%2:
				i = 1
			else:
				i = 2
			convs. append ("convers-TestBlocks%s_CONV%d"%(t, i) +  "_%03d"%j)
	return convs

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
	test_index =  range (pos * nb_predictions, (pos + 1) * nb_predictions)

	train_index = []

	for i in range (len (df)):
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
def get_predictive_features_set (target_column, method, model, all, all_m, type):

	"""
		all_m: test non-realted features specific to each brain area
		all: test all features as input
	"""

	# if model is the baseline, choose some random variables
	if model == "baseline":
		set_of_behavioral_predictors =  [{"speech_left": ["IPU_left", "disc_IPU_left"]}]

	elif all_m:
		set_of_behavioral_predictors = manual_selection (target_column)

	elif all:
		# Add  of all features
		set_of_behavioral_predictors =  [
		{"speech_left": ["SpeechActivity_left", "disc_SpeechActivity_left", "IPU_left", "disc_IPU_left", "Polarity_left", "Subjectivity_left", "Overlap_left",\
		 				"ReactionTime_left", "FilledBreaks_left", "Feedbacks_left", "Discourses_left", "Particles_left", "Laughters_left", "LexicalRichness1_left",\
						"LexicalRichness2_left",  "SpeechRate_left", "UnionSocioItems_left"],\

		"speech_ts": ["SpeechActivity", "disc_SpeechActivity", "IPU", "disc_IPU", "Overlap", "ReactionTime", "FilledBreaks", "Feedbacks",\
					 "Discourses", "Particles", "Laughters", "LexicalRichness1", "LexicalRichness2", "SpeechRate", "UnionSocioItems", "Polarity", "Subjectivity"],

		 "facial_features": ["dlib_smiles", "Smile", "mouth_AU","eyes_AU", "total_AU", "head_rotation_energy", "head_translation_energy","gaze_angle_x","gaze_angle_y",\
		  					"pose_Tx", "pose_Ty", "pose_Tz","pose_Rx", "pose_Ry", "pose_Rz",\
							"angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"],

		 "eyetracking_ts": ["Vx", "Vy", "saccades", "Face", "Mouth", "Eyes"]
		}]

	else:
		#set_of_behavioral_predictors = manual_selection (target_column)
		if target_column in ["CSF", "GreyMatter", "WhiteMatter"]:
			target_name = "rMPFC"
		else:
			target_name = target_column
		results = pd. read_csv ("results/prediction/RF_%s_selected.tsv"%(type), sep = "\t")
		set_of_behavioral_predictors = literal_eval (results. loc [results. region == target_name, "selected_predictors"]. values[0])

		set_of_behavioral_predictors  = [str (x) for x in set_of_behavioral_predictors]

		# eliminate temporal representation
		set_of_behavioral_predictors = list (set ([('_'). join (a. split ('_')[:-1]) for a in set_of_behavioral_predictors]))

		#print (type (set_of_behavioral_predictors))
		set_of_behavioral_predictors = [{"selected": set_of_behavioral_predictors}]
		#print (target_column, set_of_behavioral_predictors)
		#exit (1)



	return set_of_behavioral_predictors
