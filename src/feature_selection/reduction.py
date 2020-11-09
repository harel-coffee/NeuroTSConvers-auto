import os, inspect, importlib
import numpy as np
import pandas as pd


from sklearn.decomposition import PCA, IncrementalPCA, KernelPCA
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVR, LinearSVR
from sklearn_extra.cluster import KMedoids
from sklearn.cluster import KMeans

from collections import defaultdict
from itertools import chain

from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mutual_info_score

#============================================================

def merge_two_dict (a, b):

	c = defaultdict (list)
	for k, v in chain (a.items (), b.items ()):
		c[k].extend(v)

	return c

#============================================================

def merge_dict (list_dict):
	if len (list_dict) == 1:
		return list_dict [0]

	c = list_dict [0]
	for i in range (1, len (list_dict)):
		c = merge_two_dict (c, list_dict[i])

	return c

#===========================================
def feature_selection_modalities (X, y, modalities, n_comps, method = "PCA", estimator_name = "RF"):
	"""
	    feature selection by modalities
		modalities: begin_end of modalities in index of X
	"""
	new_data = []
	begin_end = []
	begin = 0
	selectors = []

	for (mode, n_comp) in zip (modalities, n_comps):
		X_modality, indices, selector = generic_reduction (X[:, mode[0] : mode[1]], y, method, n_comp, estimator_name)
		selectors. append (selector)

		if len (new_data) == 0:
		    new_data = X_modality
		else:
		    new_data = np. concatenate ((new_data, X_modality), axis = 1)
		end = begin + X_modality. shape [1]

		begin_end. append ([begin, end])
		begin = end

	return new_data, begin_end, selectors

#===============================================================#
def transform_test_data_modalities (test_data, modalities, selectors, method = "PCA"):
	new_data = []

	for [begin, end], selector in zip (modalities, selectors):
		if selector is None:
			X_modality = test_data [:, begin: end]
		elif method in ["PCA", "KPCA", "IPCA", "Model_RANK"]:
			X_modality = selector. transform (test_data [:, begin: end])
		elif method in ["RFE"]:
			X_modality = selector. predict (test_data [:, begin: end])

		if len (new_data) == 0:
		    new_data = X_modality

		else:
			new_data = np. concatenate ((new_data, X_modality), axis = 1)

	return new_data

#===============================================================#
def generic_reduction (X, y, method,  n_comps = 0, estimator_name = "RF"):

	"""
	Reduce the dimmensionality of the train and test data with model fitted on train data.
	method : dimension reduction or feature selection methode used.
	n_comps :  reduction size.
	"""

	# compute number of features to select from percentage
	if n_comps == 0 or n_comps >= X. shape[1]:
		return X, list (range (X. shape [1])), None

	if method == "None":
	    return X, range (0, X. shape[1]), None

	elif method == "RFE":
	    return ref_local (X, y, n_comps, estimator_name)

	elif method == "Model_RANK":
		#clf = LogisticRegression()
		params = params = {'bootstrap': True, 'max_depth': 100, 'max_features': 'auto', 'n_estimators': 100, 'random_state': 5}
		clf = RandomForestClassifier (**params)
		clf = clf.fit(X, y)
		model = SelectFromModel (clf, max_features = n_comps, prefit = True, threshold=-np.inf)
		support = model. get_support()
		best_index = []
		for i in range (len (support)):
			if support [i]:
				best_index. append (i)
		return model.transform (X), best_index , model

	elif method == "PCA":
		model = PCA (n_components = n_comps, random_state = 5)

	elif method == "KPCA":
		model = KernelPCA (n_components = n_comps)

	elif method == "IPCA":
		model = IncrementalPCA (batch_size = None, n_components = n_comps)

	model = model.fit (X)

	return model. transform (X), list (range (X. shape [1])), model


#===============================================================#
def ref_local (X, y, n_comp, estimator_name = "RF"):

	# Prediction model to use
	if estimator_name == "RF":
		estimator = RandomForestClassifier (n_estimators = 150, max_features = 'auto', bootstrap = True, max_depth = 10)
	elif estimator_name in ["RIDGE", "Ridge", "LASSO", "Lasso"]:
		estimator = LinearSVR ()

	selector = RFE (estimator, n_comp, step = 1)
	selector = selector.fit (X, y)

	support = selector. support_
	best_indices = []

	if len (support) == 0:
		best_indices = [i for i in range (train. shape [1] - 1)]
	else:
		for i in range (len (support)):
		    if support[i]:
		        best_indices. append (i)

	return (X[:, best_indices], best_indices, selector)

#=====================================================================
def rfe_reduction (train_, test_, percs, estimator_name = "RF"):

	"""
	- Reduce train and test data with a model fitted on train data
		based on the recursive feature elimination method.
	- percs: percentage of features to select
	"""

	score = 0
	results = []
	support = []

	# make a copy to do not change the input data
	train = train_.copy ()
	test = test_. copy ()

	# find the best subset of features
	for perc_comps in percs:

		# Compute the number of features to select
		k = int ((train_. shape[1] - 1) * perc_comps)

		if k <= 0:
		    continue

		# Prediction model to use
		if estimator_name == "RF":
			estimator = RandomForestClassifier (n_estimators = 150, max_features = 'auto', bootstrap = True, max_depth = 10)
		elif estimator_name in ["RIFGE", "Ridge", "LASSO", "Lasso"]:
			estimator = linear_model.Ridge ({'alpha': 0.01})
		selector = RFE (estimator, k, step = 1)
		selector = selector.fit (train [:, 1: ], train [:, 0])

		if score < selector. score (train [:, 1: ], train [:, 0]):
		    score = selector. score (train [:, 1: ], train [:, 0])
		    support = selector. support_

	best_indices = []

	if len (support) == 0:
		best_indices = [i for i in range (train. shape [1] - 1)]
		score = 1

	else:
		for i in range (len (support)):
		    if support[i]:
		        best_indices. append (i)

	return (best_indices, score)


#======================================================================================
def manual_selection (region):

	speech_ipu = {"speech_ts": ["IPU_I"]}
	speech_P_ipu = {"speech_P_ts": ["IPU_P"]}
	speech_P_disc_ipu = {"speech_P_ts": ["disc_IPU_P"]}
	speech_disc_ipu = {"speech_ts": ["disc_IPU_I"]}

	speech_items = {"speech_ts": ["IPU_I", "disc_IPU_I", "SpeechRate_I",\
	 				"Overlap_I", "ReactionTime_I", "FilledBreaks_I", "Feedbacks_I", "Discourses_I",\
					"Particles_I", "Laughters_I", "LexicalRichness_I", "TypeTokenRatio_I",\
					 "UnionSocioItems_I"]}

	speech_P_items = {"speech_P_ts": ["IPU_P", "disc_IPU_P", "SpeechRate_P",\
	 					  "Overlap_P", "ReactionTime_P", "FilledBreaks_P", "Feedbacks_P", "Discourses_P", "Particles_P",\
						   "Laughters_P", "LexicalRichness_P", "TypeTokenRatio_O", "UnionSocioItems_P"]}

	if region in ["PMotor", "RightMotor"]:
		set_of_behavioral_predictors = [speech_P_ipu, speech_P_disc_ipu, speech_P_items, merge_dict ([speech_P_ipu, speech_ipu])]

	elif region in ["LeftSTS", "RightSTS"]:
		set_of_behavioral_predictors = [speech_items, speech_disc_ipu, speech_ipu, merge_dict ([speech_P_ipu, speech_ipu])]

	else:
		print ("ERROR, ROI: %s has not been processed in reduction step!"%region)
		exit (1)

	return set_of_behavioral_predictors

#====================================================================================
def mi_ranking (X, y, k):

	mi_table = []
	for j in range (X.shape[1]):
		if len (np.unique (X[:,j])) > 7:
			clustering = KMeans (n_clusters=7, algorithm = "elkan").fit (X [:,j:j+1])
			discr = clustering.labels_

		elif len (np.unique (X[:,j])) > 2:
			clustering = KMeans (n_clusters=2, algorithm = "elkan").fit (X [:,j:j+1])
			discr = clustering.labels_

		else:
			discr = X[:,j]

		mi_table. append (mutual_info_score (discr, y))

	sort_index = np.argsort(mi_table)

	return sort_index[-k:]

#====================================================================================
def mi_ranking_for_lstm (X_lagged, y, k, lag):

	# Get original variables from lagged ones
	index = []
	for i in range (0, X_lagged. shape [1], lag):
		index.append (i)
	X = X_lagged[:, index]

	# Feature selection on original variables
	mi_table = []
	for j in range (X.shape[1]):
		if len (np.unique (X[:,j])) > 7:
			clustering = KMeans (n_clusters=7, algorithm = "elkan").fit (X [:,j:j+1])
			discr = clustering.labels_

		elif len (np.unique (X[:,j])) > 2:
			clustering = KMeans (n_clusters=2, algorithm = "elkan").fit (X [:,j:j+1])
			discr = clustering.labels_

		else:
			discr = X[:,j]

		mi_table. append (mutual_info_score (discr, y))

	sort_index = np. argsort (mi_table)[-k:]

	# Extend selected indices on lagged variables
	sort_index_orig = []
	for i in range (len (sort_index)):
		sort_index_orig. append (sort_index[i]*lag)
		for j in range (1, lag):
			sort_index_orig. append (sort_index[i]*lag + j)

	return sort_index_orig
#====================================================================================
def gfsm_feature_selection (X, y, k):

	if k >= X.shape [1]:
		return list (range (X.shape [1]))

	kmedoids = KMedoids (n_clusters = k, max_iter=1000). fit (np. transpose (X))
	labels = kmedoids. labels_

	# Get the best variable_indice from each classe
	#best_index = [0 for i in np. unique (labels)]
	best_index = []

	for i in  range (len (np. unique (labels))):
		best_cor = 0
		for j in range (len (labels)):
			if labels[j] == i:
				correlation = np. absolute (np.corrcoef (X[:,j], y) [0,1])
				if correlation  > best_cor or correlation >= 0.9:
					#best_index [i] = j
					best_index. append (j)
					best_cor = correlation

	return best_index

#=========================================================
if __name__ == '__main__':
	print ("test")
