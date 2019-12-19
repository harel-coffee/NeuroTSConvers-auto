import os, inspect, importlib
import numpy as np
import pandas as pd


from sklearn.decomposition import PCA, IncrementalPCA, KernelPCA
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVR, LinearSVR

from collections import defaultdict
from itertools import chain

# import fcbf
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
resampling_spec = importlib.util.spec_from_file_location("fcbf", "%s/fcbf.py"%currentdir)
fcbf = importlib.util.module_from_spec(resampling_spec)
resampling_spec.loader.exec_module(fcbf)
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
	"""
	new_data = []
	begin_end = []
	begin = 0
	selectors = []

	for mode, n_comp in zip (modalities, n_comps):
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
		elif method in ["PCA", "KPCA", "IPCA"]:
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

	#return (train [:, [0] + best_indices], test [:, [0] + best_indices], best_indices, score)
	return (best_indices, score)


#======================================================================================
def manual_selection (region):

    # TODO: store information in a file instead of computing each time dictionaries
	AUrs = {"facial_features_ts": [" AU01_r"," AU02_r"," AU04_r"," AU05_r"," AU06_r"," AU07_r"," AU10_r"," AU12_r"," AU14_r"," AU15_r"," AU17_r"," AU20_r"," AU23_r"," AU25_r"," AU26_r"]}
	AUcs = {"facial_features_ts": [" AU01_c"," AU02_c"," AU04_c"," AU05_c"," AU06_c"," AU07_c"," AU10_c"," AU12_c"," AU14_c"," AU15_c"," AU17_c"," AU20_c"," AU23_c"," AU25_c"," AU26_c"]}
	eyetracking =  {"eyetracking_ts": ["Vx", "Vy", "saccades", "Face", "Mouth", "Eyes"]}
	face =  {"eyetracking_ts": ["Face"]}
	energy = {"energy_ts": ["mouth_cont", "eyes_cont"]}

	head_pose_r = {"facial_features_ts": [" pose_Rx", " pose_Ry", " pose_Rz"]}
	head_pose_t = {"facial_features_ts": [" pose_Tx", " pose_Ty", " pose_Tz"]}
	gaze_angle = {"facial_features_ts": [" gaze_angle_x", " gaze_angle_y"]}
	colors = {"colors_ts": ["colorfulness"]}

	audio = {"speech_ts": ["Signal"]}
	audio_left = {"speech_left_ts": ["Signal_left"]}
	speech_activity = {"speech_ts": ["SpeechActivity"]}
	speech_activity_left = {"speech_left_ts": ["SpeechActivity_left"]}
	speech_ipu = {"speech_ts": ["IPU"]}
	speech_left_ipu = {"speech_left_ts": ["IPU_left"]}
	speech_left_disc_ipu = {"speech_left_ts": ["disc_IPU_left"]}
	speech_disc_ipu = {"speech_ts": ["disc_IPU"]}
	speech_talk = {"speech_ts": ["talk"]}
	speech_left_talk = {"speech_left_ts": ["talk_left"]}

	all_0 = AUrs
	all_1 = merge_dict ([AUrs, eyetracking, head_pose_r])
	all_2 = merge_dict ([eyetracking])
	all_3 = merge_dict ([face, energy])
	all_4 = merge_dict ([eyetracking, head_pose_r, head_pose_t])
	all_5 = merge_dict ([face, head_pose_r, head_pose_t])
	all_6 = merge_dict ([head_pose_r, head_pose_t])
	all_7 = merge_dict ([AUrs, head_pose_r, head_pose_t])
	all_8 = merge_dict ([eyetracking, AUrs])
	all_9 = merge_dict ([face, head_pose_r, head_pose_t])
	all_10 = merge_dict ([energy, eyetracking])
	all_11 = merge_dict ([energy, head_pose_r, head_pose_t])


	items = {"speech_ts": ["FilledBreaks", "Feedbacks", "Discourses", "Particles", "Laughters"]}
	emotions = {"speech_ts":["Polarity", "Subjectivity"]}
	lexicalR = {"speech_ts":["LexicalRichness1", "LexicalRichness2"]}
	items_left = {"speech_left_ts": ["FilledBreaks_left", "Feedbacks_left", "Discourses_left", "Particles_left", "Laughters_left"]}
	speech_items = merge_dict ([speech_ipu, items])
	speech_items_left = merge_dict ([speech_left_ipu, items_left])

	if region in ["Fusiform Gyrus", "LeftFusiformGyrus", "RightFusiformGyrus"]:
		#set_of_behavioral_predictors = [all_1, all_8]
		set_of_behavioral_predictors = [face, energy, head_pose_r, all_0, all_1, all_2, all_3, all_4, all_5, all_6, all_7, all_8, all_9, all_10, all_11]

	elif region in ["LeftFrontaleyeField"]:
		#set_of_behavioral_predictors = [all_6]
		set_of_behavioral_predictors = [{"eyetracking_ts": ["x", "y"]}, eyetracking, {"eyetracking_ts": ["Vx", "Vy"]}, {"eyetracking_ts": ["saccades"]}]

	elif region in ["LeftMotor", "RightMotor"]:
		set_of_behavioral_predictors = [audio_left, speech_left_ipu, speech_left_talk, speech_left_disc_ipu, speech_activity_left]
		'''set_of_behavioral_predictors = [audio, audio_left, speech_left_ipu, speech_ipu, speech_talk, speech_left_talk,
										merge_dict ([speech_ipu, speech_left_ipu]),
										merge_dict ([audio_left, speech_talk]),
										merge_dict ([speech_left_ipu, audio_left])]'''

	elif region in ["Left Motor Cortex", "Right Motor Cortex"]:
		#set_of_behavioral_predictors = [audio, audio_left, speech_left_ipu, speech_ipu, speech_talk, speech_left_talk, merge_dict ([speech_ipu, speech_left_ipu]),
										#merge_dict ([speech_ipu, speech_talk]),  merge_dict ([speech_left_ipu, speech_left_talk]), merge_dict ([speech_left_ipu, AUcs]), merge_dict ([speech_left_ipu, AUrs])]
		#set_of_behavioral_predictors = [speech_left_ipu]

		set_of_behavioral_predictors = [audio, speech_activity_left, audio_left, speech_left_ipu, speech_ipu, speech_talk, speech_left_talk, speech_left_disc_ipu,
		merge_dict ([speech_left_disc_ipu, speech_disc_ipu]),
		merge_dict ([speech_ipu, speech_left_ipu]),
		merge_dict ([speech_activity_left, speech_left_talk]),
		merge_dict ([speech_left_ipu, speech_left_talk]),
		merge_dict ([speech_left_ipu, {"speech_left_ts":["Overlap_left"]}]),
		merge_dict ([speech_left_ipu, {"speech_left_ts":["Overlap_left", "ReactionTime_left"]}]),
		merge_dict ([speech_left_disc_ipu, {"speech_left_ts":["Overlap_left", "ReactionTime_left"]}])
		]

	elif region in ["Left Superior Temporal Sulcus", "Right Superior Temporal Sulcus", "LeftSTS", "RightSTS"]:

		'''set_of_behavioral_predictors = [speech_items, speech_disc_ipu, speech_ipu, speech_left_ipu,
										#merge_dict([speech_items, AUrs]),
										#merge_dict ([speech_ipu, AUrs]), audio,
										#merge_dict ([speech_ipu, audio]),
										#merge_dict ([speech_left_ipu, audio_left]),
										merge_dict ([speech_disc_ipu, items, emotions, lexicalR]),
										merge_dict ([speech_ipu, items, emotions])]'''

		set_of_behavioral_predictors = [speech_disc_ipu, speech_ipu, speech_activity,
										merge_dict ([speech_disc_ipu, items, emotions, lexicalR]),
										merge_dict ([speech_ipu, items, emotions, lexicalR])]

	elif region in ["region_6", "region_7"]:
		set_of_behavioral_predictors = [speech_items, emotions, eyetracking,
										merge_dict ([speech_items, emotions]),
										merge_dict ([speech_items, eyetracking]),
										merge_dict ([emotions, eyetracking]),
										merge_dict ([speech_items, emotions, eyetracking]),
										merge_dict ([speech_items, emotions, eyetracking])]


	elif region in ["region_8", "region_9"]:
	    set_of_behavioral_predictors = [speech_items, all_1, merge_dict ([speech_items, all_1, gaze_angle])]

	else:
		print ("ERROR, brain area has not been processed in reduction step!!")
		exit (1)

	return set_of_behavioral_predictors

if __name__ == '__main__':
	print ("test")
