"""
	Author: Youssef Hmamouche
	Year: 2019
	Description: predicting multiple brain areas using one prediction model.
"""

import os
from joblib import Parallel, delayed

# local files
from src.prediction.tools import *
from src.prediction.training import *
from src.concat_data import get_lagged_colnames

from src.feature_selection.reduction import gfsm_feature_selection, mi_ranking
import warnings
warnings.filterwarnings("ignore")


def summarize_features_importance (features, importance):

	reduce_features = [('_'). join (a. split ('_')[:-1]) for a in features]
	df = pd.DataFrame ()
	df ["feats"] = reduce_features
	df ["scores"] = importance
	df = df. groupby ('feats')["scores"].sum (). reset_index (). round (2)


	df = df. drop (df[df.scores <= 0.02].index). sort_values (by = "scores", ascending = False). values

	feats = df[:,0]
	scores = df[:,1]
	return df[:,0]. tolist (), df[:,1]. tolist ()

#============================================================
def predict_area (behavioral_variables, target_column, model, lag, filename, find_params = False, type = "HH", all = False, all_m = False,  method = "None"):
	"""
		- target_column:
		- set_of_behavioral_predictors:
		- lag: the lag parameter
		- model: the prediction model name
		- filename: where to put the results
		- find_params: if TRUE, a k-fold-cross-validation  is used to find the parameters of the models, else using the previous one stored.
		- method: the feature selection method. None for no feature selection, and rfe for recursive feature elimination.
		- all: if True, we use all predictive variables as input.
	"""

	if target_column in ["CSF", "GreyMatter", "WhiteMatter"]:
		method = "None"
		all = False

	print (target_column)

	y = pd. read_pickle ("concat_time_series/discr_bold_%s_data.pkl"%str (type). lower ()). loc[:, [target_column]]. values

	if model == "baseline":# or all_m:
		method = "None"

	all_lagged_variables = [a for a in behavioral_variables. columns]
	set_of_behavioral_predictors = get_predictive_features_set (target_column, method, model, all, all_m, type)

	# Split data into train and test set, where the test set is the last 20% of the data
	if find_params or model == "LSTM":
		resample = False
	else:
		resample = True
	X_train_all, X_test_all, y_train, y_test = train_test_from_df (behavioral_variables, y, perc = 0.333, pos = 2, normalize = False, resample = resample)

	# A loop to test each set of predictive features
	for set_of_predictors in set_of_behavioral_predictors:
		lagged_names = get_lagged_colnames (set_of_predictors, lag)

		for x in lagged_names:
			if x not in all_lagged_variables:
				print ("Warning, concat data do not contain",x)
				exit (1)

		X_train = X_train_all [:, get_indices (lagged_names, all_lagged_variables)]
		X_test = X_test_all [:, get_indices (lagged_names, all_lagged_variables)]


		# test multiple number of features for feature selection_
		if model == "baseline" or method in ["None", "", "TREE_ALL"]: #or all_m:
			set_k = [X_train. shape [1]]

		else:
			#set_k = [2, 10, 20, 30, 40, 50, 60, 70]
			#set_k = [60, 70]
			set_k = list (range (2, 21, 2))


		for n_comp in set_k:
			print ("%s K = %s ----------"%(method, n_comp))

			if method in ["None", ""]:
				dm_method = "%s_%d"%(method, X_train. shape [1])
			else:
				dm_method = "%s_%s"%(method, str (n_comp))

			if n_comp > X_train. shape [1]:
				continue

			if method == "K_MEDOIDS":
				selected_indices = gfsm_feature_selection (X_train, y_train, n_comp)
				X_train_select = X_train [:,selected_indices]
				X_test_select = X_test [:,selected_indices]

			elif method == "MI_RANK":
				selected_indices = mi_ranking (X_train, y_train, n_comp)
				X_train_select = X_train [:,selected_indices]
				X_test_select = X_test [:,selected_indices]

			else:
				X_train_select, X_test_select, selected_indices = specific_feature_selection (X_train, X_test, y_train, n_comp, method, model)

			# train the model and extract the best model parameters from the  k_fold_cross_validation experiment
			train_data = np. concatenate ((y_train. reshape ((-1,1)), X_train_select), axis = 1)


			selected_list = [lagged_names [i] for i in selected_indices]

			if find_params:
				best_model_params, evaluations, std_errors, k_Fscores, _ = train_pred_model (model, method, train_data, target_column, lagged_names, lag, type, find_params)
				str_dict = get_names_from_laggedNames (selected_list)
				row = [target_column,  dm_method, lag, str (best_model_params), str_dict, lagged_names, selected_list] + evaluations + std_errors + [','. join (map (str, k_Fscores))]

			else:
				pred_model, best_model_params, std_errors, features_importance = train_pred_model (model, method, train_data, target_column, lagged_names, lag, type, find_params)
				evaluations = test_model (X_test_select, y_test, pred_model)
				reduced_features, importances = summarize_features_importance (selected_list, features_importance)
				row = [target_column,  dm_method, lag, str (best_model_params), reduced_features, selected_list, importances] + evaluations + std_errors

			write_line (filename, row, mode = "a")


#=========================================================================
def predict_all (_regions, lag, model, remove, _find_params, all, all_m, method = "None"):

	print ("-- MODEL :", model)

	if _find_params:
		colnames = ["region", "dm_method", "lag", "models_params", "predictors_dict",   "predictors_list", "selected_predictors",
				"recall. mean",  "precision. mean", "fscore. mean",  "recall. std",  "precision. std",  "fscore. std", "K-Fscores"]
	else:
		colnames = ["region", "dm_method", "lag", "models_params", "predictors_dict", "selected_predictors", "features_importance",
				"recall. mean",  "precision. mean", "fscore. mean",   "recall. std",  "precision. std",  "fscore. std"]

	if _find_params:
		filename_hh = "results/models_params/%s_HH.csv"%(model)
		filename_hr = "results/models_params/%s_HR.csv"%(model)

	else:
		filename_hh = "results/prediction/%s_HH.csv"%(model)
		filename_hr = "results/prediction/%s_HR.csv"%(model)

	for filename in [filename_hh, filename_hr]:
		# remove previous output files ir remove == true
		if remove:
			os. system ("rm %s"%filename)
		if not os.path.exists (filename):
			f = open (filename, "w+")
			f.write (';'. join (colnames))
			f. write ('\n')
			f. close ()

	behavioral_variables_hh = pd. read_pickle ("concat_time_series/behavioral_hh_data.pkl")
	behavioral_variables_hr = pd. read_pickle ("concat_time_series/behavioral_hr_data.pkl")


	if behavioral_variables_hh. isnull (). any (). any () or behavioral_variables_hr. isnull (). any (). any ():
		print (" ERROR: input data contain Nan values!")

	# Predict HH  and HR conversations separetely
	Parallel (n_jobs = 3) (delayed (predict_area)
	(behavioral_variables_hh, target_column,  model = model, lag = lag, filename = filename_hh, find_params = _find_params, type = "HH", all = all, all_m = all_m, method = method)
									for target_column in _regions)


	Parallel (n_jobs = 3) (delayed (predict_area)
	(behavioral_variables_hr, target_column,  model = model, lag = lag, filename = filename_hr, find_params = _find_params, type = "HR", all = all, all_m = all_m, method = method)
									for target_column in _regions)

if __name__ == '__main__':
	print ("test")
