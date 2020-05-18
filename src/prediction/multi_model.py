"""
	Author: Youssef Hmamouche
	Year: 2019
	Prediction multiple brain areas in parallel with a new multimodal classification model
"""

import sys, os, inspect, glob, itertools
from sklearn import preprocessing
from joblib import Parallel, delayed

# for oversampling
#from imblearn.over_sampling import RandomOverSampler,  SMOTE, ADASYN

# local files
from src.feature_selection. reduction import manual_selection, generic_reduction, feature_selection_modalities, transform_test_data_modalities
from src.prediction.tools import *
from src.prediction.training import *
from src.prediction.new_model import new_model
from sklearn.preprocessing import KBinsDiscretizer

#from src.prediction.prediction import get_predictive_features_set, train_test_from_df, get_indices


global_dict_models = {
	"BAG": BaggingClassifier (),
	"MLP":RandomForestClassifier (),
	"SGD": SGDClassifier (),
	"GB": GradientBoostingClassifier (),
	"RF": RandomForestClassifier (),
	"SVM": RandomForestClassifier (),
	"FUZZY": RandomForestClassifier (),
	"RIDGE": linear_model.Ridge (),
	"LASSO": linear_model.Lasso (),
	"baseline": DummyClassifier (),
	"LREG": linear_model.LogisticRegression (),
	"ada": AdaBoostClassifier (),
	"KNN": KNeighborsClassifier ()
	}


#=============================================================
def get_best_model_per_modality (ROI, variables_dict, model, dm_method, convers_type):

	if model == "LSTM":
		return "LSTM", [0, 0, 0]
	best_models = []
	std_errors_list = []
	models_params_file = glob. glob ("results/models_params/*%s_%s.csv" %(model, convers_type. upper ()))[0]

	#for [begin, end] in begin_end:
	for key in variables_dict. keys ():
		#sub_variables = variables_names[begin:end]
		sub_variables = {key: variables_dict [key]}

		params, std_errors = extract_models_params_from_crossv (models_params_file, ROI, sub_variables, dm_method)
		best_model = global_dict_models [model]
		best_model. set_params (**params)
		best_models. append (params)
		std_errors_list. append (std_errors)

	if len (std_errors_list) == 1:
		std_errors_mean = std_errors_list [0]

	else:
		std_errors_mean = np. mean (np. array (std_errors_list), axis = 0). tolist ()


	return best_models, std_errors_mean

#=============================================================
# Get begin and end of each modality from a dictioanry of variables
def get_begin_end (dict_variables, lags):
	begin_end = []
	begin = 0
	for item in dict_variables. keys ():
		end = begin + len (dict_variables [item]) * lags
		begin_end. append ([begin, end])
		begin = end

	return begin_end

#============================================================
def predict_area (behavioral_variables, target_column, set_of_behavioral_predictors, model, lag, filename, find_params = False, method = "TREE", type = "HH", all = False, all_m = False):
	"""
		- target_column:
		- set_of_behavioral_predictors:
		- lag: the lag parameter
		- model: the prediction model name
		- filename: where to put the results
		- find_params: if TRUE, a k-fold-cross-validation  is used to find the parameters of the models, else using the previous one stored.
		- method: the feature selection method. None for no feature selection, and rfe for recursive feature elimination.
	"""
	#print (target_column, "\n", 18*'-')

	# load target data
	if model in ["RIDGE", "LASSO"]:
		y = pd. read_csv ("concat_time_series/bold_%s_data.csv"%str (type). lower (), sep = ';'). loc[:, [target_column]]. values

	else:
		y = pd. read_csv ("concat_time_series/discr_bold_%s_data.csv"%str (type). lower (), sep = ';'). loc[:, [target_column]]. values

	results = []
	if target_column in ["CSF", "GreyMatter", "WhiteMatter"]:
		method = "TREE_ALL"
		all = False


	# select the set of behavioral predictors depending on arguments
	set_of_behavioral_predictors = get_predictive_features_set (target_column, set_of_behavioral_predictors, method, model, all, all_m)

	# Split data into train and test set, where the test set is the last 20% of the data
	if find_params or model == "LSTM":
		resample = False
	else:
		resample = True
	X_train_all, X_test_all, y_train, y_test = train_test_from_df (behavioral_variables, y, perc = 0.1667, pos = 2, normalize = True, resample = resample)


	all_lagged_variables = [a for a in behavioral_variables. columns]
	set_of_behavioral_predictors = get_predictive_features_set (target_column, method, model, all, all_m, type)


	# A loop to test each set of predictive features
	for behavioral_predictors in set_of_behavioral_predictors:

		lagged_names = get_lagged_colnames (behavioral_predictors, lag)
		X_train = X_train_all [:, get_indices (lagged_names, all_lagged_variables)]
		X_test = X_test_all [:, get_indices (lagged_names, all_lagged_variables)]

		# test multiple number of features for feature selection_
		if method in ["None", "", "TREE_ALL"] or all_m:
			set_k = [X_train. shape [1]]

		else:
			set_k = list (range (2, 41, 4))

		for n_comp in set_k:
			print ("%s K = %s ----------"%(method, n_comp))
			score = []

			if method in ["None", ""]:
				dm_method = method
			else:
				dm_method = "%s_%s"%(method, str (n_comp))

			if n_comp > X_train. shape [1]:
				break


			begin_end = get_begin_end (behavioral_predictors, lag - 2)

			best_models_modalities, std_errors = get_best_model_per_modality (target_column, behavioral_predictors, model, dm_method, type)


			if method in ["TREE", "RFE", "PCA", "KPCA"]:
				# Feature selection per modality

				X_train_select, new_begin_end, selectors = feature_selection_modalities (X_train, y_train, begin_end, [n_comp for u in range (len (begin_end))], method, "RF")
				X_test_select = transform_test_data_modalities (X_test, begin_end, selectors, method = method)

				#if method in ["TREE", "RFE"]:
					#X_test_select = transform_test_data_modalities (X_test, new_begin_end, [None for n in range (len (begin_end))], method)

				# Train the model
				pred_model = new_model (new_begin_end, best_models_modalities, strategy = "mean")
				pred_model. fit (X_train_select, y_train)
				best_model_params = ""

				selected_indices = []
				for b_e in new_begin_end:
					selected_indices += list (range (b_e [0], b_e [1]))

				evaluations = test_model (X_test_select, y_test, pred_model)

			else:
				pred_model = new_model (begin_end, best_models_modalities)
				pred_model. fit (X_train, y_train. flatten ())
				best_model_params = ""
				evaluations = test_model (X_test, y_test, pred_model)
				selected_indices = [a for a in range (len (lagged_names))]

			#==========================================================
			selected_list = [lagged_names [i] for i in selected_indices]
			str_dict = get_names_from_laggedNames (selected_list)
			row = [target_column,  dm_method, lag, str (best_model_params), str_dict, lagged_names, selected_list] + evaluations + std_errors
			write_line (filename, row, mode = "a")


#=========================================================================
def multimodal_predict (_regions, lag, model, remove, _find_params,  all, all_m):

	print ("-- MODEL :", model)
	colnames = ["region", "dm_method", "lag", "models_params", "predictors_dict",   "predictors_list", "selected_predictors",
			"recall. mean",  "precision. mean", "fscore. mean",   "recall. std",  "precision. std",  "fscore. std"]

	filename_hh = "results/prediction/multimodal-%s_HH.csv"%(model)
	filename_hr = "results/prediction/multimodal-%s_HR.csv"%(model)


	for filename in [filename_hh, filename_hr]:
		# remove previous output files ir remove == true
		if remove:
			os. system ("rm %s"%filename)
		if not os.path.exists (filename):
			f = open (filename, "w+")
			f.write (';'. join (colnames))
			f. write ('\n')
			f. close ()

	behavioral_variables_hh = pd. read_csv ("concat_time_series/behavioral_hh_data.csv", sep = ';')
	behavioral_variables_hr = pd. read_csv ("concat_time_series/behavioral_hr_data.csv", sep = ';')

	# Predict HH  and HR conversations separetely
	Parallel (n_jobs = 5) (delayed (predict_area)
	(behavioral_variables_hh, target_column, target_column, model = model, lag = lag, filename = filename_hh, find_params = _find_params, type = "HH", all = all, all_m = all_m)
									for target_column in _regions)


	Parallel (n_jobs = 5) (delayed (predict_area)
	(behavioral_variables_hr, target_column, target_column, model = model, lag = lag, filename = filename_hr, find_params = _find_params, type = "HR", all = all, all_m = all_m)
									for target_column in _regions)

if __name__ == '__main__':
	print ("test")
