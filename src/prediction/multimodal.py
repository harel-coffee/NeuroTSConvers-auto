"""
	Author: Youssef Hmamouche
	Year: 2019
	Prediction multiple brain areas in parallel with a new multimodal classification model
"""

import sys, os, inspect, glob, itertools
from sklearn import preprocessing
from joblib import Parallel, delayed

# for oversampling
from imblearn.over_sampling import RandomOverSampler,  SMOTE, ADASYN

# local files
from src.feature_selection. reduction import manual_selection, generic_reduction, feature_selection_modalities, transform_test_data_modalities
from src.prediction.tools import *
from src.prediction.training import *
from src.prediction.new_model import new_model
from sklearn.preprocessing import KBinsDiscretizer

import warnings
warnings.filterwarnings("ignore")

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
	"ada": AdaBoostClassifier ()
	}

#=============================================================
def get_models_params_from_crossv (crossv_results_filename, brain_area, features_dict, reduction_method):
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
		best_model_params = models_params_. ix [best_model_params_index, "models_params"]
		return literal_eval (best_model_params)

	# find the models_paras associated to each predictors_list with dimension reduction method
	for i in list (models_params. index):
		if set (literal_eval(models_params. loc [i, "predictors_dict"])) == str (features_dict) and models_params. loc [i, "dm_method"] == reduction_method:
			features_exist_in_models_params = True
			best_model_params = models_params. ix [i, "models_params"]
			break

	# find the models_paras associated to each predictors_list without dimension reduction method
	if not features_exist_in_models_params:
		for i in list (models_params. index):
			if set (literal_eval(models_params. loc [i, "predictors_dict"])) == str (features_dict):
				features_exist_in_models_params = True
				best_model_params = models_params. ix [i, "models_params"]
				break

	# else, choose the best model_params without considering features
	if not features_exist_in_models_params:
		best_model_params_index = models_params ["fscore. mean"].idxmax ()
		best_model_params = models_params. ix [best_model_params_index, "models_params"]

	return literal_eval (best_model_params)

#=============================================================
def get_best_model_per_modality (ROI, variables_dict, model, dm_method, convers_type):

	best_models = []
	models_params_file = glob. glob ("results/models_params/*%s_%s.csv" %(model, convers_type. upper ()))[0]

	#for [begin, end] in begin_end:
	for key in variables_dict. keys ():
		#sub_variables = variables_names[begin:end]
		sub_variables = {key: variables_dict [key]}

		params = get_models_params_from_crossv (models_params_file, ROI, sub_variables, dm_method)
		best_model = global_dict_models [model]
		best_model. set_params (**params)
		best_models. append (params)

	return best_models

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
def predict_area (behavioral_variables, target_column, set_of_behavioral_predictors, model, lag, filename, find_params = False, method = "None", type = "HH"):
	"""
		- target_column:
		- set_of_behavioral_predictors:
		- lag: the lag parameter
		- model: the prediction model name
		- filename: where to put the results
		- find_params: if TRUE, a k-fold-cross-validation  is used to find the parameters of the models, else using the previous one stored.
		- method: the feature selection method. None for no feature selection, and rfe for recursive feature elimination.
	"""

	# load target data
	y = pd. read_csv ("concat_time_series/discr_bold_%s_data.csv"%str (type). lower (), sep = ';'). loc[:, [target_column]]

	# Extract the selected features from features selection results
	if os.path.exists ("results/selection/selection_%s.csv" %(type)):
		selection_results = pd.read_csv ("results/selection/selection_%s.csv" %(type),  sep = ';', header = 0, na_filter = False, index_col=False)

	if find_params:
		numb_test = 1
	else:
		numb_test = 5

	for behavioral_predictors in set_of_behavioral_predictors:

		score = []
		lagged_names = get_lagged_colnames (behavioral_predictors, lag) #+ ["Sex", "Age"]
		selected_indices = [a for a in range (len (lagged_names))]

		X = behavioral_variables. loc[:, lagged_names]

		# Concatenate target and predictive features
		all_data = np. concatenate ((y, X), axis = 1)
		#all_data = outlier_detection (all_data, alpha = 0.01)

		'''for j in range (1, all_data. shape[1]):
			discretizer = KBinsDiscretizer (n_bins=2, encode='ordinal', strategy='kmeans')
			all_data[:,j] = discretizer.fit_transform (all_data[:,j:j+1]). flatten ()'''

		#print (all_data)
		#exit (1)

		# test multiple number of features for feature selection_
		if model == "baseline" or method in ["None", ""]:
			set_k = [int (all_data. shape [1] - 1)]

		else:
			set_k = list (range (2, 6))

		for n_comp in set_k:
			print ("%s K = %s ----------"%(method, n_comp))
			score = []

			if method in ["None", ""]:
				dm_method = method
			else:
				dm_method = "%s_%s"%(method, str (n_comp))

			if n_comp >= all_data. shape [1]:
				break

			# numb observations for test data
			nb_obs = int (all_data. shape [0] * (0.2))

			# train test set index
			stratified_indexes = []
			for i in range (numb_test):
				test_index =  range (i * nb_obs, (i + 1) * nb_obs)
				train_index = range (len (all_data))
				train_index = np. delete (train_index, test_index)
				stratified_indexes. append ([train_index, test_index])


			# Make a train test over the list stratified_indexes
			for train_index, test_index in stratified_indexes:
				train_data = all_data [train_index, :].copy ()
				test_data = all_data [test_index, :]. copy ()

				# normalization
				'''min_max_scaler = preprocessing. MinMaxScaler ()
				train_data [:,1:] = min_max_scaler. fit_transform (train_data [:,1:])
				test_data [:,1:] = min_max_scaler. transform (test_data [:,1:])'''

				X_test = test_data [:,1:]
				y_test = test_data [:,0]


				#X_train, y_train = train_data [:,1:], train_data [:,0]
				# Handling oversampling
				ros = RandomOverSampler(random_state=5)
				try:
					X_train, y_train =  ros. fit_resample (train_data [:,1:], train_data [:,0])
					X_test, y_test =  ros. fit_resample (X_test, y_test)

					train_data = np. concatenate ((y_train. reshape ((-1,1)), X_train), axis = 1)
				except:
					print ("Already balanced data")
					pass

				# Get begin and end of each modality
				begin_end = get_begin_end (behavioral_predictors, lag - 2 + 1)

				best_models_modalities = get_best_model_per_modality (target_column, behavioral_predictors, "RF", dm_method, type)


				if method in ["TREE", "RFE", "PCA", "KPCA"]:
					# Feature selection per modality
					X_train, new_begin_end, selectors = feature_selection_modalities (X_train, y_train, begin_end, [n_comp for u in range (len (begin_end))], "TREE", "RF")
					X_test = transform_test_data_modalities (X_test, begin_end, selectors, method = "TREE")

					# Train the model
					pred_model = new_model (new_begin_end, best_models_modalities)
					pred_model. fit (X_train, y_train)
					best_model_params = ""

					selected_indices = []
					for b_e in new_begin_end:
						selected_indices += list (range (b_e [0], b_e [1]))

				else:
					pred_model = new_model (begin_end, best_models_modalities)
					pred_model. fit (X_train, y_train)
					best_model_params = ""

					selected_indices = [a for a in range (len (lagged_names))]

				evaluations = test_model (X_test, y_test, pred_model, lag, model)

				# Compute the score on test data
				score. append (evaluations)


			row = [target_column, dm_method, lag, best_model_params,\
			 			str (dict(behavioral_predictors)), str (lagged_names),  str ([lagged_names [i] for i in selected_indices])] \
						+ np. mean (score, axis = 0). tolist () + np. std (score, axis = 0). tolist ()

			write_line (filename, row, mode = "a")

#=========================================================================
def multimodal_predict (_regions, lag, k, model, remove, _find_params):

	print ("-- MODEL :", model)
	colnames = ["region", "dm_method", "lag", "models_params", "predictors_dict", "predictors_list", "selected_predictors",
				"recall. mean", "precision. mean", "fscore. mean", "accuracy. mean", "recall. std", "precision. std", "fscore. std", "accuracy. std"]

	if _find_params:
		filename_hh = "results/models_params/%s_HH.csv"%(model)
		filename_hr = "results/models_params/%s_HR.csv"%(model)
	else:
		filename_hh = "results/prediction/%s_HH.csv"%(model)
		filename_hr = "results/prediction/%s_HR.csv"%(model)


	for filename in [filename_hh, filename_hr]:
		# remove previous output files if remove == true
		if remove:
			os. system ("rm %s"%filename)
		if not os.path.exists (filename):
			f = open (filename, "w+")
			f.write (';'. join (colnames))
			f. write ('\n')
			f. close ()

	behavioral_variables_hh = pd. read_csv ("concat_time_series/behavioral_hh_data.csv", sep = ';')
	behavioral_variables_hr = pd. read_csv ("concat_time_series/behavioral_hr_data.csv", sep = ';')

	'''if behavioral_variables_hh. isnull (). any (). any () or behavioral_variables_hr. isnull (). any (). any ():
		print (" Warning: data contains nan or very small values!")

	behavioral_variables_hh. fillna (0, inplace = True)
	behavioral_variables_hr. fillna (0, inplace = True)

	behavioral_variables_hh [behavioral_variables_hh < 0.00001] = 0
	behavioral_variables_hr [behavioral_variables_hr < 0.00001] = 0'''

	if behavioral_variables_hh. isnull (). any (). any () or behavioral_variables_hr. isnull (). any (). any ():
		print (" ERROR: data still contains nans after elimination!")

	# Predict HH  data
	Parallel (n_jobs = 6) (delayed (predict_area)
	(behavioral_variables_hh, target_column, manual_selection (target_column), model = model, lag = lag, filename = filename_hh, find_params = _find_params, type = "HH")
									for target_column in _regions)

	# Predict HR  data
	Parallel (n_jobs = 6) (delayed (predict_area)
	(behavioral_variables_hr, target_column, manual_selection (target_column), model = model, lag = lag, filename = filename_hr, find_params = _find_params, type = "HR")
									for target_column in _regions)

if __name__ == '__main__':
	print ("test")
