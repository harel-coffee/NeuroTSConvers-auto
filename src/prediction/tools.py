import sys
import os

import numpy as np
import pandas as pd
import random as rd
from ast import literal_eval

from sklearn.decomposition import PCA, IncrementalPCA, KernelPCA
from sklearn.ensemble import IsolationForest

from sklearn.cluster import KMeans

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
def extract_models_params_from_crossv (crossv_results_filename, brain_area, features, reduction_method):
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
		best_model_params = models_params_. loc [best_model_params_index, "models_params"]
		std_errors =  models_params. loc [best_model_params_index, ["recall. std",  "precision. std",  "fscore. std",  "kappa. std"]]
		return literal_eval (best_model_params), std_errorss

	# find the models_paras associated to each predictors_list with dimension reduction method
	for i in list (models_params. index):
		if set (literal_eval(models_params. loc [i, "predictors_list"])) == set (features) and models_params. loc [i, "dm_method"] == reduction_method:
			features_exist_in_models_params = True
			best_model_params_index = i
			#best_model_params = models_params. loc [i, "models_params"]
			break

	# find the models_paras associated to each predictors_list without dimension reduction method
	if not features_exist_in_models_params:
		for i in list (models_params. index):
			if set (literal_eval(models_params. loc [i, "predictors_list"])) == set (features):
				features_exist_in_models_params = True
				best_model_params_index = i
				#best_model_params = models_params. loc [i, "models_params"]
				break

	# else, choose the best model_params without considering features
	if not features_exist_in_models_params:
		best_model_params_index = models_params ["fscore. mean"].idxmax ()


	best_model_params = models_params. loc [best_model_params_index, "models_params"]
	std_errors =  models_params. loc [best_model_params_index, ["recall. std",  "precision. std",  "fscore. std",  "kappa. std"]]. values

	return literal_eval (best_model_params), std_errors. tolist ()

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

#=======================================================

def get_behavioral_data (subject, convers, external_predictors):
	'''
	concatenate data of one conversation with multiple modalities
	'''
	external_data = np. array ([])
	external_filenames = ["time_series/%s/%s/%s.pkl"%(subject, data_type, convers) for data_type in external_predictors. keys ()]
	external_columns = [external_predictors[item] for item in external_predictors. keys ()]

	for filename, columns in zip (external_filenames, external_columns):

		read_file = pd.read_pickle (filename)[columns]. values
		if read_file. shape[0] < 50:
			print ("Error in %s %s %s"%(subject, convers, external_predictors))
			exit (1)
		if external_data. size == 0:
			if os. path. exists (filename):
				try:
					external_data = read_file
				except:
					continue
		else:
			if os. path. exists (filename):
				try:
					external_data = np. concatenate ((external_data, read_file), axis = 1)
				except:
					continue

	return external_data

#============================================================

def shuffle_data_by_blocks (data, block_size):

	blocks = [data [i : i + block_size] for i in range (0, len(data), block_size)]
	# shuffle the blocks
	rd.shuffle (blocks)
	# concatenate the shuffled blocks
	output = [b for bs in blocks for b in bs]
	output = np. array (output)
	return output


#============================================================

def train_test_split (data, test_size = 0.2):

	nb_obs = int (data. shape [0] * (1 - test_size))
	train_d = data [0 : nb_obs, :]
	test_d = data [nb_obs :, :]

	return train_d, test_d

#============================================================

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
		#lagged_columns. extend ([item + "_sum"])
		#lagged_columns. extend ([item + "_t%d"%(p) for p in range (lag, 0, -1)])

	return (lagged_columns)

#=========================================================================================
def concat_ (subject, target_column, convers, lag, behavioral_predictors, add_target = False, reg = False):

	data = pd. DataFrame ()

	if subject == "sub-01":
		convers = convers [0:9]

	for conv in convers:

		if reg:
			filename = "time_series/%s/physio_ts/%s.pkl"%(subject, conv)
		else:
			filename = "time_series/%s/discretized_physio_ts/%s.pkl"%(subject, conv)

		target = pd.read_pickle (filename). loc [:,[target_column]]. values

		# Load neurophysio and behavioral predictors
		if len (behavioral_predictors) > 0:
			external_data = get_behavioral_data (subject, conv, behavioral_predictors)
		else:
			external_data = None

		if external_data. shape[0] == 0:
			continue

		# add to rows to behavioral features
		external_data = np. append (external_data, external_data[-2:,:], axis = 0)
		#external_data = external_data. append (external_data[-1])

		if target. shape [0] < external_data. shape [0]:
			external_data = external_data[0:target. shape [0],:]

		# smoothing
		#target = target[1:]
		#external_data = np. diff (external_data, axis = 0)

		# concatenate data of all conversations
		if data.shape [0] == 0:
			data = np. concatenate ((target, external_data), axis = 1)#[8:-8, :]
			if lag > 0:
				supervised_data = toSuppervisedData (data, lag, add_target = add_target)
				data = np.concatenate ((supervised_data. targets[:,0:1], supervised_data.data), axis = 1)

		else:
			#U = pd. concat ([target, external_data], axis = 1)
			U = np. concatenate ((target, external_data), axis = 1)#[8:-8, :]
			if lag > 0:
				supervised_U = toSuppervisedData (U, lag, add_target = add_target)
				U = np.concatenate ((supervised_U. targets[:,0:1], supervised_U.data), axis = 1)
			data =  np. concatenate ((data, U), axis = 0)

		# DEBUG CONVERSATION
		'''if data. shape [1] == 0:
			print (18 * "=")
			print (conv)
			print (subject)
			exit (1)'''

	return data

#=======================================================
class toSuppervisedData:
	targets = np.empty([0,0],dtype=float)
	data = np.empty([0,0],dtype=float)

	## constructor
	def __init__(self, X, p, test_data = False, add_target = False):

		self.targets = self.targets_decomposition (X, p)
		self.data = self.matrix_decomposition (X, p, test_data)

		if not add_target:
			self.data = np. delete (self.data, range (0, p), axis = 1)

		delet = []

		if X.shape[1] > 0 and p > 4:
			for j in range (0, self.data. shape [1], p):
				# delete 3 first lagged variables (keep those close to 5s)
				delet. extend ([j + p - i for i in range (1, 3)])

		self.data = np. delete (self.data, delet, axis = 1)

		# compute the mean of lagged variables
		n_var = int (self.data. shape [1] / (p - 2))# + self.data. shape [1]

		#new_data = np.empty ([self.data. shape [0], n_var]) #, dtype = np. float64)
		#new_data = np.empty ([self.data. shape [0], n_var])
		new_data = np.array ([])

		for j in range (0, self.data. shape [1], p - 2):
			# [0, 1, 2] is equivalent to t-3, t-4, t-5
			cols = self.data [:, [j + i for i in range (p - 2)]]

			#for i in range (4):
				#new_data [:, j + i] = self. data [:, j + i]
			#new_data [:, j : j + 4] = self.data [:, j : j + 4]

			if len (new_data) == 0:
				new_data = self.data [:, j : j + (p - 2)]
				#new_data = np. sum (self.data [:, j : j + 4], axis = 1). reshape (-1, 1)
			else:
				new_data = np. append (new_data, self.data [:, j : j + p - 2], axis = 1)
				#new_data = np. append (new_data, np. sum (self.data [:, j : j + 4], axis = 1). reshape (-1, 1), axis = 1)



			new_data = np. append (new_data, np. sum (self.data [:, j : j + p - 2], axis = 1). reshape (-1, 1), axis = 1)
			#new_data [:, j + 4] =  np. sum (self.data [:, j : j + 4], axis = 1)

			#new_data [:, int (j / 4)] = np. sum (cols, axis = 1)
			#new_data [:, int (j / 4) + 1] = np. std (cols, axis = 1)
			#new_data [:, int (j / 4)] = self.data [:, j + 2]


		#print (pd. DataFrame (new_data))
		#exit (1)
		self.data = new_data

	## p-decomposition of a vector
	def vector_decomposition (self, x, p, test = False):
		n = len(x)
		if test:
			add_target_to_data = 1
		else:
			add_target_to_data = 0

		output = np.empty([n-p,p],dtype=float)

		for i in range (n-p):
			for j in range (p):
				output[i,j] = x[i + j + add_target_to_data]
		return output

	# p-decomposition of a target
	def target_decomposition (self,x,p):
		n = len(x)
		output = np.empty([n-p,1],dtype=float)
		for i in range (n-p):
			output[i] = x[i+p]
		return output

	# p-decomposition of a matrix
	def matrix_decomposition (self,x,p, test=False):
		output = np.empty([0,0],dtype=float)
		out = np.empty([0,0],dtype=float)

		for i in range(x.shape[1]):
			out = self.vector_decomposition(x[:,i],p, test)
			if output.size == 0:
				output = out
			else:
				output = np.concatenate ((output,out),axis=1)

		return output
	# extract all the targets decomposed
	def targets_decomposition (self,x,p):
		output = np.empty([0,0],dtype=float)
		out = np.empty([0,0],dtype=float)
		for i in range(x.shape[1]):
			out = self.target_decomposition(x[:,i],p)
			if output.size == 0:
				output = out
			else:
				output = np.concatenate ((output,out),axis=1)
		return output
