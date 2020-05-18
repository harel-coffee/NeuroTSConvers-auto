"""
    Author: Youssef Hmamouche
    Year: 2019
    Compute predictions based on pretrained models
"""

import os
import glob
import pandas as pd
import numpy as np
import joblib
import argparse
from ast import literal_eval
import sys

from sklearn.metrics import recall_score, precision_score, f1_score, average_precision_score

class toSuppervisedData:
	targets = np.empty([0,0],dtype=float)
	data = np.empty([0,0],dtype=float)

	## constructor
	def __init__(self, X, p, test_data = False, add_target = False):

		self.targets = self.targets_decomposition (X,p)
		self.data = self.matrix_decomposition (X,p, test_data)

		if not add_target:
			self.data = np. delete (self.data, range (0, p), axis = 1)

		delet = []

		if X.shape[1] > 1 and p > 4:
			for j in range (0, self.data. shape [1], p):
				delet. extend ([j + p - i for i in range (1, 3)])

		self.data = np. delete (self.data, delet, axis = 1)

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

#---------------------------------------------------#
def get_features_from_lagged (lagged_variables):
	features = set (['_'. join (a.split ('_')[0:-1]) for a in lagged_variables])
	return ','. join (map (str, list (features)))


#---------------------------------------------------#
def get_predictors (model_name, region, type, path):
    """
    model_name: name the prediction model
    region: brain area
    type: interaction type (h (human-human) or r (human-robot))
    """
    model_params = pd. read_csv ("%s/results/prediction/%s_H%s.tsv"%(path, model_name, type. upper ()), sep = '\t', header = 0)
    print (model_params. loc [model_params["region"] == "%s"%region]["predictors_dict"])
    predictors = model_params . loc [model_params["region"] == "%s"%region]["predictors_dict"]. iloc [0]

    return predictors

#---------------------------------------------------#
def get_predictors_dict (model_name, region, type, path):
    """
    model_name: name the prediction model
    region: brain area
    type: interaction type (h (human-human) or r (human-robot))
    """
    model_params = pd. read_csv ("%s/results/prediction/%s_H%s.tsv"%(path, model_name, type. upper ()), sep = '\t', header = 0)
    predictors = model_params . loc [model_params["region"] == "%s"%region]["selected_predictors"]. iloc [0]

    return predictors

#---------------------------------------------------#
if __name__ == '__main__':
    parser = argparse. ArgumentParser ()
    requiredNamed = parser.add_argument_group('Required arguments')
    requiredNamed. add_argument ('--regions','-rg', help = "Numbers of brain areas to predict (see brain_areas.tsv)", nargs = '+', type=int)
    requiredNamed. add_argument ('--type','-t', help = ' conversation type (human or robot)')
    requiredNamed. add_argument ('--lag','-lag', default = 6, type=int)
    requiredNamed. add_argument ('--pred_module_path','-pmp', help = "path of the prediction module")
    requiredNamed. add_argument ('--input_dir','-in', help = "path of input directory")

    args = parser.parse_args()

    if args. pred_module_path [-1] == '/':
    	args. pred_module_path = args. pred_module_path [:-1]

    out_dir =  "%s/Outputs/generated_time_series/"%args.input_dir

    # GET REGIONS NAMES FOR THEIR CODES
    brain_areas_desc = pd. read_csv ("brain_areas.tsv", sep = '\t', header = 0)

    regions = []
    for num_region in args. regions:
    	regions. append (brain_areas_desc . loc [brain_areas_desc ["Code"] == num_region, "Name"]. values [0])

    # WRIGHT MULTIMODAL TIME SERIES TO CSV FILE
    all_data = pd. read_csv ("%s/Outputs/generated_time_series/features.csv"%args.input_dir, sep = ';', header = 0)
    columns = all_data. columns

    print ("0")
    lagged_names = []
    for col in columns [1: ]:
    	lagged_names. extend ([col + "_t%d"%(p) for p in range (args. lag, 2, -1)])

    all_data = pd. DataFrame (toSuppervisedData (all_data. values, args. lag). data, columns = lagged_names)

    """ load the best models for each regions """
    if args. type == 'h':
    	conversation_type = 'HH'
    elif args. type == 'r':
    	conversation_type = 'HR'
    else:
    	print ("Error in arguments, type of conversation missing, use -h for help!")
    	exit (1)

    trained_models = glob. glob ("%s/trained_models/*%s.pkl"%(args.pred_module_path,conversation_type))

    # dictionary of predictions: results
    preds = {}
    predictors_variables = {}
    nb_r = 1

    for region in regions:
        print (region, '\n', 18 * '-')
        predictors_data = pd. DataFrame ()
        predictors_data. columns = []
        fname = ""
        for filename in trained_models:
        	if region in  filename:
        		fname = filename
        		break

        model_name = fname. split ('/')[-1]. split ('_') [0]
        #print (model_name)
        model = joblib. load (fname)

        predictors = literal_eval (get_predictors_dict (model_name, region, args. type, args.pred_module_path))
        #print ("Predictors time series: ", predictors, "\n -------------")

        predictors_data = all_data. loc [:, predictors]

        pred = model. predict (predictors_data)

        preds [region] = [0 for i in range (args. lag)] + pred. tolist ()

        # Selected variables without lags
        predictors_variables [region] = get_features_from_lagged (predictors)
        #predictors_variables [region] = literal_eval (get_predictors (model_name, region, args. type, args.pred_module_path))
        #print (region, "\n", 18 * '-')
        print (int (100 * float (nb_r) / len (regions)))
        nb_r += 1
    preds_var = pd.DataFrame ()


    for col in predictors_variables. keys ():
    	preds_var[col] = [str (predictors_variables [col])]

    # time index : fMRI recording frequency
    step = 1.205
    index = [step]
    for i in range (1, args. lag + len (pred)):
    	index. append (index [i - 1] + step)

    pd. DataFrame (preds, index = index). to_csv ("%s/Outputs/predictions.csv"%args.input_dir, sep = ';', index_label = ["Time (s)"])
    preds_var. to_csv ("%s/Outputs/predictors.csv"%args.input_dir, sep = ';', index = False)
