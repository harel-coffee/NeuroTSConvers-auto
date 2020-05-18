"""
	Author: Youssef Hmamouche
	Year: 2019
	Description:  main execution file. It permits to predict a set of brain areas with multiple prediction models
"""

import os, sys, inspect, argparse
import pandas as pd

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
sys.path.append (currentdir)

from src.prediction.prediction import predict_all
from src.prediction.multi_model import multimodal_predict

import warnings
warnings.filterwarnings("ignore")


def prediction (regions, lag, remove, lstm, find_params, all, allm, method):

	if lstm:
		predict_all (regions, lag, "LSTM", remove, find_params, all, allm, method)

	else:
		predict_all (regions, lag, "RF", remove, find_params, all, allm, method)
		#predict_all (regions, lag, "SVM", remove, find_params, all, allm)
		#predict_all (regions, lag, "LREG", remove, find_params, all, allm)
		#predict_all (regions, lag, "MLP", remove, find_params, all, allm)
		#multimodal_predict (regions, lag, "RF", remove, find_params, all, allm)
		#predict_all (regions, lag, "baseline", remove, find_params, all, allm)

	print ("... Done.")

if __name__=='__main__':

	# read arguments
	parser = argparse. ArgumentParser ()
	parser. add_argument ("--lag", "-p", default = 6, type=int)
	parser. add_argument ("--remove", "-rm", help = "remove previous files", action="store_true")
	parser. add_argument ("--lstm", "-lstm", help = "using lstm model", action="store_true")
	parser. add_argument ('--regions','-rg', nargs = '+', type=int)
	parser. add_argument ('--method','-mthd', help = 'dimension reduction method')
	parser. add_argument ("--find_params", "-cv", help = "find the parameters of the models with cross validation", action="store_true")
	parser. add_argument ("--all", "-all", help = "using all variables as input", action="store_true")
	parser. add_argument ("--allm", "-allm", help = "using all variables except best variables as input", action="store_true")

	args = parser.parse_args()

	print (args)

	brain_areas_desc = pd. read_csv ("brain_areas.tsv", sep = '\t', header = 0)

	brain_areas = []
	for num_region in args. regions:
		brain_areas. append (brain_areas_desc . loc [brain_areas_desc ["Code"] == num_region, "Name"]. values [0])

	if not os. path. exists ("results/prediction"):
		os. makedirs ("results/prediction")

	if not os. path. exists ("results/models_params"):
		os. makedirs ("results/models_params")


	prediction (brain_areas, args.lag, args.remove, args.lstm, args. find_params, args. all, args. allm, args.method)
