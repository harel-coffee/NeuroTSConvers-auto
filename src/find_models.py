"""
	Author: Youssef Hmamouche
	Year: 2019
	Description:  main execution file. It permits to process (selection, prediction) a set of brain areas.
"""

import os, sys, inspect, argparse
import pandas as pd
from prediction.select_predict import predict_multiple_areas

import warnings
warnings.filterwarnings("ignore")


def prediction (regions, model, method, lag, remove, all, allm):
	if model == 'baseline':
		predict_multiple_areas (regions, "baseline", "None", lag, remove, True, True, False)
		predict_multiple_areas (regions, "baseline", "None", lag, remove, False, True, False)

	elif model in ['MLP', 'LSTM']:
		predict_multiple_areas (regions, lag, model, remove, False, all, allm, method)
	else:
		# k_fold_cross_validation
		predict_multiple_areas (regions, lag, model, remove, True, all, allm, method)
		# evaluate models
		predict_multiple_areas (regions, lag, model, remove, False, all, allm, method)

	print ("... Done.")

if __name__=='__main__':
	parser = argparse. ArgumentParser ()
	#parser._action_groups.pop()
	required = parser.add_argument_group('Required arguments')
	required. add_argument ("--lag", "-p", default = 6, type=int)
	required. add_argument ("--remove", "-rm", help = "remove previous files", action="store_true")
	required. add_argument ("--model", "-m", help = "the prediction model to used", choices = ["baseline", "LSTM", "MLP", "RF", "SVM", "LREG"])
	required. add_argument ('--regions','-rg', nargs = '+', type=int)
	required. add_argument ('--method','-mthd', help = 'The feature selection method to use.', choices = ["None", "K_MEDOIDS", "MI_RANK", "Model_RANK"])
	required. add_argument ("--all", "-all", help = "using all variables as input.", action="store_true", default = True)
	required. add_argument ("--allm", "-allm", help = "using variables with existing hypothesis as input.", action="store_true", default = False)

	args = parser.parse_args()
	print (args)

	if args. model not in ["baseline", "LSTM", "MLP", "RF", "SVM", "LREG"] or args. method not in ["None", "K_MEDOIDS", "MI_RANK", "Model_RANK"] or args. regions is None:
		print ("\n Input arguments are not correct, use -h for details!\n")
		exit (1)
	brain_areas_desc = pd. read_csv ("brain_areas.tsv", sep = '\t', header = 0)

	brain_areas = []
	for num_region in args. regions:
		brain_areas. append (brain_areas_desc . loc [brain_areas_desc ["Code"] == num_region, "Name"]. values [0])

	if not os. path. exists ("results/prediction"):
		os. makedirs ("results/prediction")

	if not os. path. exists ("results/models_params"):
		os. makedirs ("results/models_params")

	prediction (brain_areas, args.model, args.method,  args.lag, args.remove, args. all, args. allm)

	os. system ("python src/evaluation.py")
