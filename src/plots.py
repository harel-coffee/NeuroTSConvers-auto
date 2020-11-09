import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
from glob import glob
import os
import matplotlib.ticker as ticker
import math
import ast

def short_region_names (name, brain_areas_desc):
	short_name = brain_areas_desc . loc [brain_areas_desc ["Name"] == name, "ShortName"]. values [0]
	return short_name

#===========================================================
def change_measure_name (measure):
	if measure in ["fscore.mean", "fscore. mean"]:
		return "F-score"
	elif measure in ["precision.mean", "precision. mean"]:
		return "Precision"
	elif measure in ["recall.mean", "recall. mean"]:
		return "Recall"
	else:
		return measure

#===========================================================#
def model_color (model_name):
	if "LSTM" in model_name:
		return "lightcoral"
	elif "SVM" in model_name:
		return "cadetblue"
	elif "multimodal" in model_name:
		return "teal"
	elif "LASSO" in model_name:
		return "red"
	elif "RF" in model_name:
		return "royalblue"
	elif "KNN" in model_name:
		return "green"
	elif "LSTM" in model_name:
		return "teal"
	elif model_name in ["random", "baseline"]:
		return "darkgrey"

#===========================================================#
def get_eval (data, regions, label):

	data = data [data. region.isin (regions)][label]
	return data.values

#===========================================================#
def get_models_names (evaluation_files):
	models_names = []

	for file in evaluation_files:
		models_names. append (file. split('/')[-1]. split ('_')[0])

	return models_names

#===========================================================#
def get_model_name (file):

	model_name = file. split('/')[-1]. split ('_')[0]
	if model_name == "new":
		model_name = "multi_view"
	return model_name

#===========================================================#
def process_multiple_subject (measures_mean, measures_std, out_dir, best_model = True):

	os. system ("rm results/prediction/*.pdf*")
	brain_areas_desc = pd. read_csv ("brain_areas.tsv", sep = '\t', header = 0)

	data_per_model = []

	for conv in ["HH", "HR"]:
		all_data = []

		if best_model:
			evaluation_files = glob ("%s/bestModel_%s.tsv*"%(out_dir, conv)) + glob ("%s/baseline_%s.tsv*"%(out_dir, conv))
		else:
			evaluation_files = glob ("%s/*%s.tsv*"%(out_dir, conv))
			for e_file in evaluation_files:
				if "bestModel_" in e_file:
					evaluation_files. remove (e_file)

		fig, ax = plt.subplots (nrows = len (measures_mean), ncols = 1, figsize=(19,6),  sharex=True)

		if len (measures_mean) == 1:
			ax = [ax]

		bar_with = 0.003
		distance = 0
		local_dist = 0.0005

		evaluation_files  = sorted (evaluation_files,  key=str.lower)

		for file in evaluation_files:

			model_name = get_model_name (file)
			data = pd.read_csv (file, sep = '\t', header = 0, na_filter = False, index_col=False)

			if len (data) == 0:
				continue

			data. sort_values (["region"], inplace = True)
			data = data. assign (model = lambda x : model_name)

			if len (all_data) == 0:
				all_data = data
			else:
				all_data = pd. concat([all_data, data], axis = 0)

			if data. shape [0] == 0:
				continue

			regions = data .loc [:,"region"]. tolist ()
			x_names = [(i + 1) / 60.0 +  distance for i in range (len (regions))]
			regions_names = regions[:]
			#regions_names = [short_region_names(x, brain_areas_desc) for x in regions]
			distance += bar_with + local_dist

			for i in range (len (measures_mean)):
				evaluations = get_eval (data, regions, measures_mean [i])
				errors = get_eval (data, regions, measures_std [i])

				for d in range (len (errors)):
					errors[d] = errors [d] / 2

				ax[i]. bar (x_names, evaluations, label = model_name, width = bar_with, capsize=3, color = model_color (model_name), yerr = errors, align='center', alpha=1, ecolor='black')
				ax[i]. set_ylabel (change_measure_name (measures_mean [i]))
				ax[i]. set_xlabel ("Brain areas")

			for i in range (len (measures_mean)):
				ax[i].xaxis. set_major_locator ((ticker. IndexLocator (base = 1.0 / 60.0, offset= 2 * bar_with)))
				ax[i].yaxis. set_major_locator (ticker. MultipleLocator (0.1))
				ax[i].set_ylim (0, 1)
				ax[i]. set_xticklabels (regions_names, minor = False, rotation=-10, fontsize=9)
				ax[i]. grid (which='major', linestyle=':', linewidth='0.25', color='black')

		#all_data .region = all_data. region. apply (lambda x: short_region_names (x, brain_areas_desc))
		all_data. reset_index (inplace = True)
		all_data = all_data. loc [:, ["region", measures_mean[0], "model"]]
		all_data = all_data. pivot_table (values = [measures_mean[0]], index = "region", columns =  "model")

		all_data.columns. names = ['', '']
		all_data. reset_index (inplace = True)
		data_per_model. append (all_data)

		plt.legend (loc='upper right', fancybox=True, shadow=True, ncol=4, fontsize = "medium", markerscale = 0.2, labelspacing = 0.1, handletextpad=0.2, handlelength=1)
		plt.tight_layout()
		plt. savefig ("%s/results_%s.pdf"%(out_dir, conv))

	return (data_per_model [0]. set_index ("region"). join (data_per_model [1].  set_index ("region"), rsuffix = "_hr", lsuffix = "_hh"))

#===========================================================#
if __name__=='__main__':

	parser = argparse. ArgumentParser ()
	parser. add_argument ("--crossv", "-cv", help = "handle cross-validation results", action="store_true")
	args = parser.parse_args()

	measures_mean = ["fscore. mean", "recall. mean"]
	measures_std = ["fscore. std", "recall. std"]

	if args.crossv:
		working_directory = "results/models_params"
	else:
		working_directory = "results/prediction"

	# generate pdf plots and results of each model
	for m_mean, m_std in zip (measures_mean, measures_std):
		df = process_multiple_subject ([m_mean], [m_std], working_directory, best_model = False)
		df = df.round (2)
		print (df)
		#print (df. to_latex ())
