import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
from glob import glob
import os
import matplotlib.ticker as ticker


def color_negative_red (value):
	"""
	Colors elements in a dateframe
	green if positive and red if
	negative. Does not color NaN
	values.
	"""

	if value > 0.8:
		color = 'red'

	else:
		color = 'black'

	return 'color: %s' % color

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

def nb_to_region (region):
	for i in range (len (region)):
		if region[i] == "1":
			region[i] = "FFA"
		if region[i] == "2":
			region[i] = "L-MC"

		if region[i] == "3":
			region[i] = "R-MC"
		if region[i] == "4":
			region[i] = "L-STS"

		if region[i] == "5":
			region[i] = "R-STS"
		if region[i] == "6":
			region[i] = "L-FP"

		if region[i] == "7":
			region[i] = "R-FP"
		if region[i] == "8":
			region[i] = "L-VPFC"
		if region[i] == "9":
			region[i] = "R-VPFC"

#===========================================================#

def model_color (model_name):
	if "GB" in model_name:
		return "indianred"
	elif "SVM" in model_name:
		return "skyblue"
	elif "multi_view" in model_name:
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
		return "grey"

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

def process_multiple_subject (measures_mean, measures_std):

	#os. system ("rm results/prediction/*.pdf*")
	for conv in ["HH", "HR"]:

		evaluation_files = glob ("results/prediction/*%s.tsv*"%conv)

		fig, ax = plt.subplots (nrows = len (measures_mean), ncols = 1, figsize=(16,8),  sharex=True)

		if len (measures_mean) == 1:
			ax = [ax]
			#ax. append (axs)

		bar_with = 0.003
		distance = 0
		local_dist = 0.001

		evaluation_files  = sorted (evaluation_files,  key=str.lower)

		for file in evaluation_files:
			data = pd.read_csv (file, sep = '\t', header = 0, na_filter = False, index_col=False)
			data. sort_values (["region"], inplace = True)

			if data. shape [0] == 0:
				continue

			regions = data .loc [:,"region"]. tolist ()
			x_names = [(i + 1) / 20.0 +  distance for i in range (len (regions))]
			'''regions_names = [region.split ('_')[-1] for region in regions]
			nb_to_region (regions_names)'''
			distance += bar_with + local_dist


			for i in range (len (measures_mean)):
				evaluations = get_eval (data, regions, measures_mean [i])
				errors = get_eval (data, regions, measures_std [i])

				for d in range (len (errors)):
					errors[d] = errors [d] / 2

				model_name = get_model_name (file)

				ax[i]. bar (x_names, evaluations, label = model_name, width = bar_with, capsize=1.5, color = model_color (model_name), yerr = errors, align='center', alpha=0.9, ecolor='black')
				ax[i]. set_ylabel (change_measure_name (measures_mean [i]))
				ax[i]. set_xlabel ("Brain areas")

			for i in range (len (measures_mean)):
				ax[i].xaxis. set_major_locator ((ticker. IndexLocator (base = 1.0 / 20.0, offset= 2 * bar_with)))
				ax[i].yaxis. set_major_locator (ticker. MultipleLocator (0.05))
				ax[i].set_ylim (0, 1)
				ax[i]. set_xticklabels (regions, minor = False, rotation=-10)
				ax[i]. grid (which='major', linestyle=':', linewidth='0.25', color='black')
				#ax[i]. legend (loc='upper right', fancybox=True, shadow=True, ncol=3, fontsize = "small", markerscale = 0.2, labelspacing = 0.1, handletextpad=0.2, handlelength=1)

		#plt.legend (loc='upper right', fancybox=True, shadow=True, ncol=3, fontsize = "small", markerscale = 0.2, labelspacing = 0.1, handletextpad=0.2, handlelength=1)
		#plt.gca().legend (loc='upper center', bbox_to_anchor = (0.5, 1.7), fancybox=True, shadow=True, ncol=5, fontsize = "x-small", markerscale = 0.2, labelspacing = 0.1, handletextpad=0.2, handlelength=1)
		plt.legend (loc='upper right', bbox_to_anchor = (1.1, 1), ncol=1, fontsize = "small", markerscale = 0.2, labelspacing = 0.1, handletextpad=0.2, handlelength=1) #, bbox_transform = plt.gcf().transFigure)
		plt.tight_layout()
		plt. savefig ("results/prediction/results_%s.pdf"%conv)
		#plt.tight_layout()
		#plt. show ()

#===========================================================#

if __name__=='__main__':

	measures_mean = ["fscore. mean"]
	measures_std = ["fscore. std"]
	#measures_mean = ["fscore. mean"]
	#measures_std = ["fscore. std"]

	os. system ("rm results/prediction/html/*.html")
	# Group data by max of fscore to find the best set of predictive variables
	csv_files = glob ("results/prediction/*.csv*")

	for csv_file in csv_files:
		data = pd. read_csv (csv_file, sep = ';', header = 0, na_filter = False, index_col=False)
		data.replace ('', 0, inplace=True)
		data = data. loc [data. groupby ("region") ["fscore. mean"]. idxmax (), :]

		data. sort_index (inplace = True)
		data = data. loc[:, ["region", "dm_method", "lag", "predictors_dict",  "selected_predictors", "recall. mean", "precision. mean", "fscore. mean", "kappa. mean", \
							"recall. std", "precision. std", "fscore. std", "kappa. std"]]

		data.to_csv (csv_file. split ('.')[0]+ ".tsv", sep = '\t', index = False)

		data. loc [:, ["region", "dm_method", "lag", "predictors_dict", "selected_predictors", "fscore. mean"]].to_html (csv_file. split ('.')[0]+ ".html",  index = False, border=False)

	os. system ("mv  results/prediction/*.html results/prediction/html")
	process_multiple_subject (measures_mean, measures_std)

#=============================================================#
