import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
from glob import glob
import os
import matplotlib.ticker as ticker
import math

#===========================================================
def groupby_max_fscore (df):
	regions = np.unique (df.loc[:,"region"].values)
	df_group = []

	for region in regions:
		df_rg = df.loc [df["region"] == region]
		max_fscore = np.max (df_rg.loc[:,"fscore. mean"].values)
		sub_df_group = df.loc [(df["region"] == region) & (df["fscore. mean"] == max_fscore)]

		if len (df_group) == 0:
			df_group = sub_df_group
		else:
			df_group = pd.concat ([df_group, sub_df_group], axis = 0)

	return df_group.reset_index ()

#===========================================================
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
		if region[i] in ["1", "LeftMotor"]:
			region[i] = "Left MC"

		if region[i] == "LeftTemporoParietalJunction":
			region[i] = "LeftTPJ"

		if region[i] == "RightTemporoParietalJunction":
			region[i] = "RightTPJ"

		if region[i] in ["2", "RightMotor"]:
			region[i] = "Right MC"

		if region[i] in ["3", "LeftSTS"]:
			region[i] = "Left STS"

		if region[i] in ["4", "RightSTS"]:
			region[i] = "Right STS"

		if region[i] in ["5", "LeftFusiformGyrus"]:
			region[i] = "Left FG"

		if region[i] in ["6", "RightFusiformGyrus"]:
			region[i] = "Right FG"


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

def process_multiple_subject (measures_mean, measures_std, out_dir):

	#os. system ("rm results/prediction/*.pdf*")

	data_per_model = []
	for conv in ["HH", "HR"]:

		all_data = []

		evaluation_files = glob ("%s/*%s.tsv*"%(out_dir, conv))

		fig, ax = plt.subplots (nrows = len (measures_mean), ncols = 1, figsize=(10,4),  sharex=True)

		if len (measures_mean) == 1:
			ax = [ax]
			#ax. append (axs)

		bar_with = 0.003
		distance = 0
		local_dist = 0.0005

		evaluation_files  = sorted (evaluation_files,  key=str.lower)

		for file in evaluation_files:

			model_name = get_model_name (file)
			data = pd.read_csv (file, sep = '\t', header = 0, na_filter = False, index_col=False)
			data. sort_values (["region"], inplace = True)


			data = data. assign (model = lambda x : model_name)
			if len (all_data) == 0:
				all_data = data
			else:
				all_data = pd. concat([all_data, data], axis = 0)

			if data. shape [0] == 0:
				continue

			regions = data .loc [:,"region"]. tolist ()
			'''try:
				regions.remove ("LeftMotor")
				regions. remove ("LeftSTS")
			except:
				pass'''

			x_names = [(i + 1) / 30.0 +  distance for i in range (len (regions))]

			regions_names = regions[:]
			nb_to_region (regions_names)

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
				ax[i].xaxis. set_major_locator ((ticker. IndexLocator (base = 1.0 / 30.0, offset= 2 * bar_with)))
				ax[i].yaxis. set_major_locator (ticker. MultipleLocator (0.05))
				ax[i].set_ylim (0, 1)
				ax[i]. set_xticklabels (regions_names, minor = False, rotation=-10, fontsize=12)
				#ax[i]. set_xticklabels (regions, minor = False, rotation=-10, fontsize=12)
				ax[i]. grid (which='major', linestyle=':', linewidth='0.25', color='black')
				#ax[i]. legend (loc='upper right', fancybox=True, shadow=True, ncol=3, fontsize = "small", markerscale = 0.2, labelspacing = 0.1, handletextpad=0.2, handlelength=1)


		all_data. reset_index (inplace = True)
		all_data = all_data. loc [:, ["region", "fscore. mean", "model"]]
		all_data = all_data. pivot_table (values = ["fscore. mean"], index = "region", columns =  "model")


		all_data.columns. names = ['', '']
		all_data. reset_index (inplace = True)
		#print (all_data)
		data_per_model. append (all_data)

		plt.legend (loc='upper right', fancybox=True, shadow=True, ncol=4, fontsize = "medium", markerscale = 0.2, labelspacing = 0.1, handletextpad=0.2, handlelength=1)
		plt.tight_layout()
		plt. savefig ("%s/results_%s.pdf"%(out_dir, conv))
		plt. show ()


	return (data_per_model [0]. set_index ("region"). join (data_per_model [1].  set_index ("region"), rsuffix = "_hr", lsuffix = "_hh"))

	#return data_per_model

#===========================================================#

if __name__=='__main__':

	parser = argparse. ArgumentParser ()
	parser. add_argument ("--crossv", "-cv", help = "handle cross-validation results", action="store_true")
	args = parser.parse_args()

	measures_mean = ["fscore. mean"]
	measures_std = ["fscore. std"]


	if args.crossv:
		working_directory = "results/models_params"
		colnames = ["region", "dm_method", "lag", "models_params", "predictors_dict",  "selected_predictors", "recall. mean", "precision. mean", "fscore. mean", \
							"recall. std", "precision. std", "fscore. std", "K-Fscores"]
	else:
		working_directory = "results/prediction"
		colnames = ["region", "dm_method", "lag", "models_params", "predictors_dict",  "selected_predictors", "recall. mean", "precision. mean", "fscore. mean", "features_importance", \
							"recall. std", "precision. std", "fscore. std"]


	os. system ("rm %s/html/*.html"%working_directory)

	csv_files = glob ("%s/*.csv*"%working_directory)

	for csv_file in csv_files:
		print (csv_file)
		data = pd. read_csv (csv_file, sep = ';', header = 0, na_filter = False, index_col=False). loc[:, colnames]
		if data.empty:
			continue
		data.replace ('', 0, inplace=True)
		print (data.columns)
		data["fscore. mean"] = data["fscore. mean"]. round (2)
		data["nb_predictors"] = data["dm_method"]
		data["nb_predictors"] = data["nb_predictors"]. apply (lambda x: int (x.split ('_')[-1]))
		data = groupby_max_fscore (data)
		data = data. loc [data. groupby (["region"]) ["nb_predictors"]. idxmin ()]


		data.to_csv (csv_file. split ('.')[0]+ ".tsv", sep = '\t', index = False)

		if not args. crossv:
			pd.set_option('display.max_colwidth', 10)
			html_data = data. loc [:, ["region", "lag", "dm_method", "predictors_dict",  "features_importance", "fscore. mean"]]
			html_data["dm_method"] = html_data["dm_method"].apply (lambda x: ('_').join (x.split ('_')[:-1]))
			html_data. columns = ["ROI", "Lag","dm_method", "Predictors", "Features_importances", "Fscore"]
			html_data. to_html (csv_file. split ('.')[0] + ".html",  index = False, border=True)


			myhtml = html_data.style.set_properties(**{'font-size': '12pt', 'font-family': 'Calibri','border-collapse': 'collapse','border': '1px solid black'}).render()

			with open(csv_file. split ('.')[0] + ".html",'w') as f:
			    f.write(myhtml)

		if not args. crossv:
			os. system ("mv  %s/*.html %s/html"%(working_directory, working_directory))


	df = process_multiple_subject (measures_mean, measures_std, working_directory)
	df = df.round (2)

	from tabulate import tabulate

	print (df)
	#print (print(df.to_latex(index=True, multirow = True)))

#=============================================================#
