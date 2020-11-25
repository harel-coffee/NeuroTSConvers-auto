import pandas as pd
import numpy as np
import argparse
from glob import glob
import os
import math
from ast import literal_eval
from evaluation import regroupe_features_list, regroupe_features, summarize_features_importance, short_dm_method_name, styling_specific_cell, select_best_model, short_region_names

def filter_fscores (df):
	resumed_df = df. values

	for i in range (len (resumed_df)):
		#for Predictors, Features_importances, Fscore in small_df. values:
		scores = [a for a in literal_eval(resumed_df [i, 4]) if  a >= 0.1]
		features = literal_eval (resumed_df [i, 3])[0:len (scores)]

		resumed_df [i, 3], resumed_df [i, 4] = str (features), str (scores)

	return pd. DataFrame (resumed_df, columns = df. columns)

#========================================================
def reduce_features_importance (V):
	V = literal_eval (V)
	V = [a for a in V if a >= 0.1]
	return str (V)

#===========================================================#
def get_model_name (file):
	model_name = file. split('/')[-1]. split ('_')[0]
	return model_name

#================================================
def res_all_models (type, files_dir):
	all_data = []
	evaluation_files = glob ("%s/*%s.tsv*"%(files_dir, type))

	for file in evaluation_files:
		model_name = get_model_name (file)
		if model_name == "baseline":
			continue

		data = pd.read_csv (file, sep = '\t', header = 0, na_filter = False)
		data = data. loc [:, ["region", "dm_method", "predictors_dict",  "features_importance", "fscore. mean", "recall. mean"]]

		if data. shape [0] == 0:
			continue

		data. sort_values (["region"], inplace = True)

		#data = data. assign (model = lambda x : model_name)
		data.insert (1, 'model', model_name)

		if len (all_data) == 0:
			all_data = data
		else:
			all_data = pd. concat([all_data, data], axis = 0)

	return all_data

#================================================
def df_to_html (df, output_filename):

	# change colnames
	df. columns =  ["ROI", "Classifier", "Feature selection", "Predictors", "Features_importances", "Fscore", "Recall"]
	df. sort_values (["ROI"], inplace = True)
	# Make different colors for each brain area
	ind_colors = [[0,0]]
	regions = df. ROI. values
	#df['index'] = [i for i in range (1, len (df) + 1)]
	df. index = [i for i in range (1, len (df) + 1)]
	#df = df. reindex ([i for i in range (1, len (df) + 1)])
	last_region = regions[0]
	last_color = 0
	for i in range (1, len (df)):
		if regions[i] == last_region:
			ind_colors. append ([i, last_color])
		else:
			last_color = 1 if last_color == 0 else 0
			ind_colors. append ([i, last_color])
		last_region =  regions [i]
	# write html files
	myhtml = df. style. apply (styling_specific_cell, index_colors = ind_colors, colors = ["lightgray", "bisque"], axis = None)
	myhtml = myhtml.set_properties(**{'font-size': '11pt', 'font-family': 'Calibri','border-collapse': 'collapse','border': '1px solid black'}).render()
	pd. set_option ('max_colwidth', 40)
	with open("results/best_results/%s.html"%(output_filename),'w') as f:
			f.write(myhtml)

#================================================
def df_to_latex (df, output_filename):
	df. sort_values (by = "ROI", inplace = True, ignore_index = True)
	with pd.option_context("max_colwidth", 1000):
		df. to_latex (buf = "file.txt", multirow = True, index = False)
		bestModel_file = "results/best_results/%s"%output_filename
		command = "tr -s \" \" < file.txt > %s"%bestModel_file
		os.system (command)
		os.system ("rm file.txt")
#================================================
if __name__ == '__main__':

	os. system ("rm results/best_results/*.html")
	os. system ("rm results/best_results/*.txt")
	ttest_results = pd. read_csv ("results/ttest_pvalues.tsv", sep = "\t")
	brain_areas_desc = pd. read_csv ("brain_areas.tsv", sep = '\t', header = 0)

	working_directory = "results/prediction"
	for conv in ["HH", "HR"]:

		significant_areas = ttest_results. loc [ttest_results[conv] < 0.05, "ROIs"]. values

		results = res_all_models (conv, working_directory)
		#results. region = results. region. apply (lambda x: short_region_names (x, brain_areas_desc))
		results = regroupe_features (results)
		#results.dm_method = results.dm_method.apply (short_dm_method_name)

		best_results = results. loc [results. region. isin (significant_areas), :]

		best_results_strict = filter_fscores (select_best_model (best_results, 0.0))
		best_results = filter_fscores (select_best_model (best_results))

		text_results = filter_fscores (select_best_model (results, 0.0))
		text_results. columns =  ["ROI", "Classifier", "Feature selection", "Predictors", "Features_importances", "Fscore", "Recall"]

		text_all_results = filter_fscores (results. loc [results. region. isin (significant_areas), :])
		text_all_results. columns =  ["ROI", "Classifier", "Feature selection", "Predictors", "Features_importances", "Fscore", "Recall"]
		text_all_results. sort_values (by = ["ROI", "Classifier"], inplace = True)

		df_to_html (results, "results_all_models_%s"%conv)
		df_to_html (best_results, "best_results_%s"%conv)
		df_to_html (best_results_strict, "best_results_stricts_%s"%conv)

		df_to_latex (best_results_strict. loc [:, ["ROI", "Classifier", "Feature selection", "Predictors", "Features_importances", "Fscore"]],  "best_results_%s.txt"%conv)

		df_to_latex (text_all_results. loc [:, ["ROI", "Classifier",  "Predictors"]],  "all_results_%s.txt"%conv)

		#df_to_latex (best_results_strict. loc [:, ["ROI", "Feature selection", "Predictors", "Features_importances"]],  "best_results_strict_%s.txt"%conv)
