import pandas as pd
import numpy as np
import argparse
from glob import glob
import os
import math
from ast import literal_eval
#===========================================
def summarize_features_importance (features, importance):

	empty_scores = False
	if len (importance) == 0:
		empty_scores = True
		importance = [0 for a in features]
		#return features, importance

	df = pd. DataFrame ()
	df ["feats"] = features
	df ["scores"] = importance
	df = df. groupby ('feats')["scores"].sum (). reset_index (). round (2)

	df = df. sort_values (by = "scores", ascending = False). values

	feats = df[:,0]
	scores = df[:,1]

	if empty_scores:
		return str (df[:,0]. tolist ()), ""
	else:
		return str (df[:,0]. tolist ()), str (df[:,1]. tolist ())
#===========================================

def regroupe_features_list (V_):

	V = literal_eval (V_)
	i = 0
	for a in V:
		if a in ["Head_Rx_I", "Head_Rz_I", 'Head_Ry_I', 'Head_Tx_I', 'Head_Ty_I', 'Head_Tz_I', 'Head_translation_energy_I', 'Head_rotation_energy_I']:
			V[i] = "Head-movement_I"
		elif a in ["AU_all_I",  'AU_mouth_I', 'AUs_mouth_I',  'AU_eyes_I', 'Neutral_I']:
			V[i] = "Facial-movement_I"
		elif a in ['Angry_I', 'Disgust_I', 'Fear_I', 'Happy_I','Sad_I', 'Surprise_I', 'Smiles_I']:
			V[i] = "Emotions_I"
		elif a in ["Gaze_speed_P", "Gaze_movement_energy_P", "Saccades_P"]:
			V[i] = "Eyetracking_P"
		elif a in ['Face_looks_P', 'Mouth_looks_P', 'Eyes_looks_P', 'Direct_gaze_I']:
			V[i] = "Social-gaze"
		elif a in ["FilledBreaks_I", "Feedbacks_I", "Discourses_I", "Discourse_Markers_I", "Particles_I", "Spoken_Particles_I", "Laughters_I","Interpersonal_I", "UnionSocioItems_I", "Polarity_I", "Subjectivity_I", "ReactionTime_I"]:
			V[i] = "Interpersonal_I"
		elif a in ["FilledBreaks_P", "Feedbacks_P", "Discourses_P","Discourse_Markers_P", "Particles_P", "Spoken_Particles_P", "Laughters_P", "UnionSocioItems_P", "Interpersonal_P", "Polarity_P", "Subjectivity_P", "ReactionTime_P"]:
			V[i] = "Interpersonal_P"
		elif a in ["IPU_I", "disc_IPU_I", "SpeechActivity_I", "Overlap_I", "SpeechRate_I"]:
			V[i] = "SpeechActivity_I"
		elif a in ["IPU_P", "disc_IPU_P", "SpeechActivity_P", "Overlap_P", "SpeechRate_P"]:
			V[i] = "SpeechActivity_P"
		elif a in ["LexicalRichness_I", "TypeTokenRatio_I",  "TypeToken_Ratio_I"]:
			V[i] = "Linguistic-Complexity_I"
		elif a in ["LexicalRichness_P", "TypeTokenRatio_P",  "TypeToken_Ratio_P"]:
			V[i] = "Linguistic-Complexity_P"
		else:
			print ("Unknown feature %s!"%a)
			exit (1)
		i = i + 1

	return V
#=========================================
def regroupe_features (html_df):
	resumed_df = html_df. values

	for i in range (len (resumed_df)):
		#for Predictors, Features_importances, Fscore in small_df. values:
		Predictors = regroupe_features_list (resumed_df [i, 3])
		Features_importances = literal_eval (resumed_df [i, 4])
		resumed_df [i, 3], resumed_df [i, 4] = summarize_features_importance (Predictors, Features_importances)

	return pd. DataFrame (resumed_df, columns = html_df. columns)

#=========================================
def short_dm_method_name (name):
	if ("model_select" in name) or ("TREE" in name):
		return "WFS"
	if len (name. split ('_')) > 0:
		return ('_').join (name. split ('_')[0:-1]).lower ()
	else:
		return name
#=========================================

def short_region_names (name, brain_areas_desc):
	try:
		short_name = brain_areas_desc . loc [brain_areas_desc ["Name"] == name, "ShortName"]. values [0]
		return short_name
	except:
		pass

#===========================================================
def replace_some_features_names (V):
	V = literal_eval (V)

	for i in range (len (V)):
		if "UnionSocioItems" in V[i]:
			V[i] = V[i]. replace ("UnionSocioItems", "Interpersonal")
		elif V[i]. split ('_')[0] == "Discourses":
			V[i] = V[i]. replace ("Discourses", "Discourse_Markers")
		elif V[i]. split ('_')[0] == "Particles":
			V[i] = V[i]. replace ("Particles", "Spoken_Particles")
		elif "TypeTokenRatio" in V[i]:
			V[i] = V[i]. replace ("TypeTokenRatio", "TypeToken_Ratio")
	return str (V)
#=============================================
def replace_fs_methods (V):
	V = literal_eval (V)
	for i in range (len (V)):
		if "k_medoids" in V[i]:
			V[i] = V[i]. replace ("k_medoids", "k-medoids")
		if "model_select" in V[i]:
			V[i] = V[i]. replace ("model_select", "WFS")
		if "mi_rank" in V[i]:
			V[i] = V[i]. replace ("mi_rank", "mi-rank")

	return str (V)

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

	return df_group#.reset_index ()


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

		data = pd.read_csv (file, sep = '\t', header = 0, na_filter = False, index_col=0)

		if data. shape [0] == 0:
			continue

		data. sort_values (["region"], inplace = True)

		data = data. assign (model = lambda x : model_name)
		if len (all_data) == 0:
			all_data = data
		else:
			all_data = pd. concat([all_data, data], axis = 0)

	all_data. reset_index (inplace = True)

	return all_data

#============================================================
def select_best_model (all_results, threshold = 0.03):

	best_results = pd.DataFrame ()
	regions =  list (set (all_results. loc[:, "region"]. tolist ()))
	for roi in regions:
		roi_data = all_results. loc [all_results.region == roi,:]
		max_fscore = roi_data. loc [:,"fscore. mean"]. max()
		roi_data = roi_data [roi_data["fscore. mean"] >= max_fscore - threshold]
		best_results = best_results. append (roi_data)

	return best_results

#===========================================================#
def styling_specific_cell(x, index_colors, colors):

	df_styler = pd.DataFrame('', index=x.index, columns=x.columns)
	for [i, t] in index_colors:
		df_styler.iloc[i, :] = 'background-color: %s; color: black'%colors[t]
	return df_styler

#===========================================================#
if __name__=='__main__':
	'''
		generate resumed results for each model, csv -> tsv
	'''
	parser = argparse. ArgumentParser ()
	parser. add_argument ("--crossv", "-cv", help = "handle cross-validation results", action="store_true")
	args = parser.parse_args()

	brain_areas_desc = pd. read_csv ("brain_areas.tsv", sep = '\t', header = 0)

	measures_mean = ["fscore. mean"]
	measures_std = ["fscore. std"]

	if args.crossv:
		working_directory = "results/models_params"
		colnames = ["region", "dm_method", "lag", "models_params", "predictors_dict",  "selected_predictors", "accuracy. mean", "recall. mean", "fscore. mean", \
							"accuracy. std", "recall. std", "fscore. std", "K-Fscores"]
	else:
		working_directory = "results/prediction"
		colnames = ["region", "dm_method", "lag", "models_params", "predictors_dict",  "selected_predictors",  "features_importance", "accuracy. mean", "recall. mean", "fscore. mean", \
							"accuracy. std", "recall. std", "fscore. std"]

	csv_files = glob ("%s/*.csv*"%working_directory)

	# process each model separately
	for csv_file in csv_files:
		data = pd. read_csv (csv_file, sep = ';', header = 0, na_filter = False, index_col=False). loc[:, colnames]
		if data.empty:
			continue

		data.replace ('', 0, inplace=True)
		data["fscore. mean"] = data["fscore. mean"]. round (2)
		data["recall. mean"] = data["recall. mean"]. round (2)
		data["nb_predictors"] = data["predictors_dict"]
		data["nb_predictors"] = data["nb_predictors"]. apply (lambda x: len (literal_eval (x)))
		data = groupby_max_fscore (data)

		data = data. loc [data. groupby (["region"]) ["nb_predictors"]. idxmin ()]
		data. region = data. region. apply (lambda x: short_region_names (x, brain_areas_desc))
		data.dm_method = data.dm_method.apply (short_dm_method_name)
		data.to_csv ("%s.tsv"%csv_file. split ('.')[0], sep = '\t', index = False)


	if not args. crossv:
		os.system ("rm results/best_results/*")

		for conv in ["HH", "HR"]:
			results_all_models = res_all_models (conv, working_directory)
			best_results_per_roi = select_best_model (results_all_models)
			best_results_per_roi["fscore. mean"] = best_results_per_roi["fscore. mean"]. round (2)
			best_results_per_roi["nb_predictors"] = best_results_per_roi["features_importance"]
			best_results_per_roi["nb_predictors"] = best_results_per_roi["nb_predictors"]. apply (lambda x: len (literal_eval (x)))

			# to csv file
			best_results_per_roi = best_results_per_roi. loc [:, ["region", "dm_method", "model", "predictors_dict",  "selected_predictors", "features_importance", "fscore. mean", "recall. mean"]]
			best_results_per_roi. to_csv ("results/best_results/bestModel_%s.tsv"%(conv), sep = '\t', index = False)

		os.system ("cp -r results/best_results results/last_best_results")
