# t-test for independent samples
from math import sqrt
import numpy as np
from ast import literal_eval
import pandas as pd
import argparse
from glob import glob

from scipy import stats

from evaluation import get_model_name

#========================================================
class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    BRIGHTGREEN = '\033[1;32;40m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    NEGATIVE = '\033[3;37;40m'

#========================================================
def print_df (df_hh, df_hr):
    print ("\n","                    ",bcolors.BRIGHTGREEN, "HUMAN-HUMAN", bcolors.ENDC)

    print (bcolors.NEGATIVE)

    print (df_hh)
    print ( bcolors.ENDC)

    print ("\n","                    ",bcolors.BRIGHTGREEN, "HUMAN-ROBOT", bcolors.ENDC)
    print (bcolors.NEGATIVE)

    print (df_hr)
    print ( bcolors.ENDC)

#===================================================================
def res_all_models (type, files_dir):

    all_data = []
    evaluation_files = glob ("%s/*%s.tsv*"%(files_dir, type))

    for file in evaluation_files:
        model_name = get_model_name (file)
        data = pd.read_csv (file, sep = '\t', header = 0, na_filter = False, index_col=False)

        if data. shape [0] == 0:
            continue

        data. sort_values (["region"], inplace = True)

        data = data. assign (model = lambda x : model_name)
        if len (all_data) == 0:
            all_data = data
        else:
            all_data = pd. concat([all_data, data], axis = 0)

    all_data. reset_index (inplace = True)
    #all_data = all_data. loc [all_data. groupby ("region") ["fscore. mean"]. idxmax (), :]

    return all_data

#========================================================
def get_best_results (brain_area, data):

    #all_results = pd. read_csv (filename, sep = ';')
    best_results = data. loc [data. groupby ("region") ["fscore. mean"]. idxmax (), :]
    best_results = best_results. loc [best_results ["region"] == brain_area, :]

    worst_results = data. loc [data. groupby ("region") ["fscore. mean"]. idxmin (), :]
    worst_results = worst_results. loc [worst_results ["region"] == brain_area, :]

    return best_results, worst_results
#========================================================
def evaluate_baseline (brain_areas, type, files_dir, alpha = 0.05):

    results_df = []
    results_all_models = res_all_models  (type, files_dir)

    best_results_per_roi = results_all_models. loc [results_all_models. groupby ("region") ["fscore. mean"]. idxmax (), :]

    for br_area in brain_areas:
        best_model = best_results_per_roi. loc [best_results_per_roi.region == br_area, "model"]. values[0]
        bestModel_results, _ = get_best_results (br_area, results_all_models. loc [results_all_models.model == best_model])
        baseline_results, _ = get_best_results (br_area, results_all_models. loc [results_all_models.model == "baseline"])
        Fscore_best_model = bestModel_results. loc [:,"fscore. mean"]. values [0]
        Fscore_baseline = baseline_results. loc [:,"fscore. mean"]. values [0]

        if files_dir == "results/prediction":
            pvalue = "Nan"
        else:
            # Extract 10 fscores (10-fold-cross-validation)
            bestF_Kfscores = list (literal_eval (bestModel_results. loc[:,"K-Fscores"]. values[0]))
            baselineF_Kfscores = list (literal_eval (baseline_results. loc[:,"K-Fscores"]. values[0]))
            # Compute Ttest and the pvalue (between model of the best-features  and the model of non-related)
            try:
                #print (bestF_Kfscores, "\n", baselineF_Kfscores)
                _, pvalue = stats.ttest_rel (bestF_Kfscores, baselineF_Kfscores)
            except:
                #print (bestF_Kfscores, "\n", baselineF_Kfscores)
                pvalue = 1

        # alternative0:  independent ttest: stats.ttest_ind()
        # alternative1:  paired related ttest: stats.ttest_rel()
        # alternative2: The Wilcoxon signed-rank test: stats.wilcoxon

        results_df. append ([br_area, best_model, Fscore_best_model, Fscore_baseline, pvalue])
    return (pd. DataFrame (results_df, columns = ["ROIs", "Best model", "Fscore best model", "Fscore baseline", "T-test"]))
#========================================================
def evaluate_features (brain_areas, model, type, files_dir, alpha = 0.05):

    # load all behavioral data
    all_data = pd. read_pickle ("concat_time_series/behavioral_hh_data.pkl")

    filename = "%s/%s_%s.csv"%(files_dir, model, type)

    # all results concatenated
    results_all_models = res_all_models  (type, files_dir)

    # best model for each ROI
    best_results_per_roi = results_all_models. loc [results_all_models. groupby ("region") ["fscore. mean"]. idxmax (), :]

    results_df = []
    # Evaluate fscores of each brain area
    for br_area in brain_areas:

        best_model = best_results_per_roi. loc [best_results_per_roi.region == br_area, "model"]. values[0]
        df = results_all_models. loc [results_all_models.model == best_model]

        best_results, worst_results = get_best_results (br_area, df)

        # list of best features and all features
        best_features = literal_eval (best_results. loc[:,"selected_predictors"]. values[0])
        non_related_features = literal_eval (worst_results. loc[:,"selected_predictors"]. values[0])

        #print (br_area, "Worst features: ", non_related_features)

        # Extract lagged data from features names
        best_features_data = all_data. loc [:, best_features]
        non_related_features_data =  all_data. loc [:, non_related_features]

        bestF_mean = best_results. loc [:,"fscore. mean"]. values [0]
        non_relatedF_mean = worst_results. loc [:,"fscore. mean"]. values [0]

        if files_dir == "results/prediction":
            pvalue = "Nan"
        else:
            # Extract 10 fscores (10-fold-cross-validation)
            bestF_Kfscores = list (literal_eval (best_results. loc[:,"K-Fscores"]. values[0]))
            non_relatedF_Kfscores = list (literal_eval (worst_results. loc[:,"K-Fscores"]. values[0]))
            # Compute Ttest and the pvalue (between model of the best-features  and the model of non-related)
            _, pvalue = stats.ttest_rel (bestF_Kfscores, non_relatedF_Kfscores)

        # alternative0:  independent ttest: stats.ttest_ind()
        # alternative1:  paired related ttest: stats.ttest_rel()
        # alternative2: The Wilcoxon signed-rank test: stats.wilcoxon

        results_df. append ([br_area, best_model, bestF_mean, non_relatedF_mean, pvalue])
        #print ("\n", 25 * '=')

    return (pd. DataFrame (results_df, columns = ["ROIs", "Best model", "Fscore Best features", "Fscore non-related features", "T-test"]))


#==================================================================
if __name__ == '__main__':

    parser = argparse. ArgumentParser ()
    parser. add_argument ('--regions','-rg', nargs = '+', type=int, default = [1, 2, 3, 4, 5, 6])
    parser. add_argument ('--model','-ml', help = "prediction model", default = "RF")
    parser. add_argument ("--crossv", "-cv", help = "handle cross-validation results", action="store_true")
    parser. add_argument ("--baseline", "-b", help = "compare the best model with the baseline", action="store_true")
    args = parser.parse_args()

    brain_areas_desc = pd. read_csv ("brain_areas.tsv", sep = '\t', header = 0)
    brain_areas = []

    print (args. model)

    for num_region in args. regions:
    	brain_areas. append (brain_areas_desc . loc [brain_areas_desc ["Code"] == num_region, "Name"]. values [0])

    #get_best_predictions (type, files_dir)

    if args. baseline:
		if args.crossv:
            files_dir = "results/models_params"
		else:
            _files_dir = "results/prediction"
		df_hh = evaluate_baseline (brain_areas, type = "HH", files_dir = _files_dir)
		df_hr = evaluate_baseline (brain_areas, type = "HR", files_dir = _files_dir)


	else:
		if args.crossv:
		    _files_dir = "results/models_params"
		else:
		    _files_dir = "results/prediction"

		df_hh = evaluate_features (brain_areas, args. model, type = "HH", files_dir = _files_dir)
		df_hr = evaluate_features (brain_areas, args. model, type = "HR", files_dir = _files_dir)


	print_df (df_hh, df_hr)

	df = pd. concat ([df_hh, df_hr. iloc [:,1:]], axis = 1)


	cols = ["T-test"]
	header1 = ["ROIS"] + ["Human-human"] + ["Human-machine"]
	header2 = ["ROIS"] + cols + cols

	header = [np. array (header1), np. array (header2)]

	df = pd.DataFrame (df. iloc[:, [0, 4, 8]].values, columns = header)

    print (df)
