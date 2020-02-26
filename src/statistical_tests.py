# t-test for independent samples
from math import sqrt
from numpy.random import seed
from numpy.random import randn
from numpy import mean
from scipy.stats import sem
from scipy.stats import t

from ast import literal_eval
import pandas as pd
# function for calculating the t-test for two independent samples
def independent_ttest(mean1, se1, mean2, se2, alpha):
    """
        mean1, se1: mean and standart deviation of first sample
        mean2, se2: mean and standart deviation of second sample
        alpha: significance thresholf

        interpretation: if p > alpha, accept null hypothesis that the means are equal.
    """

    # standard error on the difference between the samples
    sed = sqrt(se1**2.0 + se2**2.0)
    # calculate the t statistic
    t_stat = (mean1 - mean2) / sed
    # degrees of freedom
    df = len(data1) + len(data2) - 2
    # calculate the critical value
    cv = t.ppf(1.0 - alpha, df)
    # calculate the p-value
    p = (1.0 - t.cdf (abs(t_stat), df)) * 2.0
    # return everything
    return t_stat, df, cv, p


def evaluate_features (model, type = "HH", alpha = 0.05):

    # load all behavioral data
    all_data = pd. read_pickle ("concat_time_series/behavioral_hh_data.pkl")

    # Read prediction results
    filename = "results/prediction/%s_%s.tsv"%(model, type)
    df = pd. read_csv (filename, sep = '\t')

    # extract brain areas
    brain_areas = df. loc[:, ["region"]]. values. flatten ()

    # Extract results of best features and all features
    sub_df = df. loc[:, ["region", "selected_predictors", "fscore. mean", "fscore. std"]]
    all_features_results = pd. read_csv ("results/prediction/%s_%s.csv"%(model, type), sep = ';')
    all_features_results_all_br = all_features_results. loc [all_features_results ["predictors_dict"] == "all_minus_best",  ["region", "selected_predictors", "fscore. mean", "fscore. std"]]

    # Evaluate fscores of each brain area
    for br_area in brain_areas:
        print ("\n - Brain area: ", br_area)
        best_features_results = sub_df. loc [sub_df["region"] == br_area]
        all_features_results =  all_features_results_all_br. loc [all_features_results_all_br["region"] == br_area]

        # list of best features and all features
        best_features = literal_eval (best_features_results. loc[:,"selected_predictors"]. values[0])
        all_features = literal_eval (all_features_results. loc[:,"selected_predictors"]. values[0])

        # Extract lagged data from features names
        best_features_data = all_data. loc [:, best_features]
        all_features_data =  all_data. loc [:, all_features]

        print (" - F1-score best features: ", best_features_results. loc[:,"fscore. mean"]. values[0])
        print (" - F1-score all features: ", all_features_results. loc[:,"fscore. mean"]. values[0])

        print ("\n", 25 * '=')


if __name__ == '__main__':
    print ("=========== HUMAN-HUMAN ============")
    evaluate_features ("RF", type = "HH")
    #print ("=========== HUMAN-ROBOT ===========")
    #evaluate_features ("RF", type = "HR")
