import os
import glob
import pandas as pd
import numpy as np
import argparse

from sklearn import preprocessing
from mat4py import loadmat
from sklearn.linear_model import Ridge, Lasso
from sklearn.neural_network import MLPRegressor
from sklearn.decomposition import PCA, IncrementalPCA, KernelPCA

#=============================================
# generate time series from transcriptions files
def process_transcriptions (subject, type = "speech_ts"):
    files = glob. glob ("time_series/%s/%s/*.pkl"%(subject, type))
    return sorted (files)

#===============================================
def generate_stats (subject, type, colnames):
    files = process_transcriptions (subject, type)

    if not os. path. exists ("stats_ts/%s"%subject):
    	os. makedirs ("stats_ts/%s"%subject)

    HH_data = []
    HR_data = []

    for filename in files:
        data = pd. read_pickle (filename)

        if data. shape [0] < 50:
            print ("Subject: %s, the conversation %s have less than 50 lines"%(subject, filename))
        if "CONV1" in filename:
            HH_data. append (data. mean (axis = 0). loc [colnames]. tolist ())
        if "CONV2" in filename:
            HR_data. append (data. mean (axis = 0). loc [colnames]. tolist ())

    if type == "speech_ts":
        type = "speech_right_ts"

    elif type == "speech_left_ts":
        colnames = [a. split ('_')[0] for a in colnames]

    HH_data = pd.DataFrame (HH_data, columns = colnames)
    HR_data = pd.DataFrame (HR_data, columns = colnames)

    return HH_data, HR_data


#=============================================
if __name__ == '__main__':

    parser = argparse. ArgumentParser ()
    if not os. path. exists ("global_results"):
        os. makedirs ("global_results")

    parser. add_argument ("--generate", "-g", help = "generate data", action="store_true")
    args = parser.parse_args()

    colnames_right = ["IPU", "Overlap", "ReactionTime", "FilledBreaks", "Feedbacks", "Discourses", "Particles", "Laughters", "Polarity", "Subjectivity"]
    colnames_left = [a + "_left" for a in colnames_right]
    colnames = colnames_left + colnames_right


    if args. generate:
        #=======================================================
        #   define the subjects to process: the participants
        #=======================================================
        subjects = ["sub-%02d"%i for i in range (1, 25)]
        #subjects = ["sub-11"]
        for sub in ["sub-01", "sub-19", "sub-04",  "sub-16"]:
            if sub in subjects:
                subjects. remove (sub)

        #==========================
        #    Process fMRI data
        #==========================

        fmri_data = loadmat("data/hypothalamus_physsocial.mat")
        fmri_hh = []
        fmri_hr = []
        for i in range (len (subjects)):
            fmri_hh. extend (np. array (fmri_data ["hypothal"][i]) [:,[0,2,4]]. flatten ())
            fmri_hr. extend (np. array (fmri_data ["hypothal"][i]) [:,[1,3,5]]. flatten ())


        #==========================
        #    Process speech data
        #==========================

        if not os. path. exists ("stats_ts"):
        	os. makedirs ("stats_ts")

        speech_hh_data = pd. DataFrame ()
        speech_hr_data = pd. DataFrame ()

        for subject in subjects:
            subj_hh_left, subj_hr_left = generate_stats (subject, "speech_left_ts", colnames_left)
            subj_hh_right, subj_hr_right = generate_stats (subject, "speech_ts", colnames_right)

            subj_hh_data = pd. concat ([subj_hh_left, subj_hh_right], axis = 1, sort=False, ignore_index = True)
            subj_hr_data = pd. concat ([subj_hr_left, subj_hr_right], axis = 1, sort=False, ignore_index = True)

            speech_hh_data = speech_hh_data. append (subj_hh_data, ignore_index = True)
            speech_hr_data = speech_hr_data. append (subj_hr_data, ignore_index = True)

        speech_hh_data. columns = colnames
        speech_hr_data. columns = colnames

        #=====================================
        #   Concatenate data and save them
        #=====================================

        fmri_speech_hh_data = pd. concat ([pd. DataFrame (fmri_hh, columns = ["hypothal"]), speech_hh_data], axis = 1, sort=False)
        fmri_speech_hr_data = pd. concat ([pd. DataFrame (fmri_hr, columns = ["hypothal"]), speech_hr_data], axis = 1)
        fmri_speech_hh_data. to_pickle ("global_results/fmri_speech_hh_data.pkl")
        fmri_speech_hr_data. to_pickle ("global_results/fmri_speech_hr_data.pkl")
        fmri_speech_hh_data. to_csv ("global_results/fmri_speech_hh_data.csv", sep = ';', index = False)
        fmri_speech_hr_data. to_csv ("global_results/fmri_speech_hr_data.csv", sep = ';', index = False)

    #=======================================
    #              REGRESSION
    #=======================================
    else:
        # example: regression on hh data
        data = pd. read_pickle ("global_results/fmri_speech_hh_data.pkl")[["hypothal"] + colnames]

        # Normalisation
        min_max_scaler = preprocessing. MinMaxScaler ()
        data = min_max_scaler. fit_transform (data)

        # build and fit the regression model
        #reg_model = MLPRegressor (hidden_layer_sizes= [5], shuffle = False, activation='logistic')
        reg_model = Ridge (alpha=0.1, normalize = True, random_state = 5)

        # Dimension reduction
        #model = PCA (random_state = 5)
        #X = model. fit_transform (data[:, 1:])
        X = data [:, 1:]
        Y = data [:, 0]
        reg_model.fit (X, Y)

        # prrint regression score
        print ("\n1. Data dimension :\n %s"%str(data.shape))
        print ("\n1. Coefficient of determination (R²):\n%f"%(reg_model. score (X, Y)))
