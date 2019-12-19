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

    if type == "facial_features_ts":
        files = glob. glob ("time_series/%s/%s/*/"%(subject, type))
    else:
        files = glob. glob ("time_series/%s/%s/*.pkl"%(subject, type))



    for i in range (len (files)):
        conv_name = files [i]. split ('/')[-2]
        files [i] += conv_name + ".csv"

    return sorted (files)

#===============================================
def regroupe_data (list_, mode):
    """
        reducing a list of lists into one list by means of
        mean, mode, max, or binary
    """
    for row in list_:
        if mode == "mean":
            return np.nanmean (list_, axis=0). tolist ()
        elif mode == "std":
            return np.std (list_, axis=0). tolist ()
        elif mode == "max":
            return np.nanmax (list_, axis=0). tolist ()
        elif mode == "mode":
            return sc_mode (list_, axis=0, nan_policy = 'omit')[0][0]. tolist ()
        elif mode == "binary":
            res = np.nanmean (list_, axis=0). tolist ()
            for i in range (len (res)):
                if res [i] > 0:
                    res [i] = 1
                else:
                    res [i] = 0
            return res

#===============================================
def generate_stats (subject, type):
    files = process_transcriptions (subject, type)

    mean_data = []
    for filename in files:

        conversation = filename. split ('/')[-1]. split('.') [0]. split ('_')
        conversation = str (conversation[0] + '_' + conversation[2]). split('-')[-1]

        if type == "facial_features_ts":
            data = pd. read_csv (filename, sep = ','). iloc[:,1:]
            FAU = data. loc [:,[" AU01_r", " AU02_r", " AU04_r", " AU05_r", " AU06_r", " AU07_r", " AU09_r",\
                                " AU10_r", " AU12_r"," AU14_r"," AU15_r"," AU17_r"," AU20_r", " AU23_r", " AU25_r", " AU26_r"]]
            POSE =  data. loc[:, [" pose_Tx", " pose_Ty", " pose_Tz", " pose_Rx", " pose_Ry"," pose_Rz"]]
            mean_data. append ([subject,  conversation] + FAU. mean (axis = 0). tolist () + POSE. std (axis = 0). tolist ())
            colnames = ["Subject", "Conversation"] + list (FAU. columns) + list (POSE. columns)

        else:
            data = pd. read_pickle (filename). iloc[:,1:]
            if data. shape [0] < 50:
                print ("Subject: %s, the conversation %s have less than 50 lines"%(subject, filename))
            mean_data. append ([subject,  conversation] + data. mean (axis = 0). tolist ())
            colnames = ["Subject", "Conversation"] + list (data. columns)

    mean_data = pd.DataFrame (mean_data, columns = colnames). sort_values (by = "Conversation")

    return mean_data


#=============================================
if __name__ == '__main__':

    parser = argparse. ArgumentParser ()

    parser. add_argument ("--data_type", "-t", help = "behavioral data type")
    args = parser.parse_args()

    if not os. path. exists ("stats_ts"):
    	os. makedirs ("stats_ts")

    #=======================================================
    #   define the subjects to process: the participants
    #=======================================================
    subjects = ["sub-%02d"%i for i in range (1, 25)]
    for sub in ["sub-01",  "sub-12", "sub-19", "sub-14", "sub-04",  "sub-16"]:
        if sub in subjects:
            subjects. remove (sub)

    data = pd. DataFrame ()

    for subject in subjects:
        #print (subject)
        subj_data = generate_stats (subject,args. data_type)
        data = data. append (subj_data, ignore_index = True)
    #print (hh_data)
    data. to_csv ("stats_ts/%s.csv"%args. data_type, sep = ';', index = False)
