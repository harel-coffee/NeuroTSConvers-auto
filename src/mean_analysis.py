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

from resampling import compute_energy

#=============================================
def list_files (subject, type = "speech_ts"):

    files = []
    if subject == 'sub-01':
        blocks = range (1, 4)
    else:
        blocks = range (1, 5)

    for block in blocks:
        for conv in [1, 2, 3, 4, 5, 6]:
            conv_type = conv % 2
            if conv_type == 0:
                conv_type = 2
            files. append ('time_series/%s/%s/convers-TestBlocks%d_CONV%d_00%d.pkl'%(subject, type, block, conv_type, conv))

    '''if type == "facial_features_ts":
        files = glob. glob ("time_series/%s/%s/*/"%(subject, type))
    else:
        files = glob. glob ("time_series/%s/%s/*.pkl"%(subject, type))'''

    if type == "facial_features_ts":
        for i in range (len (files)):
            files [i] = files [i]. split ('.')[0]
            conv_name = files [i]. split ('/')[-1]
            files [i] += '/' + conv_name + ".csv"

    #print (files)
    #exit (1)
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
    files = list_files (subject, type)

    mean_data = []
    for filename in files:

        conversation = filename. split ('/')[-1]. split('.') [0]. split ('_')
        conversation = str (conversation[0] + '_' + conversation[2]). split('-')[-1]

        if type == "facial_features_ts":
            data = pd. read_csv (filename, sep = ','). iloc[:,1:]
            FAU = data. loc [data[" success"] == 1,[" AU01_r", " AU02_r", " AU04_r", " AU05_r", " AU06_r", " AU07_r", " AU09_r",\
                                " AU10_r", " AU12_r"," AU14_r"," AU15_r"," AU17_r"," AU20_r", " AU23_r", " AU25_r", " AU26_r"]]

            #head_translation = compute_energy (data. loc [data[" success"] == 1, [" pose_Tx", " pose_Ty", " pose_Tz"]]. values)
            head_translation = compute_energy (data. loc [data[" success"] == 1, [" x_28", " y_28"]]. values)
            head_rotation = compute_energy (data. loc [data[" success"] == 1, [" pose_Rx", " pose_Ry"," pose_Rz"]]. values, rotation = True)

            total_AU = FAU. sum (axis = 1)
            mean_data. append ([subject,  conversation] + [head_translation] + [head_rotation] + [total_AU. mean (axis = 0)] +  FAU. mean (axis = 0). tolist ())

            colnames = ["Subject", "Conversation"] +  ["Head_translation_energy", "Head_rotation_energy", "Total_AU"] + list (FAU. columns)

        else:
            if not os.path.exists (filename):
                mean_data. append ([subject,  conversation])
            else:
                data = pd. read_pickle (filename). iloc[:,1:]
                if data. shape [0] == 0:
                    mean_data. append ([subject,  conversation] + ["NaN" for i in data. shape[1]])
                else:
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
    subjects = ["sub-%02d"%i for i in range (1, 26)]
    for sub in ["sub-19"]:
        if sub in subjects:
            subjects. remove (sub)

    data = pd. DataFrame ()

    for subject in subjects:
        #print (subject)
        subj_data = generate_stats (subject,args. data_type)
        data = data. append (subj_data, ignore_index = True)
    #print (hh_data)
    data. to_csv ("stats_ts/%s.csv"%args. data_type, sep = ';', index = False)
