import os
import glob
import pandas as pd
import numpy as np
import argparse
from sklearn.ensemble import IsolationForest

from sklearn import preprocessing

from prediction.tools import toSuppervisedData, get_lagged_colnames


#=============================================
# generate time series from transcriptions files
def process_transcriptions (subject, type = "speech_ts"):
    files = glob. glob ("time_series/%s/%s/*.pkl"%(subject, type))
    return sorted (files)

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

    if type == "facial_features_ts":
        for i in range (len (files)):
            files [i] = files [i]. split ('.')[0]
            conv_name = files [i]. split ('/')[-1]
            files [i] += '/' + conv_name + ".csv"

    return sorted (files)

#===============================================
def reorganize_data (data_, lag = 6, step = 1):
    """
        step: ratio between frequencies of two time index (1.205 s)
        lag: lag
    """

    out_data = []

    # first point 0.6s correspont to the first BOLD acquisition
    lag0 = 1
    real_lag = lag - 2

    '''for i in range (0, lag, step):
        row = []
        for j in range (data_. shape [1]):
            row = row + [0 for x in range (real_lag)] #+ [np. sum (data_ [i - lag : i - lag + 3, j])]
        out_data. append (row)'''

    for i in range (lag, len (data_), step):
        row = []
        for j in range (data_. shape [1]):
            row = row + list (data_ [i - lag : i -  lag + real_lag, j]. flatten ()) #+ [np. sum (data_ [i - lag : i - lag + 3, j])]
        out_data. append (row)

    return np. array (out_data)

#===============================================
def get_unimodal_ts (subject, type, lag, bold = False):
    # do not consider the first columns (the time index)
    files = list_files (subject, type)
    # initilization
    HH_data = []
    HR_data = []

    # get colnames, and excluding first column, which is time
    colnames = list (pd. read_pickle (files[0]).iloc[:, 1:]. columns)
    lagged_columns = []
    #if type not in ["physio_ts", "discretized_physio_ts"]:
    if not bold:
        lagged_columns = get_lagged_colnames (colnames, lag, dict = False)

    else:
        lagged_columns = colnames

    # concatenate data of all conversations
    for filename in files:
        # handle exceptions: lack of some files for some subjects
        if subject == "sub-01" and filename.split ('/')[-1] == "convers-TestBlocks3_CONV2_002.pkl":
            continue

        # Extract behavioral variables without the first column, which is time.
        data = pd. read_pickle (filename).iloc[:, 1:]. values

        # neuro_physio data must contain 52 observations for each conversation
        if type in ["physio_ts", "discretized_physio_ts"]:
            if data. shape[0] < 52:
                while (data. shape[0] < 52):
                    data = np. append (data, data[-1:,:], axis = 0)

        if bold:
            lagged_ts = data [lag:,]
            #lagged_ts = data

        else:
            # add two lines for bold signal synchronization
            if type not in ["physio_ts", "discretized_physio_ts"]:
                data = np. append (data, data[-2:,:], axis = 0)

            lagged_ts = reorganize_data (data, lag = lag)

        #lagged_ts = lagged_ts[5:-5,:]

        if "CONV1" in filename:
            if len (HH_data) == 0:
                HH_data = lagged_ts
            else:
                HH_data = np. concatenate ((HH_data, lagged_ts), axis = 0)

        if "CONV2" in filename:
            if len (HR_data) == 0:
                HR_data = lagged_ts
            else:
                HR_data = np. concatenate ((HR_data, lagged_ts), axis = 0)

    return [pd.DataFrame (HH_data, columns = lagged_columns), pd.DataFrame (HR_data, columns = lagged_columns)]

#=============================================
def get_behavior_ts_one_subject (subject, behaviours, lag):
    subj_behavioral = get_unimodal_ts (subject,  behaviours [0], lag)
    for type in behaviours[1:]:
        unimodal_data = get_unimodal_ts (subject, type, lag)
        subj_behavioral[0] = pd. concat ([subj_behavioral [0] , unimodal_data [0]], axis = 1)
        subj_behavioral[1] = pd. concat ([subj_behavioral [1], unimodal_data [1]], axis = 1)
    # return two outputs: for human-human and human-robot
    return subj_behavioral
#=============================================s'imprégner
if __name__ == '__main__':

    parser = argparse. ArgumentParser ()
    parser. add_argument ("--lag", "-p", help = "lag parameter", type = int, default = 6)
    args = parser.parse_args()

    if not os.path.exists ("concat_time_series"):
        os.makedirs ("concat_time_series")

    # subjects to process
    subject_exceptions = ["sub-01", "sub-14", "sub-19", "sub-12"]
    subjects = ["sub-%02d"%i for i in range (1, 26)]
    for sub in subject_exceptions:
        if sub in subjects:
            subjects. remove (sub)

    # 3 types of data to process
    behavioral_hh_data = pd. DataFrame ()
    behavioral_hr_data = pd. DataFrame ()

    bold_hh_data = pd. DataFrame ()
    bold_hr_data = pd. DataFrame ()

    disc_bold_hh_data = pd. DataFrame ()
    disc_bold_hr_data = pd. DataFrame ()

    subjs_info = pd.read_csv ("data/participants_info.txt", sep = '\t')

    # Concatenate data of the subjects
    for subject in subjects:

        print (subject, "\n", 18*'-')

        subj_behavioral = get_behavior_ts_one_subject (subject, ["speech_left_ts", "speech_ts", "eyetracking_ts", "dlib_smiles_ts", "energy_ts", "smiles_ts"], args.lag)
        subj_bold = get_unimodal_ts (subject, "physio_ts", args.lag, True)
        subj_disc_bold = get_unimodal_ts (subject, "discretized_physio_ts", args.lag, True)

        behavioral_hh_data = behavioral_hh_data. append (subj_behavioral[0], ignore_index=True, sort=False)
        behavioral_hr_data = behavioral_hr_data. append (subj_behavioral[1], ignore_index=True, sort=False)

        bold_hh_data = bold_hh_data. append (subj_bold [0], ignore_index=True, sort=False)
        bold_hr_data = bold_hr_data. append (subj_bold [1], ignore_index=True, sort=False)


        disc_bold_hh_data = disc_bold_hh_data. append (subj_disc_bold [0], ignore_index=True, sort=False)
        disc_bold_hr_data = disc_bold_hr_data. append (subj_disc_bold [1], ignore_index=True, sort=False)


    # Replace nan and very small values with 0
    behavioral_hh_data. fillna (0, inplace = True)
    behavioral_hh_data. fillna (0, inplace = True)
    behavioral_hh_data [behavioral_hh_data < 0.00001] = 0
    behavioral_hh_data [behavioral_hh_data < 0.00001] = 0

    behavioral_hr_data. fillna (0, inplace = True)
    behavioral_hr_data. fillna (0, inplace = True)
    behavioral_hr_data [behavioral_hr_data < 0.00001] = 0
    behavioral_hr_data [behavioral_hr_data < 0.00001] = 0


    # Store data in csv files
    behavioral_hh_data. to_csv ("concat_time_series/behavioral_hh_data.csv", sep = ';', index = False)
    behavioral_hr_data. to_csv ("concat_time_series/behavioral_hr_data.csv", sep = ';', index = False)

    bold_hh_data. to_csv ("concat_time_series/bold_hh_data.csv", sep = ';', index = False)
    bold_hr_data. to_csv ("concat_time_series/bold_hr_data.csv", sep = ';', index = False)

    disc_bold_hh_data. to_csv ("concat_time_series/discr_bold_hh_data.csv", sep = ';', index = False)
    disc_bold_hr_data. to_csv ("concat_time_series/discr_bold_hr_data.csv", sep = ';', index = False)


    # Store data in pickle files
    behavioral_hh_data. to_pickle ("concat_time_series/behavioral_hh_data.pkl")
    behavioral_hr_data. to_pickle ("concat_time_series/behavioral_hr_data.pkl")

    bold_hh_data. to_pickle ("concat_time_series/bold_hh_data.pkl")
    bold_hr_data. to_pickle ("concat_time_series/bold_hr_data.pkl")

    disc_bold_hh_data. to_pickle ("concat_time_series/discr_bold_hh_data.pkl")
    disc_bold_hr_data. to_pickle ("concat_time_series/discr_bold_hr_data.pkl")

    print (behavioral_hh_data. shape)
    print (bold_hh_data. shape)
