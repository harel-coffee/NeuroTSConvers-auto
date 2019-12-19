import os
import glob
import pandas as pd
import numpy as np
import argparse

from sklearn import preprocessing

from prediction.tools import toSuppervisedData

#=============================================
# generate time series from transcriptions files
def process_transcriptions (subject, type = "speech_ts"):
    files = glob. glob ("time_series/%s/%s/*.pkl"%(subject, type))
    return sorted (files)

#===============================================
def get_unimodal_ts (subject, type, lag = 6):
    # do not consider the first columns (the time index)
    files = process_transcriptions (subject, type)

    # initilization
    HH_data = []
    HR_data = []

    # get colnames, and excluding first column, which is time
    colnames = list (pd. read_pickle (files[0]).iloc[:, 1:]. columns)
    lagged_columns = []
    if type not in ["physio_ts", "discretized_physio_ts"]:
        for item in colnames:
            lagged_columns. extend ([item + "_t%d"%(p) for p in range (lag, 2, -1)])
    else:
        lagged_columns = colnames

    # concatenate data of all conversations
    for filename in files:
        data = pd. read_pickle (filename).iloc[:, 1:]. values

        if type in ["physio_ts", "discretized_physio_ts"]:
            lagged_ts = toSuppervisedData (data, lag, add_target = True). targets
            # optimization to check later
            #lagged_ts = data. loc[lag:,:]
        else:
            # smoothing
            '''data_s = data. copy ()
            for i in range (1, len (data)):
                for j in range (data.shape[1]):
                    data_s[i, j] = 0.9 * data[i, j] + 0.1 * data[i - 1, j]'''
            '''data = np. diff (data, axis = 0)
            data = np. insert (data, 0, 0, axis = 0)'''


            # add two lines for bold signal synchronization
            data = np. append (data, data[-2:,:], axis = 0)

            lagged_ts = toSuppervisedData (data, lag, add_target = True). data

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
def get_behavior_ts_one_subject (subject, behaviours = ["speech_left_ts", "speech_ts", "eyetracking_ts"]):
    subj_behavioral = get_unimodal_ts (subject,  behaviours [0])
    for type in behaviours[1:]:
        unimodal_data = get_unimodal_ts (subject, type)
        subj_behavioral[0] = pd. concat ([subj_behavioral [0] , unimodal_data [0]], axis = 1)
        subj_behavioral[1] = pd. concat ([subj_behavioral [1], unimodal_data [1]], axis = 1)
    return subj_behavioral
#=============================================
if __name__ == '__main__':

    if not os.path.exists ("concat_time_series"):
        os.makedirs ("concat_time_series")

    # subjects to process
    subjects = ["sub-%02d"%i for i in range (1, 25)]
    for sub in ["sub-01", "sub-19", "sub-04",  "sub-12", "sub-14", "sub-16"]:
        if sub in subjects:
            subjects. remove (sub)

    lag = 6

    # 3 types of data to process
    behavioral_hh_data = pd. DataFrame ()
    behavioral_hr_data = pd. DataFrame ()

    bold_hh_data = pd. DataFrame ()
    bold_hr_data = pd. DataFrame ()

    discr_bold_hh_data = pd. DataFrame ()
    discr_bold_hr_data = pd. DataFrame ()

    # Concatenate data of the subjects
    for subject in subjects:
        print (subject)
        try:
            subj_behavioral = get_behavior_ts_one_subject (subject, ["speech_left_ts", "speech_ts", "eyetracking_ts", "facial_features_ts", "energy_ts"])
        except:
            print (subject)
            continue
        subj_bold = get_unimodal_ts (subject, "physio_ts")
        subj_discr_bold = get_unimodal_ts (subject, "discretized_physio_ts")

        # transform the data: construct lagged variables
        behavioral_hh_data = behavioral_hh_data. append (subj_behavioral[0], ignore_index=True, sort=False)
        behavioral_hr_data = behavioral_hr_data. append (subj_behavioral[1], ignore_index=True, sort=False)

        bold_hh_data = bold_hh_data. append (subj_bold [0], ignore_index=True, sort=False)
        bold_hr_data = bold_hr_data. append (subj_bold [1], ignore_index=True, sort=False)

        discr_bold_hh_data = discr_bold_hh_data. append (subj_discr_bold [0], ignore_index=True, sort=False)
        discr_bold_hr_data = discr_bold_hr_data. append (subj_discr_bold [1], ignore_index=True, sort=False)


    behavioral_hh_data. to_csv ("concat_time_series/behavioral_hh_data.csv", sep = ';', index = False)
    behavioral_hr_data. to_csv ("concat_time_series/behavioral_hr_data.csv", sep = ';', index = False)

    bold_hh_data. to_csv ("concat_time_series/bold_hh_data.csv", sep = ';', index = False)
    bold_hh_data. to_csv ("concat_time_series/bold_hr_data.csv", sep = ';', index = False)

    discr_bold_hh_data. to_csv ("concat_time_series/discr_bold_hh_data.csv", sep = ';', index = False)
    discr_bold_hr_data. to_csv ("concat_time_series/discr_bold_hr_data.csv", sep = ';', index = False)
