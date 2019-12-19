import pandas as pd
from glob import glob
import os
import argparse
import numpy as np
import pywt

import pylab as plt

from joblib import Parallel, delayed

#===================================================

def maddest(d, axis=None):
    """
    Mean Absolute Deviation
    """
    return np.mean(np.absolute(d - np.mean(d, axis)), axis)

#===================================================

def denoise_signal( x, wavelet='db4', level=1):
    """
    1. Adapted from waveletSmooth function found here:
    http://connor-johnson.com/2016/01/24/using-pywavelets-to-remove-high-frequency-noise/
    2. Threshold equation and using hard mode in threshold as mentioned
    in section '3.2 denoising based on optimized singular values' from paper by Tomas Vantuch:
    http://dspace.vsb.cz/bitstream/handle/10084/133114/VAN431_FEI_P1807_1801V001_2018.pdf
    """

    # Decompose to get the wavelet coefficients
    coeff = pywt.wavedec( x, wavelet, mode="per" )

    #print (coeff)

    # Calculate sigma for threshold as defined in http://dspace.vsb.cz/bitstream/handle/10084/133114/VAN431_FEI_P1807_1801V001_2018.pdf
    # As noted by @harshit92 MAD referred to in the paper is Mean Absolute Deviation not Median Absolute Deviation
    sigma = (1/0.6745) * maddest( coeff[-level] )

    # Calculte the univeral threshold
    uthresh = sigma * np.sqrt( 2*np.log( len( x ) ) )

    coeff[1:] = ( pywt.threshold( i, value=uthresh, mode='hard' ) for i in coeff[1:] )

    # Reconstruct the signal using the thresholded coefficients
    return pywt.waverec( coeff, wavelet, mode='per' )

#===========================================

def read_asci (filename):
    f = open(filename, 'r')
    line = f.readline()
    comment = []
    data = []

    while line:
        if line. startswith("**"):
            comment. append ([line])

        else: data. append (line. split ('\n')[0]. split ('\t'))

        line = f.readline()
    f.close()
    return data, comment

#===========================================

def find_event (data, events):
    mess = []
    begin = False

    for line in data:
        if line [0] in events:
            mess. append (line)

    return mess

#===========================================

def is_message (line):
    if line [0] in ["MSG"]: #, "EFIX", "SSACC", "SBLINK", "EBLINK", "ESACC", "SFIX"]:
        return True
    else: return False

#===========================================

def is_R (line):
    if line [0] in ['R']: #, "EFIX", "SSACC", "SBLINK", "EBLINK", "ESACC", "SFIX"]:
        return True
    else: return False

#===========================================

def is_event (line):

    events = ["MSG", "EFIX", "SFIX", "INPUT", "END", "SSACC", "ESACC", "SBLINK", "EBLINK"]
    result = False
    for event  in events:
        if event in line [0]:
            result = True
            break

    return result

#===========================================

def to_df (list):
    return pd. DataFrame (list)

#===========================================

# TODO : better organization
def find_start_end (data, start, end):
    start_end = []
    row = []
    no_end = False

    for i in range (len (data)):
        if start in data [i][0]:
            row. append (i)
            no_end = True
        if end in data [i][0]:
            if len (row) == 0:
                row. append (1)
            row. append (i)
            start_end. append (row)
            row = []
            no_end = False

    if no_end:
        row. append (len (data) - 1)
        start_end. append (row)

    return start_end

#=========================================
def find_saccades (data):

    sacc_indices = []
    saccades = find_start_end (data, "SSACC", "ESACC")
    blinks = find_start_end (data, "SBLINK", "EBLINK")

    # remove saccades that contain blinks
    '''for sacc in saccades:
        for blink in blinks:
            if blink [0] >= sacc[0] and blink [1] <= sacc[1]:
                saccades. remove (sacc)'''

    for sacc in saccades:
        sacc_indices. extend (list (range (sacc [0], sacc[1] + 1)))

    return sacc_indices

#==========================================
def find_blinks (data):

    blinks = find_start_end (data, "SBLINK", "EBLINK")
    saccades = find_start_end (data, "SSACC", "ESACC")

    sacc_contain_blinks = []

    for sacc in saccades:
        for blink in blinks:
            if blink [0] >= sacc[0] and blink [1] <= sacc[1]:
                sacc_contain_blinks. append (sacc)
                break

    blinks_indices = []
    blinks. extend (sacc_contain_blinks)

    for blink in blinks:
        blinks_indices. extend (list (range (blink [0], blink[1] + 1)))

    return list (set (blinks_indices))

#===========================================
def find_conv (data, message):
    convers = []
    i = 0
    # Saving conversations points
    for line in data:
        if is_message (line) and message in line [2]:
            convers. append ([line, i, 0, 0])
        i += 1

    for i in range (len (convers) - 1):
        convers[i][2] = convers [i + 1][1] - 1

    convers[-1][2] = len (data) - 1

    for i in range (len (convers)):
        convers[i][3] = convers[i][2] - convers[i][1]

    return convers


#===========================================
def find_convers (data):
    """
        Find the data associated to the 6 conversations, and put
        them into a list of lists. Then removing blinks and saccades.
    """

    convers = find_conv (data, "CONV")
    saccades = [[] for i in range (len (convers))]

    conversations = [data [conv [1] : conv [2]] for conv in convers]

    for i in range (len (conversations)):
        conv_cleaned = []
        blinks = find_blinks (conversations [i])

        sacc_indices = find_saccades (conversations [i])

        # indices of saccdes
        all = blinks # + sacc_indices

        for j in range (len (conversations [i])):
            if j not in all and conversations [i][j][0] and not is_event (conversations [i][j]):
                conv_cleaned. append (conversations [i][j])

                if j in sacc_indices:
                    saccades [i]. append ([1])

                else:
                    saccades [i]. append ([0])


        conversations [i] = conv_cleaned. copy ()

    return conversations, saccades

# =============================================
def process_filename (filename):
    """
        Read an asci file (1 block), detect correct coordinates, detect saccdes and blinks, and store
        the results in a dataframe, then in a pickle file
    """
    print (filename)
    short_file_name = filename. split ('_')[0:2]
    subject = short_file_name [0]. split ('/')[-1]

    testBlock = '-'. join (short_file_name [1]. split ('-')[1:3])

    """ get data from asci files """
    data, comment = read_asci (filename)
    convers, saccades = find_convers (data)

    """ Extract conversations from concatenated data,
        and initialize the begining time of each conversations at 0 """

    for i in range (0, len (convers)):
        if len (convers [i]) == 0:
            continue

        begin = int (convers[i][0][0])
        for j in range (len (convers[i])):
            convers[i][j][0] = (int (convers[i][j][0]) - begin) / 1000
            for k in range (1,4):
                convers[i][j][k] = float (convers[i][j][k])

        if i % 2 == 0:
            conv = "CONV1_%03d"%(i+1)
        else:
            conv = "CONV2_%03d"%(i+1)


        eyetracking_data = np. concatenate ( (np.array (convers [i]) [:,0:3], saccades [i]), axis = 1)
        eyetracking_data = pd. DataFrame (eyetracking_data, columns = ["Time (s)", "x", "y", "saccades"])
        eyetracking_data. to_pickle ("time_series/%s/gaze_coordinates_ts/%s_%s.pkl"%(subject, testBlock, conv))

#==============================================

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--process", "-p", help = "generate asci files from edf files.", action = "store_true")
    args = parser.parse_args()

    if args. process:
        edf_files = glob ("data/edt/*.edf*")

        for filename in edf_files:
            os. system ("data/./EDF_Access_API/Example/edf2asc -failsafe -t -miss NaN -y -v %s"%filename)

    asci_files = glob ("data/edt/*.asc*")
    asci_files. sort ()

    subjects = []
    for i in range(1, 26):
        if i < 10:
            subjects.append("sub-0%s" % str(i))
        else:
            subjects.append("sub-%s" % str(i))

    for subject in subjects:
        if not os.path.exists("time_series/%s/gaze_coordinates_ts" % subject):
            os.makedirs("time_series/%s/gaze_coordinates_ts" % subject)

    Parallel (n_jobs=4) (delayed (process_filename) (filename) for filename in asci_files)
