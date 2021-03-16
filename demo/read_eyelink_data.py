import pandas as pd
from glob import glob
import os
import argparse
import numpy as np
import pywt

import pylab as plt

from joblib import Parallel, delayed


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
    sacc_contain_blinks_indices = []
    saccades = find_start_end (data, "SSACC", "ESACC")
    blinks = find_start_end (data, "SBLINK", "EBLINK")

    # remove saccades that contain blinks
    for sacc in saccades:
        for blink in blinks:
            if blink [0] >= sacc[0] and blink [1] <= sacc[1]:
                #saccades. remove (sacc)
                sacc_contain_blinks_indices. extend (list (range (sacc [0], sacc[1] + 1)))
                break

    for sacc in saccades:
        sacc_indices. extend (list (range (sacc [0], sacc[1] + 1)))

    return sacc_indices, sacc_contain_blinks_indices

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

        if len (line) >= 3:
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
        them into a list of lists. Then removing blinks.
        return x,y and saccades
    """

    convers = find_conv (data, "CONV")

    saccades = []
    blinks = []

    conversations = data [convers[0][1] : convers[0][2]]
    conv_cleaned = []

    blinks_indices = find_blinks (conversations)
    sacc_indices, _ = find_saccades (conversations)

    for j in range (len (conversations)):

        if not is_event (conversations[j]):
            #if j not in blinks_indices :
            conv_cleaned. append (conversations[j])

            if j in sacc_indices:
                saccades. append ([1])

            else:
                saccades. append ([0])

            if j in blinks_indices:
                blinks. append ([1])
            else:
                blinks. append ([0])

    return conv_cleaned, saccades, blinks

# =============================================
def process_filename (filename, out_dir):
    """
        Read an asci file (1 block), detect correct coordinates, detect saccdes and blinks, and store
        the results in a dataframe, then in a pickle file
    """

    """ get data from asci files """
    data, comment = read_asci (filename)
    #convers, saccades = find_convers (data)
    convers, saccades, blinks = find_convers (data)

    # get display coordinates
    for line in data:
        if line[0] == "MSG" and line[2]. split (" ")[0] == "DISPLAY_COORDS":
            DISPLAY_COORDS =  line[2]. split (" ")[3:]
            break

    """ Extract conversations from concatenated data,
        and initialize the begining time of each conversations at 0 """

    begin = int (convers[0][0])
    for j in range (len (convers)):
        convers[j][0] = (int (convers[j][0]) - begin) / 1000
        for k in range (1,3):
            convers[j][k] = float (convers[j][k])

    eyetracking_data = np. concatenate ( (np.array (convers) [:,0:4], saccades, blinks), axis = 1)
    eyetracking_data = pd. DataFrame (eyetracking_data, columns = ["Time (s)", "x", "y", "pupil_area", "saccades", "blinks"])
    eyetracking_data. _metadata = ["display_coords"]
    eyetracking_data. display_coords = DISPLAY_COORDS

    eyetracking_data. to_pickle ("%s/gaze_coordinates.pkl"%(out_dir))

#==============================================

if __name__ == '__main__':
    """
        reading eyelink output text file, and extract gaze coordinates and saccades into a pkl file.
    """
    parser = argparse. ArgumentParser ()
    parser. add_argument ('input_file', help = "input txt eyelink file")
    parser. add_argument ('out_dir', help = "output txt eyelink file")
    args = parser.parse_args ()

    process_filename (args. input_file, args. out_dir)
