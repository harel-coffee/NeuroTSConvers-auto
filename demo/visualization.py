'''
    Author: Youssef Hmamouche
    Year: 2019
    Generate video of brain activity predictions
    This script uses a csv file of predictions, and annotation files for brain parcellation
'''

import numpy as np
from visbrain.gui import Brain
from visbrain.objects import (BrainObj, SceneObj, SourceObj, ConnectObj)
from visbrain.io import download_file

import nibabel as nib
import pandas as pd
import os
import cv2
import argparse

#=================================================================
def get_label_from_roi (brain_area, correspondance_tale):
    return correspondance_tale. loc [correspondance_tale["ShortName"] == brain_area, "Label"]. values [0]

#=================================================================
def create_brain_obj (annot_file_R, annot_file_L, areas):
    brain_obj = BrainObj(name = 'inflated', hemisphere='both', translucent=False, cbtxtsz = 10.) #, cblabel='Parcellates example', cbtxtsz=4.)
    left_areas = []
    right_areas = []


    for area in areas:
        if area[-1] in ['R', 'r']:
            right_areas. append (area)
        elif area[-1] in ['L', 'l']:
            left_areas. append (area)
        else:
            raise ("Error, label area does indicate if it is in left or rights hemisphere.")

    if len (left_areas) > 0:
        brain_obj.parcellize(annot_file_L, hemisphere='left',  select=left_areas)

    if len (right_areas) > 0:
        brain_obj.parcellize(annot_file_R, hemisphere='right',  select=right_areas)

    return brain_obj


#=================================================================
def render_image (areas_labels, annot_file_R, annot_file_L):

    CBAR_STATE = dict(cbtxtsz=12, txtsz=10., width=.1, cbtxtsh=3., rect=(-.3, -2., 1., 4.))
    KW = dict(title_size=14., zoom=0.5)
    obj = BrainObj('inflated', hemisphere='both', translucent=False, _scale = 1.5)
    annot_data_L = obj. get_parcellates (annot_file_L)
    annot_data_R = obj. get_parcellates (annot_file_R)
    select = []

    sc = SceneObj (size=(1400, 1000))
    brain_objs = []

    # CREATE 4 BRAIN OBJECTS EACH WITH SPECOFOC ROTATION
    for rot in ["left", "right", 'side-fl', 'side-fr', 'front', 'back']:
        brain_objs. append (create_brain_obj (annot_file_R, annot_file_L, areas_labels))

    # PLOT OBJECTS
    sc.add_to_subplot(brain_objs[0], row=0, col=0, rotate='right', title='Right')
    sc.add_to_subplot(brain_objs[1], row=0, col=1, rotate='left', title='Left')
    sc.add_to_subplot(brain_objs[2], row=1, col=0, rotate='top', title='Top')
    sc.add_to_subplot(brain_objs[3], row=1, col=1, rotate='bottom', title='Bottom')
    sc.add_to_subplot(brain_objs[4], row=2, col=0, rotate='side-fl', title='Front-left')
    sc.add_to_subplot(brain_objs[5], row=2, col=1, rotate='side-fr', title='Front-right')

    #sc.preview()
    #exit (1)
    return sc. render ()

#=================================================================
if __name__ == '__main__':
    parser = argparse. ArgumentParser ()
    requiredNamed = parser.add_argument_group('Required arguments')
    requiredNamed. add_argument ('--input_dir','-in', help = "path of input directory")
    args = parser.parse_args()

    brain_areas_csv = pd. read_csv ("brain_areas.tsv", sep = '\t', header = 0)
    brain_areas = brain_areas_csv. loc[:, "ShortName"]. values
    coresp_table = brain_areas_csv[["ShortName", "Label"]]

    l_file = "parcellation/lh.BN_Atlas.annot"
    r_file = "parcellation/rh.BN_Atlas.annot"


    predictions = pd.read_csv ("%s/Outputs/predictions.csv"%args.input_dir, sep = ';', header = 0)
    predicted_areas = list (predictions. columns[1:])


    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    fps = 30
    out = cv2.VideoWriter ("%s/Outputs/brain_activation.mp4"%args.input_dir, fourcc, fps, (1400, 1000))

    for i in range (predictions. shape [0]):
        # find activated areas
        activated_areas = []
        regions = []
        #l_activated_areas = []
        ligne = predictions. iloc [i, 1:]. values

        for j in range (len (ligne)):
            if ligne[j] == 1:
                activated_areas. append (get_label_from_roi (predicted_areas[j], coresp_table))
                regions. append (predicted_areas[j])


        img = render_image (activated_areas, r_file, l_file)
        #img = render_brain_image (r_activated_areas, l_activated_areas, r_file, l_file)
        img = cv2.cvtColor (img, cv2.COLOR_BGR2RGB)
        for j in range (36):
            out.write(img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    out.release()
    cv2.destroyAllWindows()
