"""
Parcellize the brain surface using .annot files.
"""
import numpy as np
from visbrain.gui import Brain
from visbrain.objects import (BrainObj, SceneObj, SourceObj, ConnectObj)
from visbrain.io import download_file

import nibabel as nib
import pandas as pd
import os
import cv2


def get_label_from_roi (brain_area, correspondance_tale):
    return correspondance_tale. loc [correspondance_tale["Name"] == brain_area, "Label"]. values [0]

"""================================================================="""
def create_brain_obj (annot_file_R, annot_file_L, areas):
    brain_obj = BrainObj(name = 'inflated', hemisphere='both', translucent=False, cbtxtsz = 10.) #, cblabel='Parcellates example', cbtxtsz=4.)
    left_areas = []
    right_areas = []

    for area in areas:
        if area[-1] == 'R':
            right_areas. append (area)
        elif area[-1] == 'L':
            left_areas. append (area)

    if len (left_areas) > 0:
        brain_obj.parcellize(annot_file_L, hemisphere='left',  select=left_areas)

    if len (right_areas) > 0:
        brain_obj.parcellize(annot_file_R, hemisphere='right',  select=right_areas)

    return brain_obj

'''def create_brain_obj (rfile, lfile, right_areas, left_areas):
    brain_obj = BrainObj(name = 'inflated', hemisphere='both', translucent=False, cbtxtsz = 10.) #, cblabel='Parcellates example', cbtxtsz=4.)
    brain_obj.parcellize(lfile, hemisphere='left',  select=left_areas)
    brain_obj.parcellize(rfile, hemisphere='right',  select=right_areas)
    return brain_obj'''

"""================================================================="""
def render_image (areas_labels, annot_file_R, annot_file_L):

    CBAR_STATE = dict(cbtxtsz=12, txtsz=10., width=.1, cbtxtsh=3., rect=(-.3, -2., 1., 4.))
    KW = dict(title_size=14., zoom=0.5)
    obj = BrainObj('inflated', hemisphere='both', translucent=False, _scale = 1.5)
    annot_data = obj. get_parcellates (annot_file_R)
    select = []

    for area in areas_labels:
        select. append (annot_data ["Labels"]. loc [annot_data["Labels"] == area]. values[0])

    sc = SceneObj (size=(1400, 1000))
    brain_objs = []

    # CREATE 4 BRAIN OBJECTS EACH WITH SPECOFOC ROTATION
    for rot in ["left", "right", 'side-fl', 'side-fr', 'front', 'back']:
        #brain_objs. append (create_brain_obj (r_file, l_file, select_right, select_left))
        brain_objs. append (create_brain_obj (annot_file_R, annot_file_L, select))

    # PLOT OBJECTS
    sc.add_to_subplot(brain_objs[0], row=0, col=0, rotate='right', title='Right', zoom = 3.5)
    sc.add_to_subplot(brain_objs[1], row=0, col=1, rotate='left', title='Left')
    sc.add_to_subplot(brain_objs[2], row=1, col=0, rotate='top', title='Top')
    sc.add_to_subplot(brain_objs[3], row=1, col=1, rotate='bottom', title='Bottom')
    sc.add_to_subplot(brain_objs[4], row=2, col=0, rotate='front', title='Front')
    sc.add_to_subplot(brain_objs[5], row=2, col=1, rotate='back', title='Back')

    #sc.preview()
    #exit (1)
    return sc. render ()

"""================================================================="""
def render_brain_image (r_areas_labels, l_areas_labels, r_file, l_file):

    CBAR_STATE = dict(cbtxtsz=12, txtsz=10., width=.1, cbtxtsh=3., rect=(-.3, -2., 1., 4.))
    KW = dict(title_size=14., zoom=3)

    obj = BrainObj('inflated', hemisphere='left', translucent=False) #, cblabel='Parcellates example', cbtxtsz=4.)
    #obj_right = BrainObj('inflated', hemisphere='right', translucent=False) #, cblabel='Parcellates example', cbtxtsz=4.)


    annot_data_l = obj. get_parcellates (l_file)
    annot_data_r = obj. get_parcellates (r_file)

    select_left = []
    select_right = []

    for l_area in l_areas_labels:
        select_left. append (annot_data_l ["Labels"]. loc[annot_data_l["Labels"] == l_area]. values)

    for r_area in r_areas_labels:
        select_right. append (annot_data_r ["Labels"]. loc[annot_data_r["Labels"] == r_area]. values)

    sc = SceneObj ()
    brain_objs = []

    # CREATE 4 BRAIN OBJECTS EACH WITH SPECOFOC ROTATION
    for rot in ["left", "right", 'side-fl', 'side-fr', 'front', 'back']:
        brain_objs. append (create_brain_obj (r_file, l_file, select_right, select_left))

    # PLOT OBJECTS
    sc.add_to_subplot(brain_objs[0], row=0, col=0, rotate='right', title='Right', zoom = 3)
    sc.add_to_subplot(brain_objs[1], row=0, col=1, rotate='left', title='Left', **KW)
    sc.add_to_subplot(brain_objs[2], row=1, col=0, rotate='side-fl', title='side-fl', **KW)
    sc.add_to_subplot(brain_objs[3], row=1, col=1, rotate='side-fr', title='side-fr', **KW)
    sc.add_to_subplot(brain_objs[4], row=2, col=0, rotate='front', title='front', **KW)
    sc.add_to_subplot(brain_objs[5], row=2, col=1, rotate='back', title='back', **KW)

    return sc. render ()

"""================================================================="""
if __name__ == '__main__':

    l_file = "BN_Atlas_freesurfer/fsaverage/label/lh.BN_Atlas.annot"
    r_file = "BN_Atlas_freesurfer/fsaverage/label/rh.BN_Atlas.annot"

    annot_l = nib.freesurfer.io.read_annot (l_file)
    annot_r = nib.freesurfer.io.read_annot (r_file)

    predictions = pd.read_csv ("demo/predictions.csv", sep = ';', header = 0)

    # ANNOTATION FILES TO CSV
    if not os.path.exists ("annot_visbrain/r_annot_as_tsv") or not os.path.exists ("annot_visbrain/l_annot_as_tsv"):
        df_l = pd.DataFrame ()
        df_r = pd.DataFrame ()

        df_l["Labels"] = annot_l [2]
        df_l["Colors"] = annot_l [1]. tolist ()

        df_r["Labels"] = annot_r [2]
        df_r["Colors"] = annot_r [1]. tolist ()

        df_l. to_csv ("annot_visbrain/l_annot_as_tsv", sep = "\t", index = False)
        df_r. to_csv ("annot_visbrain/r_annot_as_tsv", sep = "\t", index = False)

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    fps = 30
    out = cv2.VideoWriter ("demo/brain_activation.mp4", fourcc, fps, (1400, 1000))

    brain_areas_csv = pd. read_csv ("brain_areas.tsv", sep = '\t', header = 0)

    brain_areas = brain_areas_csv. loc[:, "Name"]. values
    coresp_table = brain_areas_csv[["Name", "Label"]]

    for i in range (predictions. shape [0]):
        # find activated areas
        activated_areas = []
        ligne = predictions. iloc [i, 1:]. values

        for j in range (len (ligne)):
            if ligne[j] == 1:
                activated_areas. append (get_label_from_roi (brain_areas[j], coresp_table))

        img = render_image (activated_areas, r_file, l_file)
        img = cv2.cvtColor (img, cv2.COLOR_BGR2RGB)
        for j in range (36):
            out.write(img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    #cv2.imshow ('image',cv2.cvtColor (img, cv2.COLOR_BGR2RGB))
    out.release()
    cv2.destroyAllWindows()
