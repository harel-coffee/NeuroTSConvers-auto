"""
    Author: Youssef Hmamouche
    Year: 2019
    Generate time series from raw signals: video audio, and eyetracking coordiantes.
"""

import os
import glob
import pandas as pd
import numpy as np
import argparse
from ast import literal_eval
import sys


#---------------------------------------------------#
def get_predictors (model_name, region, type, path):
    """
    model_name: name the prediction model
    region: brain area
    type: interaction type (h (human-human) or r (human-robot))
    """
    model_params = pd. read_csv ("%s/results/prediction/%s_H%s.tsv"%(path, model_name, type. upper ()), sep = '\t', header = 0)
    predictors = model_params . loc [model_params["region"] == "%s"%region]["predictors_dict"]. iloc [0]

    return predictors

#---------------------------------------------------#
def get_predictors_dict (model_name, region, type, path):
    """
    model_name: name the prediction model
    region: brain area
    type: interaction type (h (human-human) or r (human-robot))
    """
    model_params = pd. read_csv ("%s/results/prediction/%s_H%s.tsv"%(path, model_name, type. upper ()), sep = '\t', header = 0)
    predictors = model_params . loc [model_params["region"] == "%s"%region]["selected_predictors"]. iloc [0]

    return predictors

#---------------------------------------------------#
def speech_features (pred_path, out_dir, language):
	"""
	generate speech features from audio files
	pred_path: path of the prediction module
	compute_features: logical, for computing the features or not (if they alreadu exists)
	out_dir: output directory
	"""

	audio_input = "%s/Inputs/speech"%out_dir
	audio_output = "%s/Outputs/generated_time_series/speech"%out_dir

	if language == "fr":
		lang = "fra"
	elif language == "eng":
		lang = "eng"

	os. system ("python %s/src/utils/SPPAS/sppas/bin/normalize.py -r %s/src/utils/SPPAS/resources/vocab/eng.vocab -I %s  -l %s -e .TextGrid --quiet"%(pred_path, pred_path, audio_input, lang))
	os. system ("python %s/src/utils/SPPAS/sppas/bin/phonetize.py  -I %s -l %s -e .TextGrid"%(pred_path, audio_input, lang))
	os. system ("python %s/src/utils/SPPAS/sppas/bin/alignment.py  -I %s -l %s -e .TextGrid --aligner basic"%(pred_path, audio_input, lang))

	out = os. system ("python %s/src/generate_ts/speech_features.py %s %s/ -lg %s -n"%(pred_path, audio_input, audio_output, language))
	out = os. system ("python %s/src/generate_ts/speech_features.py %s %s/ -l -lg %s -n"%(pred_path, audio_input, audio_output, language))

	if out_dir[-1] != '/':
		out_dir += '/'
	speech = pd. read_pickle ("%s/speech_features.pkl"%audio_output)
	speech_left = pd. read_pickle ("%s/speech_features_left.pkl"%audio_output)
	return speech_left, speech

#---------------------------------------------------#
def openface_features (pred_path, out_dir, openface_path):
    """ facial features  """

    #video_input = "%s/Inputs/video"%out_dir
    video_output = "%s/Outputs/generated_time_series/video/"%out_dir
    video_path = glob.glob ("%s/Inputs/video/*.avi"%out_dir)

    facial_output = "%s/Outputs/generated_time_series/video/"%out_dir

    if len (video_path) == 0:
    	print ("Error: there no input video!")
    	exit (1)
    else:
    	video_path = video_path[0]

    video_name = video_path. split ('/')[-1]. split ('.')[0]

    if out_dir[-1] != '/':
    	out_dir += '/'

    os. system ("python %s/src/generate_ts/facial_action_units.py %s %s -op %s"%(pred_path, video_path, video_output, openface_path))
    openface_features = glob.glob (video_output + "/" + video_path[:-4]. split ('/')[-1] + "/*.csv")[0]
    #os. system ("python %s/src/generate_ts/facial_action_units.py %s %s -faf %s -d"%(pred_path, video_path, facial_output, openface_features))

    video_features = glob.glob (video_output + "/*.pkl")
    #video_feats = pd. read_pickle (video_features[0])
    facial_feats = pd. read_pickle (video_features[0])
    return facial_feats

#---------------------------------------------------#
def extra_features (pred_path, out_dir, type = "eyetracking"):
    """ eyetracking data """

    video_output = "%s/Outputs/generated_time_series/video"%out_dir
    eyetracking_output = "%s/Outputs/generated_time_series/%s"%(out_dir, type)
    video_path = glob.glob ("%s/Inputs/video/*.avi"%out_dir)
    if len (video_path) == 0:
    	print ("Error: there is no input video!")
    	exit (1)
    else:
    	video_path = video_path [0]

    #print ("Processing eyetracking data")
    if out_dir[-1] != '/':
    	out_dir += '/'

    openface_features = glob.glob (video_output + "/" + video_path[:-4]. split ('/')[-1] + "/*.csv")[0]

    if type == "eyetracking":
        gaze_coordinates_file = glob.glob ("%s/Inputs/eyetracking/*.pkl"%out_dir)[0]
        out = os. system ("python %s/src/generate_ts/eyetracking.py %s %s -d -eye %s -faf %s -sv"%(pred_path, video_path, eyetracking_output, gaze_coordinates_file, openface_features))

    elif type == "emotions":
        out = os. system ("python %s/src/generate_ts/facial_emotions.py -d %s %s"%(pred_path, video_path, eyetracking_output))

    elif type == "facial":
        out = os. system ("python %s/src/generate_ts/facial_features.py %s %s -d -faf %s"%(pred_path, video_path, eyetracking_output, openface_features))

    elif type == "smiles":
        out = os. system ("python %s/src/generate_ts/dlib_smiles.py -d %s %s"%(pred_path, video_path, eyetracking_output))


    eyetracking_filename = glob.glob ("%s/*.pkl"%eyetracking_output)[0]
    eyetracking = pd. read_pickle (eyetracking_filename)

    return eyetracking

#---------------------------------------------------#
if __name__ == '__main__':
    parser = argparse. ArgumentParser ()
    requiredNamed = parser.add_argument_group('Required arguments')
    requiredNamed. add_argument ('--regions','-rg', help = "Numbers of brain areas to predict (see brain_areas.tsv)", nargs = '+', type=int)
    requiredNamed.add_argument("--language", "-lg", default = "fr", choices = ["fr", "eng"], help="Language.")
    requiredNamed. add_argument ('--openface_path','-ofp', help = "path of Openface", required=True)
    requiredNamed. add_argument ('--pred_module_path','-pmp', help = "path of the prediction module", required=True)
    requiredNamed. add_argument ('--input_dir','-in', help = "path of input directory", required=True)
    args = parser.parse_args()


    if args. pred_module_path [-1] == '/':
    	args. pred_module_path = args. pred_module_path [:-1]

    out_dir =  "%s/Outputs/generated_time_series/"%args.input_dir

    # GET REGIONS NAMES FOR THEIR CODES
    brain_areas_desc = pd. read_csv ("brain_areas.tsv", sep = '\t', header = 0)

    regions = []
    for num_region in args. regions:
    	regions. append (brain_areas_desc . loc [brain_areas_desc ["Code"] == num_region, "Name"]. values [0])

    """ CREATE OUTPUT DIRECTORIES OF THE GENERATED TIME SERIES """
    for dirct in ["%s/Outputs"%args.input_dir, out_dir, "%s/Outputs/generated_time_series/speech"%args.input_dir, \
    			 "%s/Outputs/generated_time_series/video"%args.input_dir, \
                 "%s/Outputs/generated_time_series/eyetracking"%args.input_dir,\
                 "%s/Outputs/generated_time_series/emotions"%args.input_dir,\
                 "%s/Outputs/generated_time_series/facial"%args.input_dir,\
                 "%s/Outputs/generated_time_series/smiles"%args.input_dir\
                 ]:
    	if not os.path.exists (dirct):
    		os.makedirs (dirct)

    """ GENERATE MULTIMODAL TIME SERIES FROM RAW SIGNALS """
    speech_left, speech = speech_features (args.pred_module_path, args.input_dir, args. language)
    video = openface_features (args.pred_module_path, args.input_dir, args.openface_path)
    print ("Processing openface features, ... Done.")

    # Extract other facial features: emotions, facial based features ...
    eyetracking = extra_features (args.pred_module_path, args.input_dir, "eyetracking")
    print ("Processing eyetracking, ... Done.")
    emotions = extra_features (args.pred_module_path, args.input_dir, "emotions")
    print ("Processing emotions, ... Done.")
    facial = extra_features (args.pred_module_path, args.input_dir, "facial")
    print ("Processing facial features, ... Done.")
    smiles = extra_features (args.pred_module_path, args.input_dir, "smiles")
    print ("Processing smiles, ... Done.")

    """ CONCATENATE AND SAVE MULTIMODAL DATA """
    all_data = np. concatenate ((speech_left. values,\
                                speech. values[:,1:],\
                                eyetracking. values[:,1:],\
                                emotions. values[:,1:],\
                                facial. values[:,1:],\
                                smiles. values[:,1:],\
                               ), axis = 1)
    columns = list (speech_left. columns) +  list (speech. columns [1:]) + list (eyetracking. columns [1:]) + list (emotions. columns [1:]) + list (facial. columns [1:]) + list (smiles. columns [1:])
    pd. DataFrame (all_data, columns = columns). to_csv (out_dir + "features.csv", sep = ';', index = False)
