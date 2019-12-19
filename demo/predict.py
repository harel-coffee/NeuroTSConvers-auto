import os
import glob
import pandas as pd
import numpy as np
import joblib
import argparse
from ast import literal_eval
import sys
sys.path.append('src')
from prediction. tools import toSuppervisedData
from sklearn.metrics import recall_score, precision_score, f1_score, average_precision_score

#---------------------------------------------------#
def get_predictors (model_name, region, type):
    """
    model_name: name the prediction model
    region: brain area
    type: interaction type (h (human-human) or r (human-robot))
    """
    model_params = pd. read_csv ("results/prediction/%s_H%s.tsv"%(model_name, type. upper ()), sep = '\t', header = 0)
    predictors = model_params . loc [model_params["region"] == "%s"%region]["predictors_dict"]. iloc [0]

    return predictors

#---------------------------------------------------#
def get_predictors_dict (model_name, region, type):
    """
    model_name: name the prediction model
    region: brain area
    type: interaction type (h (human-human) or r (human-robot))
    """
    model_params = pd. read_csv ("results/prediction/%s_H%s.tsv"%(model_name, type. upper ()), sep = '\t', header = 0)
    predictors = model_params . loc [model_params["region"] == "%s"%region]["selected_predictors"]. iloc [0]

    return predictors

#---------------------------------------------------#
def speech_features (compute_features, out_dir):
	""" speech features from audio files """

	if compute_features:
		audio_path = "demo/speech"
		os. system ("python src/utils/SPPAS/sppas/bin/normalize.py -r src/utils/SPPAS/resources/vocab/eng.vocab -I "+ audio_path + " -l fra -e .TextGrid --quiet")
		os. system ("python src/utils/SPPAS/sppas/bin/phonetize.py  -I " + audio_path + " -l fra -e .TextGrid")
		os. system ("python src/utils/SPPAS/sppas/bin/alignment.py  -I " + audio_path + " -l fra -e .TextGrid --aligner basic")
		os. system ("python src/generate_ts/speech_features.py demo/speech/ %s/"%out_dir)
		os. system ("python src/generate_ts/speech_features.py demo/speech/ %s/ -l"%out_dir)

	if out_dir[-1] != '/':
		out_dir += '/'
	speech = pd. read_pickle (out_dir + "speech_features.pkl")
	speech_left = pd. read_pickle (out_dir + "speech_features_left.pkl")
	return speech_left, speech

#---------------------------------------------------#
def facial_features (compute_features, out_dir, openface_path):
    """ facial features  """

    video_path = glob.glob ("demo/video/*.avi")[0]

    if out_dir[-1] != '/':
    	out_dir += '/'

    if compute_features:
    	os. system ("python src/generate_ts/facial_action_units.py %s %s -op %s"%(video_path, out_dir, openface_path))

    video_features = glob.glob (out_dir + "*.pkl")[0]
    video_feats = pd. read_pickle (video_features)
    return video_feats

#---------------------------------------------------#
def eyetracking_features (compute_features, out_dir):
	""" eyetracking data """

	if out_dir[-1] != '/':
		out_dir += '/'

	if compute_features:
		video_path = glob.glob ("demo/video/*.avi")[0]
		openface_features = glob.glob ("demo/generated_time_series/video/"+ video_path[:-4]. split ('/')[-1] + "/*.csv")[0]
		gaze_coordinates_file = glob.glob ("demo/eyetracking/*.pkl")[0]

		os. system ("python src/generate_ts/eyetracking.py " + video_path + " %s"%out_dir + " -d -eye " + gaze_coordinates_file + " -faf " + openface_features)

	eyetracking_filename = glob.glob ("demo/generated_time_series/eyetracking/*.pkl")[0]
	eyetracking = pd. read_pickle (eyetracking_filename)

	return eyetracking

#---------------------------------------------------#
if __name__ == '__main__':
    parser = argparse. ArgumentParser ()
    parser. add_argument ('--regions','-rg', nargs = '+', type=int)
    parser. add_argument ('--type','-t', help = ' conversation type (human or robot)')
    parser. add_argument ('--lag','-p', default = 6, type=int)
    parser. add_argument ('--openface_path','-ofp', help = "path to openface")
    parser. add_argument ("--generate", "-g", help = "generate features from input signals", action="store_true")
    args = parser.parse_args()

    regions = args. regions

    # GET REGIONS NAMES FOR THEIR CODES
    brain_areas_desc = pd. read_csv ("brain_areas.tsv", sep = '\t', header = 0)
    regions = []

    for num_region in args. regions:
    	regions. append (brain_areas_desc . loc [brain_areas_desc ["Code"] == num_region, "Name"]. values [0])

    """ OUTPUT DIRECTORY FOT THE GENERATED TIME SERIES """
    for dirct in ["demo/generated_time_series", "demo/generated_time_series/speech", "demo/generated_time_series/video", "demo/generated_time_series/eyetracking"]:
    	if not os.path.exists (dirct):
    		os.makedirs (dirct)

    out_dir = "demo/generated_time_series/"

    """ GENERATE MULTIMODAL TIME SERIES FROM RAW SIGNALS """
    speech_left, speech = speech_features (args.generate, "demo/generated_time_series/speech")
    video = facial_features (args.generate, "demo/generated_time_series/video", args.openface_path)
    eyetracking = eyetracking_features (args.generate, "demo/generated_time_series/eyetracking")


    """ CONCATENATE MULTIMODAL DATA """
    all_data = np. concatenate ((speech_left. values, speech. values[:,1:], video. values[:,1:], eyetracking. values[:,1:]), axis = 1)

    columns = list (speech_left. columns) +  list (speech. columns [1:]) + list (video. columns [1:]) + list (eyetracking. columns [1:])

    # WRIGHT MULTIMODAL TIME SERIES TO CSV FILE
    pd. DataFrame (all_data, columns = columns). to_csv (out_dir + "all_features.csv", sep = ';', index = False)

    lagged_names = []
    for col in columns [1: ]:
    	lagged_names. extend ([col + "_t%d"%(p) for p in range (args. lag, 2, -1)])

    all_data = pd. DataFrame (toSuppervisedData (all_data, args. lag). data, columns = lagged_names)

    """ load the best models for each regions """
    if args. type == 'h':
    	conversation_type = 'HH'
    elif args. type == 'r':
    	conversation_type = 'HR'

    else:
    	print ("Error in arguments, use -h for help!")
    	exit (1)

    trained_models = glob. glob ("trained_models/*%s.pkl"%conversation_type)


    # dictionary of predictions: results
    preds = {}
    predictors_variables = {}
    for region in regions:
        print (region, "\n", 18 * '-')
        predictors_data = pd. DataFrame ()
        predictors_data. columns = []
        fname = ""
        for filename in trained_models:
            if region in  filename:
                fname = filename
                break

        model_name = fname. split ('/')[-1]. split ('_') [0]
        #print (model_name)
        model = joblib.load (fname)

        predictors = literal_eval (get_predictors_dict (model_name, region, args. type))
        print ("Predictors time series: ", predictors, "\n -------------")

        predictors_data = all_data. loc [:, predictors]

        #predictors_data = toSuppervisedData (predictors_data.values, 7, add_target = True). data

        pred = model. predict (predictors_data)
        #print (pred)
        preds [region] = [0 for i in range (args. lag)] + pred. tolist ()
        predictors_variables [region] = literal_eval (get_predictors (model_name, region, args. type))


    preds_var = pd.DataFrame ()
    for col in predictors_variables. keys ():
        preds_var[col] = [str (predictors_variables [col])]


    # time index : fMRI recording frequency
    step = 1.205
    index = [step]
    for i in range (1, args. lag + len (pred)):
    	index. append (index [i - 1] + step)


    pd. DataFrame (preds, index = index). to_csv ("demo/predictions.csv", sep = ';', index_label = ["Time (s)"])
    preds_var. to_csv ("demo/predictors.csv", sep = ';', index = False)
