# coding: utf8

import numpy as np
import pandas as pd

import glob, os,sys,inspect, argparse

from python_speech_features import mfcc
from python_speech_features import logfbank
import scipy.io.wavfile as wav

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
maindir = os.path.dirname(parentdir)
sys.path.insert(3,maindir)

from src.resampling import resample_ts

#================================================================
if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument("data_dir", help="the path of the file to process.")
	parser.add_argument("out_dir", help="the path where to store the results.")
	parser.add_argument("--language", "-lg", default = "fr", choices = ["fr", "eng"], help="Language.")
	parser.add_argument("--left", "-l", help="Process participant speech.", action="store_true")
	parser.add_argument("--outname", "-n", help="Rename output file names by default.", action="store_true")

	args = parser.parse_args()

	data_dir = args. data_dir
	out_dir = args. out_dir

	if args. out_dir[-1] != '/':
		args. out_dir += '/'

	# create output dir file if does not exist
	if not os.path.exists (args. out_dir):
		os.makedirs (args. out_dir)

	filename = args. data_dir.split('/')[-1]. split ('.')[0]
	conversation_name = data_dir.split ('/')[-1]

	if conversation_name == "" or args.outname:
		if args. left:
			conversation_name = "mfcc_features_left"
		else:
			conversation_name = "mfcc_features"


	print ("---------", conversation_name, "---------")
	# Index variable
	physio_index = [0.6025]
	for i in range (1, 50):
		physio_index. append (1.205 + physio_index [i - 1])

	# exit if conversation already processed
	output_filename_pkl = out_dir +  conversation_name + ".pkl"
	if os.path.isfile (output_filename_pkl):
		print ("Conversation already processed")
		exit (1)

	# Read audio, and transcription file
	for file in glob.glob(data_dir + "/*.wav"):
		if args.left and "left" in file:
		    audio_file = file
		    break
		elif "right" in file:
		    audio_file = file
		    break

	(rate,sig) = wav.read(audio_file)

	mfcc_feat = mfcc(sig,rate)
	fbank_feat = logfbank(sig,rate)



	index = [0.01]
	for i in range (1, len (fbank_feat)):
	    index. append (index [i - 1] + 0.01)

	fbank_feat = np. insert (fbank_feat, 0, index, axis = 1)
	fbank_feat = np. insert (fbank_feat, 0, [0 for i in range (fbank_feat. shape[1])], axis = 0)

	new_index = [0.6025]
	for i in range (1,50):
	    new_index. append (new_index [i - 1] + 1.205)

	resampled_fbank_feat = resample_ts (fbank_feat, new_index, 'mean')

	if args. left:
		cols = ["Time (s)"] + ["mfcc_%d_left"%i for i in range (1, resampled_fbank_feat. shape[1])]
	else:
		cols = ["Time (s)"] + ["mfcc_%d"%i for i in range (1, resampled_fbank_feat. shape[1])]
	# save dataframe in pickle file

	pd.DataFrame (resampled_fbank_feat, columns = cols). to_pickle (output_filename_pkl)
