# coding: utf8
import numpy as np
import pandas as pd

import spacy as sp

import argparse
import sys
import glob
import os

#================================================================
if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument("data_dir", help="the path of the file to process.")
	parser.add_argument("out_dir", help="the path where to store the results.")
	parser.add_argument("--left", "-l", help="Process participant speech.", action="store_true")

	args = parser.parse_args()

	data_dir = args. data_dir
	out_dir = args. out_dir

	if out_dir == 'None':
		usage ()
		exit ()

	if args. out_dir[-1] != '/':
		args. out_dir += '/'

	# create output dir file if does not exist
	if not os.path.exists (args. out_dir):
		os.makedirs (args. out_dir)

	filename = args. data_dir.split('/')[-1]. split ('.')[0]

	conversation_name = data_dir.split ('/')[-1]

	if conversation_name == "":
		if args. left:
			conversation_name = "speech_features_left"
		else:
			conversation_name = "speech_features"

	print ("-----------------", conversation_name)

	output_filename_png = out_dir +  conversation_name + ".png"
	output_filename_pkl = out_dir +  conversation_name + ".pkl"


	# Index variable
	physio_index = [0.6025]
	for i in range (1, 50):
		physio_index. append (1.205 + physio_index [i - 1])

	# exit if conversation already processed
	if os.path.isfile (output_filename_pkl) and os.path.isfile (output_filename_png):
		print ("Conversation already processed")
		exit (1)

	# Read audio, and transcription file
	'''for file in glob.glob(data_dir + "/*"):
		if "left-reduc.TextGrid" in file:
			transcription_left = file
		elif "right-filter.TextGrid" in file:
			transcription_right = file

		elif "right-filter-palign.textgrid" in file:
			transcription_right_palign = file

		elif "left-reduc-palign.textgrid" in file:
			transcription_left_palign = file

		elif ".wav" in file:
			if args.left:
				if "left-reduc.wav" in file:
					rate, signal = wav.read (file)
					speech_activity = detect_speech_activity (file)
			else:
				if "right-filter.wav" in file:
					rate, signal = wav.read (file)
					speech_activity = detect_speech_activity (file)'''

	# Read audio, and transcription file
	for file in glob.glob(data_dir + "/*"):
		if "left" in file and ".TextGrid" in file and "palign.textgrid" not in file:
			transcription_left = file
		elif "right" in file and ".TextGrid" in file and "palign.textgrid" not in file:
			transcription_right = file

		elif "right" in file and  "palign.textgrid" in file:
			transcription_right_palign = file

		elif "left" in file and  "palign.textgrid" in file:
			transcription_left_palign = file

		elif ".wav" in file:
			if args.left:
				if "left" in file:
					rate, signal = wav.read (file)
					speech_activity = detect_speech_activity (file)
			else:
				if "right" in file:
					rate, signal = wav.read (file)
					speech_activity = detect_speech_activity (file)


	analytic_signal = hilbert(signal)
	envelope = np. abs (analytic_signal). tolist ()
	step_signal =  1.0 / rate

	signal_x = [0.0]
	for i in range (1, len (signal)):
		signal_x. append (step_signal + signal_x[i-1])

	# Select language
	nlp = sp.load('fr_core_news_sm')

	# Read TextGrid files
	# get the left part of the transcription

	parser = spp.sppasRW (transcription_left)
	tier_left = parser.read(). find ("Transcription")
	if  tier_left is None:
		tier_left = parser.read(). find ("IPUs")

	# get the right part of the transcription
	parser = spp.sppasRW (transcription_right)
	tier_right = parser.read(). find ("Transcription")
	if  tier_right is None:
		tier_right = parser.read(). find ("IPUs")

	if args.left:
		tier = tier_left
	else:
		tier = tier_right

	IPU, _ = ts. get_ipu (tier, 1)
	talk = ts. get_discretized_ipu (IPU, physio_index, 1)
	discretized_ipu = sample_square (IPU, physio_index)

	for i in range (len (discretized_ipu)):
		if discretized_ipu [i] < 0.2:
			discretized_ipu [i] = 0
		else:
			discretized_ipu [i] = 1

	# Overlap
	overlap = ts. get_overlap (tier_left, tier_right)

	# recation time
	reaction_time = ts. get_reaction_time (tier_left, tier_right)
	# Lexical richness
	richess_lex1 = ts.generate_RL_ts (tier, nlp, "meth1")
	richess_lex2 = ts.generate_RL_ts (tier, nlp, "meth2")

	# Time of Filled breaks, feed_backs
	# aligment
	if args.left:
		parser = spp.sppasRW (transcription_left_palign)
		tier_align = parser.read(). find ("TokensAlign")

	else:
		parser = spp.sppasRW (transcription_right_palign)
		tier_align = parser. read(). find ("TokensAlign")

	# Time of Filled breaks, feed_backs
	filled_breaks = ts. get_durations (tier_align, list_of_tokens =  FILLED_PAUSE_ITEMS)
	main_feed_items = ts. get_durations (tier_align, list_of_tokens =  MAIN_FEEDBACK_ITEMS)
	main_discourse_items = ts. get_durations (tier_align, list_of_tokens =  MAIN_DISCOURSE_ITEMS)
	laughters = ts. get_durations (tier_align, list_of_tokens =  LAUGHTER_FORMS)


	""" handle particle items separately as continuous time series """
	main_particles_items = ts. get_particle_items (tier, nlp, list_of_tokens =  MAIN_PARTICLES_ITEMS)


	x_emotions, polarity, subejctivity = ts. emotion_ts_from_text (tier, nlp)

	# Time series dictionary
	time_series = {
				"Signal": [signal_x, envelope],
				"SpeechActivity": speech_activity,
				"talk": talk,
				"IPU": IPU,
				"disc_IPU": discretized_ipu,
				"Overlap": overlap,
				"ReactionTime":reaction_time,
				"FilledBreaks":filled_breaks,
				"Feedbacks":main_feed_items,
				"Discourses":main_discourse_items,
				"Particles":main_particles_items,
				"Laughters":laughters,
				"LexicalRichness1":richess_lex1,
				"LexicalRichness2":richess_lex2,
 				"Polarity": [x_emotions, polarity],
				"Subjectivity": [x_emotions, subejctivity],
				}

	#labels = list (time_series. keys ())
	labels = ["Signal", "SpeechActivity", "talk", "IPU", "disc_IPU", "Overlap", "ReactionTime", "FilledBreaks", "Feedbacks", "Discourses",
				"Particles", "Laughters", "LexicalRichness1", "LexicalRichness2", "Polarity", "Subjectivity"]

	markers = ['.' for i in range (len (labels))]


	df = pd.DataFrame (index = physio_index)

	# Conbstruct a dataframe with smapled time serie according to the physio index
	df["Time (s)"] = physio_index
	#df ["Silence"] = silence
	for label in labels [:]:
		if label in ["talk", "disc_IPU"]:
			df [label] = time_series [label]
		elif (label in ["IPU", "SpeechActivity", "Overlap", "FilledBreaks", "Laughters", "Feedbacks", "Discourses"]):
			df [label] = sample_square (time_series [label], physio_index)
		elif label == "Particles":
			df [label] = sample_cont_ts (time_series [label], physio_index, mode = "binary")
		else:
			if label == "Signal":
				df [label] = ts. normalize (sample_cont_ts (time_series [label], physio_index))

			else:
				df [label] = sample_cont_ts (time_series [label], physio_index)

	if args.left:
		for i in range (len (labels)):
			labels [i] += "_left"
	df. columns = ["Time (s)"] + labels
	# Output files
	df.to_pickle(output_filename_pkl)

	ts. plot_df (df, labels, output_filename_png, figsize=(12,9), y_lim = [0,1.2])
	#ts. plot_time_series ([time_series[label] for label in labels], labels, colors[0:len (labels)], markers=markers,figsize=(20,16), figname = output_filename_1)
