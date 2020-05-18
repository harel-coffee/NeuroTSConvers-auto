# coding: utf8

#import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import glob
import os,sys,inspect
import argparse

import paralleldots
import goslate

from translate import Translator
import html

from speech_features import sample_cont_ts

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
maindir = os.path.dirname(parentdir)


sys.path.insert(0,'%s/src/utils/SPPAS'%maindir)
sys.path.insert(3,maindir)
import src.utils.SPPAS.sppas.src.anndata.aio.readwrite as spp


translator= Translator(to_lang="en", from_lang="fr")
paralleldots.set_api_key( "RMD2kM9YLzi7F8ZsgNXYJJAF6T67jrzh74tAkGAe6Xo" )

#================================================================
def get_interval (sppasObject):
	label = sppasObject. serialize_labels()
	location = sppasObject. get_location ()[0][0]

	start = location. get_begin (). get_midpoint ()
	stop = location. get_end (). get_midpoint ()

	start_radius = location. get_begin (). get_radius ()
	stop_radius = location. get_end (). get_radius ()

	# the radius represents the uncertainty of the localization
	return label, [start, stop], [start_radius, stop_radius]

#=================================================
def get_features_from_phrase (phrase, labels):

	proba = []
	emotion = paralleldots.emotion( phrase )
	abuse = paralleldots.abuse( phrase )
	sentiment = paralleldots.sentiment( phrase )

	for label in ["Fear", "Sad", "Bored", "Happy", "Excited", "Angry"]:
		#try:
		proba. append ( emotion["emotion"][label] )
		#except Exception as e:
			#print (e)
			#proba. append (0)

	for label in ["negative", "neutral", "positive"]:
		try:
			proba. append ( sentiment["sentiment"][label] )
		except Exception as e:
			print (e)
			proba. append (0)

	for label in ["abusive", "neither", "hate_speech"]:
		try:
			proba. append ( abuse[label] )
		except Exception as e:
			print (e)
			proba. append (0)

	return proba


#=================================================
def get_text_features (tier, axis, mode = "mean"):
	labels = ["Fear", "Sad", "Bored", "Happy", "Excited", "Angry", "negative", "neutral", "positive", "abusive", "neither", "hate_speech"]
	probabilities = []
	times = []

	df = pd.DataFrame ()

	for sppasOb  in tier:
		label, [start, stop], [start_r, stop_r] = get_interval (sppasOb)
		i = (start + stop) / 2.0

		if label in ["#", "", " ", "***", "*"] :
			proba_label = [0 for i in range (len (labels))]

		elif label in ["@", "@@", "@@@"]:
			proba_label = get_features_from_phrase ("laughter", labels)

		elif label in [u"o.k.",u"okay",u"ok",u"OK",u"O.K."]:
			proba_label = get_features_from_phrase ("Ok", labels)
		else:
			try:
				label_en = html. unescape (translator.translate(label))
				#print (label, "\n", label_en)
				proba_label = get_features_from_phrase (label_en, labels)
			except Exception as e:
				print ("Error with label %s"%label, "\n", e)
				proba_label = [0 for i in range (len (labels))]

		while (i < stop):
			times. append (i)
			probabilities. append (proba_label)
			i = i + ((start + stop) / 10.0)

	probabilities = np.array (probabilities)

	for j in range (len (labels)):
		df [ labels [j] ] = sample_cont_ts ( [ times, probabilities [:,j] ], axis, mode = mode )

	#df = pd.DataFrame (probabilities, columns = ["%s_txt"%label for label in labels])
	df. insert (0, "Time (s)", axis)

	# Filter : for small probabilities
	print (df)
	df [df < 0.2] = 0
	exit (1)

	return df

#===============================================================
def syllables (word):
	# If the first letter in the word is a vowel then it is a syllable.
	if i == 0 and word[i] in "aeiouy" :
		syllables = syllables + 1

	# Else if the previous letter is not a vowel.
	elif word[i - 1] not in "aeiouy" :
		# If it is no the last letter in the word and it is a vowel.
		if i < len(word) - 1 and word[i] in "aeiouy" :
			syllables = syllables + 1

		# Else if it is the last letter and it is a vowel that is not e.
		elif i == len(word) - 1 and word[i] in "aiouy" :
			syllables = syllables + 1

	# Adjust syllables from 0 to 1.
	if len(word) > 0 and syllables == 0 :
		syllables == 0
		syllables = 1

	return syllables
#===============================================================
def get_speech_rate (tier, axis, mode = "mean"):

	times = []
	speech_rate = []
	df = pd.DataFrame ()

	for sppasOb  in tier:
		label, [start, stop], [start_r, stop_r] = get_interval (sppasOb)
		i = (start + stop) / 2.0

		if label in ["#", "", " ", "***", "*"] :
			rate = 0

		else:
			rate = syllables (label) / (stop - start)

		while (i < stop):
			times. append (i)
			speech_rate. append (rate)
			i = i + ((start + stop) / 10.0)

	return time, rate

#===========================================================
# Compute richess_lexicale with 2 methods
# method 1 : number of adj + number of adv / total number of tokens
# method 2 : number of different tokes / total
def richess_lexicale (phrase, nlp,  method = "meth1"):

	doc = nlp (phrase)

	if method == "meth2":
		nb_adj = 0
		nb_adv = 0
		total_tokens = 0

		for token in doc:
			if token.pos_ != "PUNCT":
				total_tokens += 1
			if token.pos_ == "ADJ":
				nb_adj += 1

			if token.pos_ == "ADV":
				nb_adv += 1

		if total_tokens == 0:
			return 0

		return float (nb_adv + nb_adj) / total_tokens

	elif method == "meth1":
		token_without_punct = []

		for token in doc:
			#if token. IS_PUNCT == 0:
			if token.pos_ != "PUNCT":
				token_without_punct. append (token.lemma_)

		# List of different tokens
		different_tokens = set (token_without_punct)

		if len (token_without_punct) == 0:
			return 0

		return float (len (different_tokens)) / len (token_without_punct)

	else:
		print ("Error, the methode name is no correct, chose 'methd1' or 'meth2'")
		exit (1)

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
			conversation_name = "text_features_left"
		else:
			conversation_name = "text_features"

	output_filename_png = out_dir +  conversation_name + ".png"
	output_filename_pkl = out_dir +  conversation_name + ".pkl"

	# Select language
	if args. language == "fr":
		nlp = sp.load('fr_core_news_sm')
	elif args. language == "eng":
		nlp = sp.load('en_core_web_sm')

	print ("---------", conversation_name, "---------")
	# Index variable
	physio_index = [0.6025]
	for i in range (1, 50):
		physio_index. append (1.205 + physio_index [i - 1])

	# exit if conversation already processed
	if os.path.isfile (output_filename_pkl):
		print ("Conversation already processed")
		exit (1)

	# Read audio, and transcription file
	for file in glob.glob(data_dir + "/*"):
		if "left" in file and ".TextGrid" in file and "palign.textgrid" not in file:
			transcription_left = file
		elif "right" in file and ".TextGrid" in file and "palign.textgrid" not in file:
			transcription_right = file


	# Read TextGrid files
	parser = spp.sppasRW (transcription_left)
	tier_left = parser.read(). find ("Transcription")
	# get the right part of the transcription
	parser = spp.sppasRW (transcription_right)
	tier_right = parser.read(). find ("Transcription")


	if args.left:
		tier = tier_left
	else:
		tier = tier_right

	# Get feautures and resample observations
	df = get_text_features (tier, physio_index)

	# save data
	df.to_pickle(output_filename_pkl)
