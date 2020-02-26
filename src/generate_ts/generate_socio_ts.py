import sys
import glob
import os
import pandas as pd

from joblib import Parallel, delayed
import multiprocessing
import argparse

import spacy as sp

sys.path.insert(0,'src/utils/SPPAS')
sys.path.insert(0,'.')

import src.utils.SPPAS.sppas.src.anndata.aio.readwrite as spp
import src. resampling as resampling


#====================================#
def get_interval (sppasObject):
	label = sppasObject. serialize_labels()
	location = sppasObject. get_location ()[0][0]

	start = location. get_begin (). get_midpoint ()
	stop = location. get_end (). get_midpoint ()

	start_radius = location. get_begin (). get_radius ()
	stop_radius = location. get_end (). get_radius ()

	# the radius represents the uncertainty of the localization
	return label, [start, stop], [start_radius, stop_radius]


#====================================#
def get_nb_ipus (subjects, left):

	subjects = ["sub-01"] + subjects
	if left:
	    file = open ("stats_ts/nb_ipus_words_left.csv","w+")
	else:
	    file = open ("stats_ts/nb_ipus_words_right.csv","w+")

	file. write ("Subject; Conversation; Number_IPUS; Number_Words\n")

	for subject in subjects:
		print (subject)
		conversations = glob. glob ("data/transcriptions/" + subject + "/*")
		conversations. sort ()

		nlp = sp.load('fr_core_news_sm')

		for conv in conversations:

			print (conv)
			nb_ipus = 0
			nb_words = 0
			if left:
			    transcr_file = glob. glob (conv + "/*left-reduc.TextGrid")
			else:
			    transcr_file = glob. glob (conv + "/*right-filter.TextGrid")

			if len (transcr_file) == 0:
			    file. write ("%s;%s;Nan;Nan\n"%(subject, conv. split ("/")[-1]))
			    continue

			else:
				transcr_file = transcr_file [0]

			parser = spp.sppasRW (transcr_file)
			if subject == 'sub-01':
				try:
					tier = parser.read(). find ("Transcription")
				except:
					tier = parser.read(). find ("IPUs")
			else:
				tier = parser.read(). find ("IPUs")

			for sppasOb  in tier:
				label, [start, stop], [start_r, stop_r] = get_interval (sppasOb)
				if label in ["#", "", " ", "***", "*"] :
					continue
				else:
					nb_words += len ( nlp (label))
					nb_ipus += 1

			file. write ("%s;%s;%s;%s\n"%(subject, conv. split ("/")[-1], nb_ipus, nb_words))

#====================================#
# generate time series from transcriptions files
def process_transcriptions (subject, left):
    print ("\t" + subject, 15*'-', '\n')

    if left:
        out_dir = "time_series/" + subject + "/speech_left_ts/"
    else:
        out_dir = "time_series/" + subject + "/speech_ts/"

    if not os. path. exists (out_dir):
        os. makedirs (out_dir)

    conversations = glob. glob ("data/transcriptions/" + subject + "/*")
    conversations. sort ()

    for conv in conversations:
        try:
            if left:
                os. system ("python3 src/generate_ts/speech_features.py %s %s --left" % (conv, out_dir))
            else:
                os. system ("python3 src/generate_ts/speech_features.py %s %s" % (conv, out_dir))
        except:
            print ("generate_socio_ts.py: Error in processing %s"%conv)

    print (subject + "Done ....")

#====================================#
# generate time series from videos

def process_videos (subject, type):

	print ("\t" + subject, 15*'-', '\n')
	out_dir_landMarks = "time_series/" + subject + "/facial_features_ts/"
	out_dir_eyetracking = "time_series/" + subject + "/eyetracking_ts/"
	out_dir_energy = "time_series/" + subject + "/energy_ts/"
	out_dir_emotions = "time_series/" + subject + "/emotions_ts/"
	out_dir_smiles = "time_series/" + subject + "/smiles_ts/"
	out_dir_dlibSmiles = "time_series/" + subject + "/dlib_smiles_ts/"

	for out_dir in [out_dir_landMarks, out_dir_eyetracking, out_dir_energy, out_dir_emotions, out_dir_smiles, out_dir_dlibSmiles]:
		if not os. path. exists (out_dir):
			os. makedirs (out_dir)

	videos = glob. glob ("data/videos/" + subject + "/*.avi")
	videos. sort ()

	# compute the index of the index BOLD signal frequency
	physio_index = [0.6025]
	for i in range (1, 50):
		physio_index. append (1.205 + physio_index [i - 1])

	for video in videos:
		#try:
		if type == "eye":
			os. system ("python3 src/generate_ts/eyetracking.py " + video + " " + out_dir_eyetracking)

		elif type == 'e':
			os.system("python3 src/generate_ts/generate_emotions_ts.py " +  video + " " + out_dir_emotions)

		elif type == 'f':
			os. system ("python3 src/generate_ts/facial_action_units.py " +  video + " " + out_dir_landMarks)

		if type == "c":
			os. system ("python3 src/generate_ts/colorfulness.py " + video + " " + out_dir_colors)

		elif type =="energy":
			os. system ("python3 src/generate_ts/energy.py " + video + " " + out_dir_energy)

		elif type =="dlib_smiles":
			os. system ("python3 src/generate_ts/dlib_smiles.py " + video + " " + out_dir_dlibSmiles)

		elif type =="smiles":
			try:
				if os. path. exists (out_dir_smiles + video.split('/')[-1]. split ('.')[0] + ".pkl"):
					print ("File already processed !")
					continue
				os. system ("Rscript src/generate_ts/generateSmiles.R " + out_dir_landMarks + video.split('/')[-1]. split ('.')[0] + " " + out_dir_smiles)

				if os. path. exists (out_dir_smiles  + video.split('/')[-1]. split ('.')[0] + ".csv"):
					csv_data = pd.read_csv (out_dir_smiles  + video.split('/')[-1]. split ('.')[0] + ".csv", sep = ';'). loc [: ,["time", "value"]]
					replace_dict = {"value":     {"S0": 0, "S1": 1, "S2": 2, "S3": 3, "S4": 4}}
					csv_data. replace (replace_dict, inplace = True)
					csv_data = pd. DataFrame (resampling. resample_ts (csv_data. values, physio_index, mode = "max"), columns = ["Time (s)", "Smile"])

				else:
					csv_data = []
					for t in physio_index:
						csv_data. append ([t, 0])
					csv_data = pd. DataFrame (csv_data, columns = ["Time (s)", "Smile"])

				csv_data.to_pickle (out_dir_smiles + video.split('/')[-1]. split ('.')[0] + ".pkl")
				os. system ("rm %s"%out_dir_smiles  + video.split('/')[-1]. split ('.')[0] + ".csv")

			except Exception as e:
				print (e)
				continue

#====================================#
if __name__ == '__main__':

	parser = argparse.ArgumentParser()

	parser. add_argument ('--subjects', '-s', nargs = '+', type=int,  help="List of subjects numbers, 0 to process all the subjects.")
	parser.add_argument("--type", "-t", help="type of data to process.", choices = ['c', 't', 'e', 'f', 'eye', 'energy', 'ipus', "smiles", "dlib_smiles"])
	parser.add_argument("--left", "-le", help="Process participant speech.", action="store_true")

	args = parser.parse_args()

	if args.subjects == [0]:
		subjects = ["sub-%02d"%i for i in range (1, 26)]
	else:
		subjects = ["sub-%02d"%i for i in args.subjects]

	print ("Processing subjects %s"%str (subjects))

	if not os. path. exists ("time_series"):
		os. makedirs ("time_series")

	nax_cores = multiprocessing.cpu_count() - 1

	if args. type == 't':
		Parallel (n_jobs = 7) (delayed(process_transcriptions) (subject, args.left) for subject in subjects)

	elif args. type == 'ipus':
	    get_nb_ipus (subjects, args.left)

	else:
		Parallel (n_jobs=3) (delayed(process_videos) (subject, args. type) for subject in subjects)
