import sys
import glob
import os

from joblib import Parallel, delayed
import multiprocessing
import argparse

import spacy as sp

sys.path.insert(0,'src/utils/SPPAS')
sys.path.insert(0,'.')

import src.utils.SPPAS.sppas.src.anndata.aio.readwrite as spp


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


def usage():
    print ("execute the script with -h for usage.")



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
            print ("Error in processing %s"%conv)

    print (subject + "Done ....")

#====================================#
# generate time series from videos

def process_videos (subject, type):

	print ("\t" + subject, 15*'-', '\n')
	out_dir_landMarks = "time_series/" + subject + "/facial_features_ts/"
	out_dir_eyetracking = "time_series/" + subject + "/eyetracking_ts/"
	out_dir_energy = "time_series/" + subject + "/energy_ts/"

	for out_dir in [out_dir_landMarks, out_dir_eyetracking, out_dir_energy]:
		if not os. path. exists (out_dir):
			os. makedirs (out_dir)

	videos = glob. glob ("data/videos/" + subject + "/*.avi")
	videos. sort ()

	for video in videos:
		try:
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
		except:
			print ("Error in processing video%s"%video)

#====================================#

if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	#parser.add_argument("subject", help="the subject name (for example sub-01), 'all' (default) to process all the subjects.", default="all")

	'''parser.add_argument("--nb_ipus","-ipus",  help="Process transcriptions.", action="store_true")
	parser.add_argument("--transcriptions","-t",  help="Process transcriptions.", action="store_true")
	parser.add_argument("--emotions", "-e", help="Process emotions.", action="store_true")
	parser.add_argument("--colors", "-c", help="Images colors.", action="store_true")
	parser.add_argument("--energy", "-energy", help="Images colors.", action="store_true")
	parser.add_argument("--eyetracking", "-eye", help="Process eye tracking.", action="store_true")
	parser.add_argument("--facial", "-f", help="Process landmarks.", action="store_true")'''

	parser. add_argument ('--subjects', '-s', nargs = '+', type=int,  help="List of subjects numbers, 0 to process all the subjects.")
	parser.add_argument("--type", "-t", help="type of data to process.", choices = ['c', 't', 'e', 'f', 'eye', 'energy', 'ipus'])
	parser.add_argument("--left", "-le", help="Process participant speech.", action="store_true")

	args = parser.parse_args()

	if args.subjects == [0]:
		subjects = ["sub-%02d"%i for i in range (2, 25)]
	else:
		subjects = ["sub-%02d"%i for i in args.subjects]

	if not os. path. exists ("time_series"):
		os. makedirs ("time_series")

	nax_cores = multiprocessing.cpu_count() - 1


	if args. type == 't':
		Parallel (n_jobs = 7) (delayed(process_transcriptions) (subject, args.left) for subject in subjects)

	elif args. type == 'ipus':
	    get_nb_ipus (subjects, args.left)

	else:
		Parallel (n_jobs=1) (delayed(process_videos) (subject, args. type) for subject in subjects)

    #except:
    	#print ("Error in Parallel loop")
