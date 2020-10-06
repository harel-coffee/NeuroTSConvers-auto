import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from celluloid import Camera
from ast import literal_eval

from matplotlib.ticker import (MultipleLocator, FormatStrFormatter, AutoMinorLocator)

import argparse


def nb_to_region (region):
	for i in range (len (region)):
		if region[i] in ["1", "LeftMotor"]:
			region[i] = "Left MC"
		if region[i] in ["2", "RightMotor"]:
			region[i] = "Right MC"

		if region[i] in ["3", "LeftSTS"]:
			region[i] = "Left STS"
		if region[i] == ["4", "RightMotor"]:
			region[i] = "Right STS"

		if region[i] in ["5", "LeftFusiformGyrus"]:
			region[i] = "Left FG"

		if region[i] in ["6", "RightFusiformGyrus"]:
			region[i] = "Right FG"

		if region[i] == "7":
			region[i] = "R-FP"
		if region[i] == "8":
			region[i] = "L-VPFC"
		if region[i] == "9":
			region[i] = "R-VPFC"


if __name__ == '__main__':
	parser = argparse. ArgumentParser ()
	requiredNamed = parser.add_argument_group('Required arguments')
	requiredNamed. add_argument ('--input_dir','-in', help = "path of input directory")
	args = parser.parse_args()

	#colors = ["darkblue", "brown", "slategrey", "darkorange", "red", "grey","blue", "indigo", "darkgreen"]

	# READ PREDICTIONS
	df = pd. read_csv ("%s/Outputs/predictions.csv"%args.input_dir, sep = ';', header = 0, index_col = 0)
	regions = list (df. columns)
	print (regions)
	index = df. index
	title = 'Brain activity predictions'

	ROIS_desc = pd. read_csv ("brain_areas.tsv", sep = '\t', header = 0)

	colors_of_rois =  []
	for region in regions:
		colors_of_rois. append (ROIS_desc. loc [ROIS_desc["Name"] == region, "Color"]. values[0])


	for i in range (len (colors_of_rois)):
		colors_of_rois[i] = literal_eval (colors_of_rois[i])[0:3]
		colors_of_rois[i] = [float (a) / 255 for a in colors_of_rois[i]]

	#print (colors_of_rois)

	# SAVE LEGENDS SEPARATELY
	fig = plt.figure()
	fig_legend = plt.figure(figsize=(6, 1))
	ax = fig.add_subplot(111)
	bars = ax.bar(range(len (regions)), range(len (regions)), color=colors_of_rois, label=regions)
	fig_legend.legend(bars.get_children(), regions, loc='center', frameon=False, ncol = 3)
	fig_legend. savefig ("%s/Outputs/legend.png"%args.input_dir)
	plt.clf ()
	plt. cla ()
	plt. close ()
	#exit (1)

	# SAVE PREDICTIONS AS A VIDEO
	fig, ax = plt.subplots (nrows = len (regions), ncols = 1, figsize=(14,8),  sharex=True)
	fig.text(0.5, 0.04, 'Time (s)', ha='center')
	fig.text(0.04, 0.5, title, va='center', rotation='vertical')

	camera = Camera(fig)
	legend_image = plt. figure (figsize = (3,5))


	nb_to_region (regions)

	for j in range (len (regions)):
		ax [j]. set_xlim (np.min (index), np. max (index) + 1)
		#ax [j]. set_xticks (index)
		#ax [j]. set_xticklabels (index, rotation=-10)
		ax [j]. xaxis.set_minor_locator(MultipleLocator(5))
		ax [j]. set_ylim (0, 1.1)

	for i in range (1,len (index)):
		for j in range (len (regions)):
			ax[j]. plot (index [:i], df. iloc [:i, j], linewidth = 2, color = colors_of_rois [j], alpha = 1, label = regions[j])
			#ax[j].legend () #loc='upper right', fancybox=True, shadow=True, ncol=4, fontsize = "medium", markerscale = 0.2, labelspacing = 0.1, handletextpad=0.2, handlelength=1)
			ax[j].legend(['%s'%regions[j]], loc='upper right', bbox_to_anchor = (1.12, 1), markerscale = 2)

		#plt.legend(['%s'%reg for reg in regions], ncol = 3)
		camera.snap()



	anim = camera.animate (repeat = False, interval = 1205)

	anim.save("%s/Outputs/predictions_video.mp4"%args.input_dir, extra_args=['-vcodec', 'h264', '-pix_fmt', 'yuv420p'])
	fig. savefig ("%s/Outputs/predictions.png"%args.input_dir)
	#plt. show ()
