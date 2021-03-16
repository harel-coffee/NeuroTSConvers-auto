import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.colors import hsv_to_rgb
from celluloid import Camera
from ast import literal_eval
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter, AutoMinorLocator)
import matplotlib.ticker as ticker
import nibabel as nib
import matplotlib as mpl

import argparse

if __name__ == '__main__':
	parser = argparse. ArgumentParser ()
	requiredNamed = parser.add_argument_group('Required arguments')
	requiredNamed. add_argument ('--input_dir','-in', help = "path of input directory")
	args = parser.parse_args()

	# READ PREDICTIONS
	df = pd. read_csv ("%s/Outputs/predictions.csv"%args.input_dir, sep = ';', header = 0, index_col = 0)
	regions = list (df. columns)

	index = df. index
	title = 'Brain activity predictions'

	ROIS_desc = pd. read_csv ("brain_areas.tsv", sep = '\t', header = 0)


	colors_of_rois =  []

# 1000 distinct colors:
	colors = [hsv_to_rgb([(i * 0.618033988749895) % 1.0, 1, 1]) for i in range(1000)]

	for k in range (1, len (regions) + 1):
		colors_of_rois. append (colors[k])

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

	mpl.style.use('seaborn')
	# SAVE PREDICTIONS AS A VIDEO
	fig, ax = plt.subplots (nrows = len (regions), ncols = 1, figsize=(10,5.6),  sharex=True)
	#fig.text(0.5, 0.04, 'Time (s)', ha='center')
	fig.text(0.5, 0.04, 'Time (s)', ha='center')
	fig.subplots_adjust(
	    top=0.981,
	    bottom=0.09,
	    left=0.03,
	    right=0.8,
	    hspace=0.2,
	    wspace=0.2
	)
	#plt.tight_layout(pad=0.05)
	#fig.text(0.04, 0.5, title, va='center', rotation='vertical')

	camera = Camera(fig)
	legend_image = plt. figure (figsize = (3,5))


	for j in range (len (regions)):
		ax [j]. set_xlim (np.min (index), np. max (index) + 1)
		ax [j]. xaxis.set_minor_locator(MultipleLocator(5))
		ax [j]. set_ylim (0, 1.1)
		ax [j]. yaxis. set_major_locator (ticker. MultipleLocator (1))
		ax [j]. set_yticklabels([])

	for i in range (1,len (index)):
		for j in range (len (regions)):
			ax[j]. plot (index [:i], df. iloc [:i, j], linewidth = 2, color = colors_of_rois [j], alpha = 1, label = regions[j])
			ax[j]. legend(['%s'%regions[j]], bbox_to_anchor=(1, 1), loc=2, borderaxespad=0.05, markerscale = 0.5, handlelength=0.5, fontsize = 10)
		camera. snap()


	anim = camera.animate (repeat = False, interval = 1205)

	anim.save("%s/Outputs/predictions_video.mp4"%args.input_dir, extra_args=['-vcodec', 'h264', '-pix_fmt', 'yuv420p'])
	fig. savefig ("%s/Outputs/predictions.png"%args.input_dir)
