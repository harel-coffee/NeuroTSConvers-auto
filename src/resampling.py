import numpy as np
import math
import scipy.stats as sc

#========================================
def compute_energy (x, rotation = False, pixel = True):
	"""
		Compute the sum of squared differences: this is proportional to the kinetic enery of a variable
		x: np array, representing the speed, or the gradient vector
		pixel: whether the coordinates are in pixel
		rotation: if True  the values are considered as rotations in degree
	"""

	if (len (x) <= 1):
		return 0

	k_energy = 0

	if pixel:
		to_mm = 0.0002645833
	else:
		to_mm = 1

	for i in range (x. shape [1]):
		# if rotation is True, we transform rotations from degree to radian
		if rotation:
			#vect = [math. radians (a) for a in x[:,i]]
			vect = [a for a in x[:,i]]
		else:
			# convert pixel to mm
			vect = [a * to_mm  for a in x[:,i]]

		k_energy += np. sum (np. square (vect))

	return k_energy

#==========================================
def regroupe_data (list_, mode, rotation, pixel):
	"""
		reducing a list of lists into one list based on:
		mean, sum, std, mode, max, or binary
	"""

	if mode == "mean":
		return np.nanmean (list_, axis=0). tolist ()
	elif mode == "sum":
		return np. sum (list_, axis=0). tolist ()
	elif mode == "std":
		return np.std (list_, axis=0). tolist ()
	elif mode == "max":
		return np.nanmax (list_, axis=0). tolist ()
	elif mode == "mode":
		return sc.mode (list_, axis=0, nan_policy = 'omit')[0][0]. tolist ()
	elif mode == "binary":
		res = np.nanmean (list_, axis=0). tolist ()
		for i in range (len (res)):
			if res [i] > 0:
				res [i] = 1
			else:
				res [i] = 0
		return res
	elif mode == "energy":
		return [compute_energy (np. array (list_), rotation, pixel)]


#===================================================
def resample_ts (timeSries, index, mode = "mean", rotation = False, pixel = True):
	set_of_points = [[] for x in range (len (index))]
	y = [[] for x in range (len (index))]

	if timeSries. shape[1] < 2:
		raise ("Input time series must have an index column in addition to other columns.")

	if timeSries. shape[0] < 2:
		raise ("Not enough observations in the input time series.")

	for i in range (len (timeSries)):
		if ((0 <= timeSries [i,0]) and (timeSries [i,0]) <= index [0]):
			set_of_points [0]. append (timeSries [i, 1:])
			continue
		for j in range (0, len (index)):
			if ((index [j - 1] < timeSries [i,0]) and (index [j] >= timeSries [i,0])):
				set_of_points [j]. append (timeSries [i, 1:])
				break

	for j in range (0, len (y)):
		if len (set_of_points[j]) > 0 and len (set_of_points[j][0]) > 0:
			y[j] = regroupe_data (set_of_points[j], mode, rotation, pixel)
		else:
			if mode == "energy":
				y[j] = [0]
			else:
				y[j] = [0 for i in range (1, timeSries. shape [1])]

	return np. insert (np. array (y), 0, index, 1)
