from network_export import *
import pandas as pd
import numpy as np

class pyNet ():
	def __init__(self, input_dim, loss_function = "mse"):
		"""
			input_dim: tuple of integers
			loss_function: in {"mse", binary_cross_entropy", "cross_entropy"}
		"""
		l = v_unsigned()

		if type (input_dim) == int:
			l[:] = [input_dim]
		elif type (input_dim) == tuple:
			l[:] = list (input_dim)
		elif type (input_dim) == list:
			l[:] = input_dim
		else:
			raise ("Input dimension of the netwprk must be a single int, tuple, or list of int.")
		self. Net = Network  (l, loss_function)

	def push_back_dense (self, dense_layer):
		add_dense (self. Net, dense_layer)

	def push_back_lstm (self, lstm_layer):
		add_lstm (self. Net, lstm_layer)

	def summary (self):
		self. Net. summary ()

	def fit (self, X, Y, iters, shuffle = True):
		std_X = tensor_double ()
		for mat in X:
			std_X. append (nparray_to_mat (mat))

		std_Y = mat_double ()
		std_Y = nparray_to_mat (Y)

		self. Net. fit (std_X, std_Y, iters, shuffle)

	def score (self, X, Y):
		std_X = tensor_double ()
		for mat in X:
			std_X. append (nparray_to_mat (mat))

		std_Y = mat_double ()
		std_Y = nparray_to_mat (Y)
		return self. Net. score (std_X, std_Y)

	def predict (self, X):
		std_X = tensor_double ()
		for mat in X:
			std_X. append (nparray_to_mat (mat))
		return np. array (mat_to_list_of_list (self. Net. predict (std_X)))

	def input_features_scores (self):
		scores = np. abs (std_vector_to_py_list (self. Net. input_features_scores ()))
		sum_scores = np. sum (scores)
		return [a / sum_scores for a in scores]
