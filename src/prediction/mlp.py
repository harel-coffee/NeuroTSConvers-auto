# -*- coding: utf-8 -*-
# Author: Youssef Hmamouche

from keras.models import Sequential
from keras.layers.core import Dense, Dropout
import tensorflow as tf
import numpy as np

#--------------------------------------------------------
class MLP_MODEL:
    #---------------------------------------#
    def __init__(self, lag):
        self. lag = lag
        self. model = None

    def save (self, file_path):
        self. model. save (file_path)

    def load_model (self, file_path):
        self. model. load_model (file_path)

    #---------------------------------------#
    def fit (self, X, Y, epochs = 100, batch_size = 1, verbose = 0, shuffle = False):

        n_samples = X.shape[0]
        n_features = X.shape[1]
        n_neurons_1 = int (0.67 * (n_features + 1))
        n_neurons_2 = int (0.33 * (n_features + 1))

        self. model = Sequential()
        self. model. add (Dense (n_neurons_1, activation='relu',  input_dim = n_features))
        self. model.add(Dropout(0.2))
        #self. model. add (Dense (n_neurons_2, activation='relu',  input_dim = n_features))
        self. model. add (Dense (1, activation='sigmoid'))

        self.model.compile (loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
        self.model.fit (X, Y,  epochs = epochs, batch_size = batch_size, verbose = verbose, shuffle = shuffle)

    #---------------------------------------#
    def predict (self, X):
        preds = self. model. predict (X, batch_size = 1). flatten ()
        for i in range (len (preds)):
            if preds [i] < 0.5:
                preds [i] = 0
            else:
                preds [i] = 1
        return preds
