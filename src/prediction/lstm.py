# -*- coding: utf-8 -*-
# Author: Youssef Hmamouche

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import LSTM, Dropout, Dense, Activation
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.models import Sequential

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

#--------------------------------------------------------
class LSTM_MODEL:
    def __init__(self, lag):
        self. look_back = lag
        self. model = None
        self.disc_value = 0.5

    def save (self, file_path):
        self. model. save (file_path)

    def load_model (self, file_path):
        self. model. load_model (file_path)

    def fit (self, X, Y, epochs = 30, batch_size = 32, verbose = 0, shuffle = True):

        n_features =  int (X.shape[1] / self.look_back)
        n_samples = X.shape[0]

        n_neurons = int (2 * n_features)

        new_shape = [n_samples, self.look_back, n_features]

        X_reshaped = np. reshape (X, new_shape, order = 'F')

        self.model = Sequential()
        self. model. add (LSTM (n_neurons, input_shape=(self. look_back , n_features)))
        self. model.add(Dropout(0.2))
        self. model. add (Dense(1, activation='sigmoid'))

        opt = SGD(lr=0.01)
        self.model.compile (loss = 'binary_crossentropy', optimizer = opt, metrics = ['accuracy'])
        self.model.fit (X_reshaped, Y,  epochs = epochs, batch_size = batch_size, verbose = verbose, shuffle = shuffle)

    def predict (self, X):
        X_reshaped = np. reshape (X, (X.shape[0], self. look_back, int (X.shape[1] / self. look_back)), order = 'F')
        preds = self. model. predict (X_reshaped, batch_size = 1). flatten ()
        return preds

    def get_normalized_weights (self):
        for layer in self.model.layers:
            weightLSTM = layer.get_weights()
            break
        warr,uarr, barr = weightLSTM
        weights = np. abs (np.mean (warr, axis=1)). tolist ()
        sum = np.sum (weights)
        return [a / sum for a in weights]


#--------------------------------------------------------
