# -*- coding: utf-8 -*-
# Author: Youssef Hmamouche

import numpy as np
from keras.layers import LSTM
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout

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

    def fit (self, X, Y, epochs = 100, batch_size = 1, verbose = 0, shuffle = True):

        #self.disc_value = np.mean (Y. flatten ()) - 0.06
        n_features =  int (X.shape[1] / self.look_back)
        n_samples = X.shape[0]

        n_neurons = int (0.67 * (n_features + 1))

        new_shape = [n_samples, self.look_back, n_features]
        X_reshaped = np. reshape (X, new_shape, order = 'F')

        self.model = Sequential()
        self. model. add (LSTM (n_neurons, input_shape=(self. look_back , n_features)))
        self. model. add (Dense(1, activation='sigmoid'))
        self.model.compile (loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
        #self. model.compile(loss='mean_squared_error', optimizer='adam')

        self.model.fit (X_reshaped, Y,  epochs = epochs, batch_size = batch_size, verbose = verbose, shuffle = shuffle)

    def predict (self, X):
        X_reshaped = np. reshape (X, (X.shape[0], self. look_back, int (X.shape[1] / self. look_back)), order = 'F')
        preds = self. model. predict (X_reshaped, batch_size = 1). flatten ()
        for i in range (len (preds)):
            if preds [i] <= self.disc_value:
                preds [i] = 0
            else:
                preds [i] = 1
        return preds

    def get_weights (self):
          return self. model. layers[0]. get_weights ()


#--------------------------------------------------------
