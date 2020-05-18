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

    def fit (self, X, Y):
        X = np.array (X)
        new_shape = [X.shape[0], self.look_back, int (X.shape[1] / self.look_back)]
        X_reshaped = X.reshape (new_shape)

        self.model = Sequential()
        self. model. add (LSTM (X_reshaped. shape [2], input_shape=(self. look_back , X_reshaped. shape [2]), dropout_W = 0.2))
        #self. model. add (Dense(1, activation='sigmoid'))
        self.model.add(Dense(1, activation='relu'))
        self.model.compile (loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

        self.model.fit (X_reshaped, Y,  epochs = 200, batch_size = 1, verbose = 0, shuffle = False)

    def predict (self, X):
        X_reshaped = X.reshape(X.shape[0], self. look_back, int (X.shape[1] / self. look_back))
        preds = self. model. predict (X_reshaped, batch_size = 1). flatten ()
        for i in range (len (preds)):
            if preds [i] < 0.5:
                preds [i] = 0
            else:
                preds [i] = 1
        return preds

'''def fit_lstm (X, Y, lag = 4, params = 0) :

    batch_size = 1
    look_back = lag
    X_reshaped = X.reshape (X. shape[0], look_back, int (X.shape[1] / look_back))

    model = Sequential()
    model.add (LSTM (30, input_shape=(look_back, X_reshaped.shape[2]), dropout_W = 0.6))
    model.add(Dense(1, activation='sigmoid'))
    model.compile (loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
    model.fit (X_reshaped, Y,  epochs = 100, batch_size = batch_size, verbose = 0, shuffle = False)

    return model


def lstm_predict (X, model, lag):

    look_back = 4
    X_reshaped = X.reshape(X.shape[0], look_back, int (X.shape[1] / look_back))
    preds = model. predict (X_reshaped, batch_size = 1). reshape (-1)
    for i in range (len (preds)):
        if preds [i] < 0.5:
            preds [i] = 0
        else:
            preds [i] = 1
    return preds'''



#--------------------------------------------------------
