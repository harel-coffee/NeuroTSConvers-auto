# -*- coding: utf-8 -*-
# Author: Youssef Hmamouche

import sys
import os
import argparse

stderr = sys.stderr
sys.stderr = open(os.devnull, 'w') # hide keras messages
from keras.layers import LSTM
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras import regularizers
from keras.utils import to_categorical

sys.stderr = stderr

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

#--------------------------------------------------------

def fit_lstm (X, Y, lag, params) :

    batch_size = 1
    #read Parameters
    nb_epochs = params ['epochs']
    nb_neurons = params ['neurons']

    look_back = 3
    X_reshaped = X.reshape (X. shape[0], look_back, int (X.shape[1] / look_back))

    model = Sequential()
    model.add (LSTM (units = int (X.shape[1] / look_back), stateful=True,  batch_input_shape = (batch_size, X_reshaped.shape[1], X_reshaped.shape[2]), kernel_initializer='random_uniform'))

    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))

    model.compile (loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
    model.fit (X_reshaped, Y,  epochs = 30, batch_size = batch_size, verbose = 0, shuffle = False)

    return model


def lstm_predict (X, model, lag):

    look_back = 3
    X_reshaped = X.reshape(X.shape[0], look_back, int (X.shape[1] / look_back))
    preds = model. predict (X_reshaped, batch_size = 1). reshape (-1)
    for i in range (len (preds)):
        if preds [i] < 0.5:
            preds [i] = 0
        else:
            preds [i] = 1
    return preds



#--------------------------------------------------------
