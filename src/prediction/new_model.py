import os, sys, inspect
import importlib
import pandas as pd
import numpy as np
import argparse

from tools import get_behavioral_data, list_convers, toSuppervisedData, concat_

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier, GradientBoostingClassifier


#from src. feature_selection. reduction import manual_selection, reduce_train_test, ref_local

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
maindir = os.path.dirname(parentdir)

resampling_spec = importlib.util.spec_from_file_location("reduce", "%s/src/feature_selection/reduction.py"%maindir)
reduce = importlib.util.module_from_spec(resampling_spec)
resampling_spec.loader.exec_module(reduce)


def get_convers ():
    convers = list_convers ()
    hh_convers = []
    hr_convers = []

    for i in range (len (convers)):
        if i % 2 == 1:
            hr_convers. append (convers [i])
        else:
            hh_convers. append (convers [i])

    return hh_convers, hr_convers

class new_model:
    ## constructor
    modalities = []
    models = []
    card_mod = []
    n_modes = 0

    def __init__(self, features, lags = 3):
        """
        features: dictionary of modalities (same order of X columns)
        """
        modalities = [features [item] for item in features. keys ()]

        # Number of modalities
        self.n_modes = len (modalities)

        # Number of variables in each modality
        self.card_mod = [len (x) for x in modalities]

        self._lags = lags
        params = {'bootstrap': True, 'max_depth': 100, 'max_features': 'auto', 'n_estimators': 10, 'random_state': 5}

        for i in range (self.n_modes):
            self. models. append (RandomForestClassifier (**params))

    def fit (self, X, y):
        """
        fit X with modalities, each modality will be fitted with a model
        X: predictive variables
        Y: target variable
        """

        begin = 0
        for i in range (self.n_modes):
            end =  begin + (self._lags * self.card_mod [i])
            self.models [i]. fit (X[:, begin: end], y)
            begin = end

    def predict (self, X):
        predictions = np. empty ([len (X), self.n_modes], dtype = float)
        begin = 0
        for i in range (self.n_modes):
            end =  begin + (self._lags * self.card_mod [i])
            predictions[:,i] = self.models [i]. predict (X[:, begin: end])
            begin = end

        pred = np.mean (predictions, axis = 1)
        for i in range (len (pred)):
            if pred[i] > 0.5:
                pred[i] = 1.0
            else:
                pred[i] = 0.0
        return pred

'''if __name__ == '__main__':
    parser = argparse. ArgumentParser ()
    parser. add_argument ('--subjects', '-s', nargs = '+', type=int)
    parser. add_argument ('--regions','-rg', nargs = '+', type=int)
    args = parser.parse_args()

    subjects = ["sub-%02d"%i for i in args. subjects]

    # list of conversations
    hh_convers, hr_convers = get_convers ()

    """ get brain areas from their codes """
    brain_areas_desc = pd. read_csv ("brain_areas.tsv", sep = '\t', header = 0)
    brain_areas = []
    for num_region in args. regions:
        brain_areas. append (brain_areas_desc . loc [brain_areas_desc ["Code"] == num_region, "Name"]. values [0])

    """ get concatenated lagged data: 1 subject, all conversations, on dictionary of features """
    features = reduce. manual_selection (brain_areas [0])
    print (18* '-', "\n Subject: %s \n"%subjects [0], "ROI: %s \n"%args.regions [0], "Features: %s \n"%str (features[-1]), 18* '-')
    data = concat_ (subjects [0], brain_areas [0], hh_convers, 6, features[-1], add_target = False, reg = False)

    """ prediction """
    model = new_model (features[-1], 3)
    X = data[:,1:]
    y =  data[:,0]
    print (X.shape)
    model. fit (X,y)

    predictions = model. predict (X)
    #print (18*'-', "\n Predictions \n", predictions, 18*'-')'''
