import os, sys, inspect
import importlib
import pandas as pd
import numpy as np
import argparse

from src. prediction. tools import get_behavioral_data, list_convers, toSuppervisedData, concat_
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

#from src. feature_selection. reduction import manual_selection, reduce_train_test, ref_local
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
maindir = os.path.dirname(parentdir)

resampling_spec = importlib.util.spec_from_file_location("reduce", "%s/src/feature_selection/reduction.py"%maindir)
reduce = importlib.util.module_from_spec(resampling_spec)
resampling_spec.loader.exec_module(reduce)

#============================================
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

#============================================
class new_model:

    models = []
    n_modes = 0

    #-----------------------------------#
    def __init__(self, begin_end, models = None, strategy = "post"):
        """
        features: dictionary of modalities (same order of X columns)

        begin_end: begin and end indices of each modality

        strategy: the way of modalities predictions fusion
                  in the choices ["post", "mean"]
                  post: using a prediction model
                  mean: averaging the prediction of each modality
        """

        # Number of modalities
        self.n_modes = len (begin_end)
        self. begin_end = begin_end
        self. strategy = strategy

        params = {'bootstrap': True, 'max_depth': 100, 'max_features': 'auto', 'n_estimators': 100, 'random_state': 5}

        if models == None:
            for i in range (self.n_modes):
                self. models. append (RandomForestClassifier (**params))
                #self. models. append (SVC(gamma='auto'))

        else:
            for i in range (self.n_modes):
                model = RandomForestClassifier ()
                #model = LogisticRegression ()
                model. set_params (**models [i])
                self. models. append (model)

        if self. strategy == "post":
            self. post_model = RandomForestClassifier (**params)

    #-----------------------------------#
    def fit (self, X, y):
        """
        fit X with modalities, each modality will be fitted with a model, with the same order of features dictionary
        X: predictive variables
        Y: target variable
        """

        i = 0
        for [begin, end] in self. begin_end:
            if begin < end:
                self.models [i]. fit (X[:, begin: end], y)
            i += 1

        # Fit the post predict model
        predictions = np. empty ([len (X), self.n_modes], dtype = float)

        i = 0
        for [begin, end] in self. begin_end:
            if begin < end:
                predictions[:,i] = self.models [i]. predict (X[:, begin: end])
            i += 1

        if self. strategy == "post":
            if self.n_modes > 1:
                self. post_model. fit (predictions, y)

    #-----------------------------------#
    def predict (self, X):

        predictions = np. empty ([len (X), self.n_modes], dtype = float)

        i = 0
        for [begin, end] in self. begin_end:
            predictions[:,i] = self.models [i]. predict (X[:, begin: end])
            i += 1

        if self.n_modes == 1:
            return predictions. flatten ()

        # Compute final predictions using the mean of predictions of each modality
        if self. strategy == "mean":
            final_preds = np. sum (predictions, axis = 1). flatten ()
            for i in range (len (final_preds)):
                if final_preds[i] >= 1:
                    final_preds[i] = 1
                else:
                    final_preds[i] = 0

            return final_preds

        elif self. strategy == "post":
            # Compute final predictions using a "post" prediction model on the  predictions of each modality
            post_preds = self. post_model. predict (predictions)
            return post_preds


#--------------------------------------------------------------------#
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
    model = new_model (features[-1], 4)
    X = data[:,1:]
    y =  data[:,0]
    print (X.shape)
    model. fit (X,y)

    predictions = model. predict (X)
    print (18*'-', "\n Predictions \n", predictions, 18*'-')'''
