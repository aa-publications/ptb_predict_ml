#!/bin/python
# This script will has functions to manipulate already trained xgboost models.
#
#
#
# Abin Abraham
# created on: 2019-04-02 13:03:52

import os
import sys
import numpy as np
import pandas as pd
from time import time

import pickle

sys.path.append('/dors/capra_lab/users/abraha1/projects/PTB_phenotyping/scripts/rand_forest_ptb_classification')
from train_test_rf import compute_metrics, metrics_to_df

def unpack_input_data(input_file, model_file):
    '''split input data into train and test + model + input_data dataframe'''

    ss=time()
    print("loading {} ...".format(os.path.basename(input_file)))

    input_data = pd.read_csv(input_file, sep="\t")
    X_train, y_train, X_test, y_test = get_train_test_arrays(input_data)

    print("loading {} ...".format(os.path.basename(model_file)))
    xgb_model = upickle_xgbmodel(model_file)

    print("done loading. took {:.2f} minutes".format( (time() - ss)/60))


    return  X_train, y_train, X_test, y_test, xgb_model, input_data

def get_train_test_arrays(input_data):
    """

        extracts train and test arrays from input dataframe


        Parameters
        ----------
        input_data : pandas.DataFrame
            input dataframe used to train xgboost model. (individ x features);
            - last two columns are 'label', and 'paritition'
            - first column is 'GRID'


        Returns
        -------
        arrays
            returns arrays of train and test data


    """


    held_out_df = input_data.loc[input_data['partition']=='held_out'].copy()
    train_df = input_data.loc[input_data['partition']=='grid_cv'].copy()
    held_out_df.set_index('GRID',inplace=True)
    train_df.set_index('GRID',inplace=True)

    binarize = lambda x: 1 if x == 'preterm' else 0

    X_train = train_df.iloc[:,:-2]
    y_train = train_df.label.apply(binarize).values
    X_test = held_out_df.iloc[:,:-2]
    y_test = held_out_df.label.apply(binarize).values


    return X_train, y_train, X_test, y_test

def upickle_xgbmodel(model_file):
    xgb_model = pickle.load(open(model_file, "rb"))

    return xgb_model

def extract_train_df(input_df):

    # create train and test df
    train_df = input_df.loc[input_df['partition']=='grid_cv'].copy()
    train_df_no_labels = train_df.drop(['GRID','label','partition'], axis=1, inplace=False)

    return train_df_no_labels, train_df

def extract_test_df(input_df):

    # create train and test df
    test_df = input_df.loc[input_df['partition']=='held_out'].copy()
    test_df_no_labels = test_df.drop(['GRID','label','partition'], axis=1, inplace=False)

    return test_df_no_labels, test_df

def validate_best_model(best_rf, X_test, y_test):
    '''
   Validate fitted model on test data.

    Params
    ------
    best_rf : object
        fitted/trained scikit-compatible model
    X_test : np.array
        observations x feature matrix for testing
    y_test : vector
        labels corresponding to each row of X_test

    Returns
    -------
    metrics_results : dict
        evalaution_metrics: value pair

    metrics_df : pandas.DataFrame
        metrics_results converted to a dataframe

    model_param : object
        parameters of the scikit-compatible model

    '''
    print("Validating best model on held out set...")
    # use best model to predict on test set
    model_params = best_rf.get_params()
    y_pred = best_rf.predict(X_test)
    y_proba = best_rf.predict_proba(X_test)

    # compute metrics
    metrics_results = compute_metrics(y_test, y_pred, y_proba[:, 1])
    metrics_df = metrics_to_df('test', metrics_results)
    metrics_df.drop(['cv_iter'], axis=1, inplace=True)

    return metrics_results, metrics_df, model_params

def get_preds(best_rf, X_array):
    '''
   Validate fitted model on test data.

    Params
    ------
    best_rf : object
        fitted/trained scikit-compatible model
    X_array : np.array
        observations x feature matrix for testing

    Returns
    -------
    y_pred : np.array
        predcit label from xgboost model

    y_proba : np.array
        predcited probability label from xgboost model

    '''

    # get predictions
    y_pred = best_rf.predict(X_array)
    y_proba = best_rf.predict_proba(X_array)


    return y_pred, y_proba


# def predict(X, y, xgb_model, prefix):
#     '''
#    Validate fitted model on test data.
#
#     Params
#     ------
#     best_rf : object
#         fitted/trained scikit-compatible model
#     X_test : np.array
#         observations x feature matrix for testing
#     y_test : vector
#         labels corresponding to each row of X_test
#
#     Returns
#     -------
#     metrics_results : dict
#         evalaution_metrics: value pair
#
#     metrics_df : pandas.DataFrame
#         metrics_results converted to a dataframe
#
#     model_param : object
#         parameters of the scikit-compatible model
#
#     '''
#     print("Validating best model on held out set...")
#     # use best model to predict on test set
#
#     y_pred = xgb_model.predict(X)
#     y_proba = xgb_model.predict_proba(X)
#
#     # compute metrics
#     metrics_results = compute_metrics(y, y_pred, y_proba[:, 1])
#     metrics_df = metrics_to_df(prefix, metrics_results)
#     metrics_df.drop(['cv_iter'], axis=1, inplace=True)
#
#     return metrics_results, metrics_df