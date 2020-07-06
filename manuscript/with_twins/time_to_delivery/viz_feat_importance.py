#!/bin/python
# This script will ...
#
#
#
# Abin Abraham
# created on: 2019-09-23 09:50:45


import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime

import seaborn as sns
import pickle
from glob import glob
import rpy2
%load_ext rpy2.ipython


DATE = datetime.now().strftime('%Y-%m-%d')


sys.path.append("/dors/capra_lab/users/abraha1/projects/PTB_phenotyping/scripts/rand_forest_ptb_classification/manuscript/0_helper_func")
from manip_trained_models_funcs import unpack_input_data, upickle_xgbmodel, extract_train_df, extract_test_df

sys.path.append("/dors/capra_lab/users/abraha1/projects/PTB_phenotyping/scripts/rand_forest_ptb_classification/shap_feat_importance/")
from shaply_funcs import filter_shap

# -----------  PATHS  -----------
FEAT_DIR="/dors/capra_lab/users/abraha1/projects/PTB_phenotyping/results/ptb_predict_machine_learning/2019-02-02_manuscript_time_to_delivery_icd_cpt/without_age_race_count/up_to_90_days/"
SHAP_VAL_DIR="/dors/capra_lab/users/abraha1/projects/PTB_phenotyping/scripts/rand_forest_ptb_classification/manuscript/time_to_delivery/feature_importance/shap_pickle"

INPUT_DF_FILE=os.path.join(FEAT_DIR, 'input_data_up_to_90_days_before_delivery_icd9_cpt_count-2019-02-16.tsv')
XGB_MODEL_FILE=os.path.join(FEAT_DIR, 'best_xgb_model_up_to_90_days_before_delivery_icd9_cpt_count-2019-02-16.pickle')
SHAP_TRAIN_PICKLE = os.path.join(SHAP_VAL_DIR, '2019-06-13_up_to_90_days_shap_train.pickle')
SHAP_TEST_PICKLE = os.path.join(SHAP_VAL_DIR, '2019-06-13_up_to_90_days_shap_test.pickle')


OUTPUT_DIR="/dors/capra_lab/users/abraha1/projects/PTB_phenotyping/scripts/rand_forest_ptb_classification/manuscript/time_to_delivery/"


# -----------
# FUNCTIONS
# -----------

def melt_feat_and_shap(shap_array, df_w_labels, top_feats_df):
    """Short summary.

    Parameters
    ----------
    shap_array : numpy array
        shap values w/ GRIDs (rows) and features (columns)
        NOTE: last column is expected to be the sum of the row in shap array

    df_w_labels : pd.DataFrame
        GRIDs(row) by features (columns) w/ value of the feature. last two columns are 'label' and 'partition'

    top_feats_df : pd.DataFrame
        top feature extracted based on shap

    Returns
    -------
    top_feat_shap_df: pd.DataFrame
        Long dataframe with one row per GRID-FEATURE-FEATURE_VALUE-SHAP_VALUE

    """

    # convert shap array to df w/ GRIDS as index and column names
    shap_df = pd.DataFrame(shap_array[:,:-1], columns = df_w_labels.iloc[:, 1:-2].columns)
    shap_df.index = df_w_labels.GRID
    shap_df.reset_index(inplace=True)

    # keep only the top features
    col_to_keep=['GRID'] + top_feats_df.feature.tolist()
    top_shap_df = shap_df.loc[:, col_to_keep]
    top_feat_df = df_w_labels.loc[:, col_to_keep]

    # melt so that one row per GRID-FEATURE-FEATURE_VALUE-SHAP_VALUE
    long_top_feat_count_df = pd.melt(top_feat_df, id_vars="GRID", var_name='feat', value_name='feat_count')
    long_top_shap_df = pd.melt(top_shap_df, id_vars="GRID", var_name='feat', value_name='feat_shap')

    top_feat_shap_df = pd.merge(long_top_feat_count_df, long_top_shap_df, on=['GRID','feat'], how='inner')

    return top_feat_shap_df

# %%
# -----------
# MAIN
# -----------


# -----------  load and melt data  -----------

# load feature matrix, labels, and xgboost model
X_train, y_train, X_test, y_test, xgb_model, input_df =  unpack_input_data(INPUT_DF_FILE, XGB_MODEL_FILE)
train_df, train_df_w_labels = extract_train_df(input_df)
test_df, test_df_w_labels = extract_test_df(input_df)

# load pickled shap values
train_shap = pickle.load( open( SHAP_TRAIN_PICKLE, 'rb'))
test_shap = pickle.load( open( SHAP_TEST_PICKLE, 'rb'))

# take top 10 shap features
train_top_feats_descrip = filter_shap(train_shap[:,:-1], train_df, top_n=15)
long_shap_feat_df = melt_feat_and_shap(train_shap, train_df_w_labels, train_top_feats_descrip)

#

long_shap_feat_df.to_csv(os.path.join(OUTPUT_DIR, 'long_shap_feat_df.tsv'), sep="\t", index=False)
# train_top_feats_descrip.to_csv(os.path.join(OUTPUT_DIR, 'top15_feat_w_descript.tsv'), sep="\t", index=False)

train_top_feats_descrip.sort_values('long_descrip')