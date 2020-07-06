#!/bin/python
# This script will
#
#
#
# Abin Abraham
# created on: 2020-03-24 00:53:18


import os
import sys
import pickle
import numpy as np
import pandas as pd
import importlib
import xgboost as xgb

from datetime import datetime
sys.path.append('/dors/capra_lab/users/abraha1/projects/PTB_phenotyping/scripts/rand_forest_ptb_classification')
from rand_forest_helper_functions import label_targets, create_icd_feature_matrix, filter_icd_by_delivery_date, fllter_by_days_to_delivery

DATE = datetime.now().strftime('%Y-%m-%d')



sys.path.append("/dors/capra_lab/users/abraha1/projects/PTB_phenotyping/scripts/rand_forest_ptb_classification/manuscript/0_helper_func")
from manip_trained_models_funcs import unpack_input_data, upickle_xgbmodel, extract_train_df, extract_test_df, validate_best_model, get_preds


# PATHS
RESULTS_DIR="/dors/capra_lab/users/abraha1/projects/PTB_phenotyping/results/ptb_predict_machine_learning/2019_05_20_time_since_conception/up_to_28_weeks_since_preg_start_icd9_cpt_count"
timepoint='28_weeks'
input_file = os.path.join(RESULTS_DIR, f'input_data_up_to_{timepoint}_since_preg_start_icd9_cpt_count-2019-06-19.tsv')
model_file = os.path.join(RESULTS_DIR, f'best_xgb_model_up_to_{timepoint}_since_preg_start_icd9_cpt_count-2019-06-19.pickle')

risk_file="/dors/capra_lab/users/abraha1/projects/PTB_phenotyping/scripts/rand_forest_ptb_classification/manuscript/time_since_conception/figures/risk_fx_matrix_w_na.tsv"

FEAT_OUTPUT_DIR="/dors/capra_lab/users/abraha1/projects/PTB_phenotyping/data/ptb_predict_machine_learning/feature_matrices/time_since_delivery_risk_fx_features"


###
#   MAIN
###


# load input_file
input_df=pd.read_csv(input_file, sep="\t")
risk_df = pd.read_csv(risk_file, sep="\t")

# convert nans to low-risk
risk_df.fillna('Low-risk', inplace=True)
risk_df.head()


# create risk matrix with only risk factors
input_risk_df = pd.merge(risk_df[risk_df.GRID.isin(input_df.GRID)],input_df.loc[:, ['GRID', 'label','partition']], on='GRID', how='inner')

input_risk_df[input_risk_df == 'High-risk'] = 1
input_risk_df[input_risk_df == 'Low-risk'] = 0

# WRITE
input_risk_df.to_csv(os.path.join(FEAT_OUTPUT_DIR, 'up_to_28_weeks_since_preg_start_risk_fx_feat_mat.tsv'), index=False, sep="\t")


# create dmatrices

# X_train, y_train, X_test, y_test, xgb_model, input_data = unpack_input_data(input_file, model_file)


X_train_df = input_risk_df.loc[input_risk_df['partition']=='grid_cv', input_risk_df.columns.difference(['GRID','label','partition'])].copy()
X_test_df = input_risk_df.loc[input_risk_df['partition']=='held_out', input_risk_df.columns.difference(['GRID','label','partition'])].copy()


X_train = X_train_df.values
X_test = X_test_df.values
y_train = input_risk_df.loc[input_risk_df['partition']=='grid_cv', 'label'].apply(lambda x: 1 if (x =='preterm') else 0).values
y_test = input_risk_df.loc[input_risk_df['partition']=='held_out', 'label'].apply(lambda x: 1 if (x =='preterm') else 0).values

dtrain = xgb.DMatrix(X_train, label=y_train, feature_names = X_train_df.columns)
dtest = xgb.DMatrix(X_test, label=y_test, feature_names = X_test_df.columns)


dtrain_file = os.path.join(FEAT_OUTPUT_DIR,  'up_to_28_weeks_since_preg_start_risk_fx_dtrain.dmatrix')
dtest_file = os.path.join(FEAT_OUTPUT_DIR,  'up_to_28_weeks_since_preg_start_risk_fx_dtest.dmatrix')


dtrain.save_binary(dtrain_file)
dtest.save_binary(dtest_file)
annotated_file = os.path.join(FEAT_OUTPUT_DIR,   'up_to_28_weeks_since_preg_start_risk_fx_annotated.tsv.feather')
input_risk_df.reset_index(drop=True).to_feather(annotated_file)
