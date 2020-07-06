#!/bin/python
# Run xgboost (native python implementation) with hyeropt
#
#
#
# Abin Abraham
# created on: 2018-12-27 11:37:41
# Py 3.6.2
# Scikit  0.19.0
# Pandas 0.23.4
# Numpy 1.13.1
# xgboost 0.81
# hyperopt 0.1.1





import os
import sys
import time
import numpy as np
import pandas as pd

from datetime import datetime

sys.path.append('/dors/capra_lab/users/abraha1/projects/PTB_phenotyping/scripts/rand_forest_ptb_classification')
from hyperparam_tune import initialize, validate_best_model, create_held_out
from get_feature_importances import get_feature_importance, barplot_feat_importance
from train_test_rf import load_labels, load_X_y, compute_metrics, metrics_to_df, plot_roc, plot_pr

import xgboost as xgb
DATE = datetime.now().strftime('%Y-%m-%d')

import argparse


# -----------
# FUNCTIONS
# -----------


def correct_col_names(df):
    # check column nmaes for foribben character (per dmatrix): [, ] or <
    replace_dict = {"[": 'sq_bracket',
                    "]": 'sq_bracket',
                    "<": 'less_than' }
    og_cols =  df.columns.values

    corrected_cols = dict() # og_col name paried with corrected_colname
    forbid_char = [ "[", "]", "<"]

    for colname in og_cols:

        corrected_colname = colname.replace(
                forbid_char[0],replace_dict[forbid_char[0]]).replace(
                forbid_char[1],replace_dict[forbid_char[1]]).replace(
                forbid_char[2],replace_dict[forbid_char[2]])

        # store in dict
        corrected_cols[colname] = corrected_colname

    return corrected_cols




# -----------
# MAIN
# -----------

# run with arguments passed in
parser = argparse.ArgumentParser(description='convert input files to Dmatrix')

# REQUIRED ARGUMENTS IN ORDER
parser.add_argument('feature_file', action='store', type=str)

results = parser.parse_args()
feature_file_path = results.feature_file
start_m = time.time()

# paths
DELIVERY_LABELS_FILE = "/dors/capra_lab/users/abraha1/projects/PTB_phenotyping/data/ptb_predict_machine_learning/multiple_deliveries/2019-01-14-ptb_hx_cohort_labels.tsv"
# feature_file_path="/dors/capra_lab/users/abraha1/projects/PTB_phenotyping/data/ptb_predict_machine_learning/feature_matrices/predict_second_ptb/icd_cpt/raw_counts_icd_cpt_within_273_days_before_second_delivery_feat_mat.tsv"
features_file_list = [feature_file_path]



###
#   LOAD DATA
###



# load the second delivery label
# this file only has one second delivery label per person.
final_labels_df = pd.read_csv(DELIVERY_LABELS_FILE, sep="\t")
final_labels_df.rename(columns={'consensus_label_2':'consensus_delivery'}, inplace=True)

final_labels_df_2 = final_labels_df.copy()
final_labels_df_2.rename(columns={'consensus_delivery':'label'}, inplace=True)
final_labels_df_2.rename(columns={'delivery_date_2':'delivery_date'}, inplace=True)


for ffile in features_file_list:
    print("on {}".format(ffile))


    # =============  file names =============
    dtrain_file = os.path.join(os.path.dirname(ffile),  os.path.basename(ffile).replace('_feat_mat.tsv', '_dtrain.dmatrix'))
    dtest_file = os.path.join(os.path.dirname(ffile),  os.path.basename(ffile).replace('_feat_mat.tsv', '_dtest.dmatrix'))

    annotated_file = os.path.join(os.path.dirname(ffile),  os.path.basename(ffile).replace('_feat_mat.tsv', '_annotated.tsv.feather'))
    new_col_name_mapping_file = os.path.join(os.path.dirname(ffile),  os.path.basename(ffile).replace('_feat_mat.tsv', 'new_col_name_mapping.tsv'))

    # =============  load data =============
    X_mat, y_labels, full_df = load_X_y(ffile, final_labels_df_2)
    X_train, y_train, X_test, y_test, annotated_df = create_held_out(X_mat, y_labels, full_df)

    corrected_col_dict = correct_col_names(annotated_df)

    change_made = False
    col_df = pd.DataFrame()
    for key, value in corrected_col_dict.items():
        col_df = col_df.append(pd.DataFrame({"orig_col_name": [key], "new_col_name": [value]}))

        if not (key == value):
            change_made = True
            break

    if change_made:
        annotated_df.rename(columns=corrected_col_dict, inplace=True)
        col_df.to_csv(new_col_name_mapping_file)

    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names = annotated_df.columns[1:-2])
    dtest = xgb.DMatrix(X_test, label=y_test, feature_names = annotated_df.columns[1:-2])

    # =============  write =============
    annotated_df.reset_index(drop=True).to_feather(annotated_file)

    print("num bytes in df = {}".format(annotated_df.memory_usage().sum()))
    dtrain.save_binary(dtrain_file)
    dtest.save_binary(dtest_file)
    print("Created and saved DMatrix files.")
    print("Number of Positives/Negatives:\n\ttrain: {}/{}\n\ttest: {}/{}".format(np.sum(y_train == 1), np.sum(y_train == 0), np.sum(y_test == 1), np.sum(y_test == 0)))
    print("Output written to {}".format(os.path.dirname(dtrain_file)))
