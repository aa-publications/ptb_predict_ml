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

def convert_to_dmatrix(feat_mat_file, final_labels_df):
    # feat_mat_file : str
    #     full path to tsv file with GRIDS x FEATURES (first column should be GRIDS) with header
    # final_labels_df : pandas.DataFrame
    #     one row per 'GRID' and its 'delivery_date' and its classification 'label'


    X_mat, y_labels, full_df = load_X_y(feat_mat_file, final_labels_df)
    X_train, y_train, X_test, y_test, annotated_df = create_held_out(X_mat, y_labels, full_df)

    corrected_col_dict = correct_col_names(annotated_df)

    change_made = False
    mapped_col_df = pd.DataFrame()
    for key, value in corrected_col_dict.items():
        mapped_col_df = mapped_col_df.append(pd.DataFrame({"orig_col_name": [key], "new_col_name": [value]}))

        if not (key == value):
            change_made = True
            break

    if change_made:
        annotated_df.rename(columns=corrected_col_dict, inplace=True)

    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names = annotated_df.columns[1:-2])
    dtest = xgb.DMatrix(X_test, label=y_test, feature_names = annotated_df.columns[1:-2])

    return dtrain, dtest, annotated_df, mapped_col_df


def convert_to_xgbDataset(feat_mat_file, final_labels_df):
    # feat_mat_file : str
    #     full path to tsv file with GRIDS x FEATURES (first column should be GRIDS) with header
    # final_labels_df : pandas.DataFrame
    #     one row per 'GRID' and its 'delivery_date' and its classification 'label'


    X_mat, y_labels, full_df = load_X_y(feat_mat_file, final_labels_df)
    X_train, y_train, X_test, y_test, annotated_df = create_held_out(X_mat, y_labels, full_df)


    train_full_df = annotated_df.loc[annotated_df['partition']=='grid_cv', :].copy()
    train_full_df.drop(['partition'], axis=1, inplace=True)
    train_full_df.reset_index(inplace=True, drop=True)
    X_train_subset, y_train_subset, X_eval, y_eval, annotated_train_df = create_held_out(X_train, y_train, train_full_df)

    feat_names = list(annotated_df.columns.difference(['GRID', 'label','partition']))
    corrected_col_dict = correct_col_names(annotated_df)

    change_made = False
    mapped_col_df = pd.DataFrame()
    for key, value in corrected_col_dict.items():
        mapped_col_df = mapped_col_df.append(pd.DataFrame({"orig_col_name": [key], "new_col_name": [value]}))

        if not (key == value):
            change_made = True


    if change_made:
        annotated_df.rename(columns=corrected_col_dict, inplace=True)

    # add feature_name
    xgb_train = xgb.DMatrix(X_train_subset, label=y_train_subset, feature_names=feat_names)
    xgb_eval = xgb.DMatrix(X_eval, label=y_eval, feature_names=feat_names)
    xgb_train_eval = xgb.DMatrix(X_train, label=y_train, feature_names=feat_names)
    xgb_test = xgb.DMatrix(X_test, label=y_test, feature_names=feat_names)

    return xgb_train, xgb_eval, xgb_train_eval, xgb_test, annotated_df, mapped_col_df

# -----------
# MAIN
# -----------

if __name__ == "__main__":

    # run with arguments passed in
    parser = argparse.ArgumentParser(description='convert input files to Dmatrix')

    # REQUIRED ARGUMENTS IN ORDER
    parser.add_argument('feature_file', action='store', type=str)

    results = parser.parse_args()
    feature_file_path = results.feature_file
    start_m = time.time()

    # paths
    DELIVERY_LABELS_FILE = "/dors/capra_lab/users/abraha1/projects/PTB_phenotyping/data/combined_cohorts_07_12_2018/full_dataset_characterization/est_delivery_date_at_least_one_icd_cpt_ega.tsv"


    features_file_list = [feature_file_path]



    ###
    #   LOAD DATA
    ###


    final_labels_df = load_labels(DELIVERY_LABELS_FILE)
    print("Loaded the first delivery date in EHR.")



    for ffile in features_file_list:
        print("on {}".format(ffile))

        # =============  file names =============
        # feature file MUST END in *_feat_mat.tsv
        dtrain_file = os.path.join(os.path.dirname(ffile),  os.path.basename(ffile).replace('_feat_mat.tsv', '_dtrain.dmatrix'))
        dtest_file = os.path.join(os.path.dirname(ffile),  os.path.basename(ffile).replace('_feat_mat.tsv', '_dtest.dmatrix'))

        annotated_file = os.path.join(os.path.dirname(ffile),  os.path.basename(ffile).replace('_feat_mat.tsv', '_annotated.tsv.feather'))
        new_col_name_mapping_file = os.path.join(os.path.dirname(ffile),  os.path.basename(ffile).replace('_feat_mat.tsv', 'new_col_name_mapping.tsv'))

        # =============  load data =============
        dtrain, dtest, annotated_df, mapped_col_df = convert_to_dmatrix(feat_mat_file, final_labels_df, new_col_name_mapping_file, )




        # =============  write =============

        annotated_df.reset_index(drop=True).to_feather(annotated_file)
        mapped_col_df.to_csv(new_col_name_mapping_file)


        print("num bytes in df = {}".format(annotated_df.memory_usage().sum()))
        dtrain.save_binary(dtrain_file)
        dtest.save_binary(dtest_file)
        print("Created and saved DMatrix files.")
        print("Number of Positives/Negatives:\n\ttrain: {}/{}\n\ttest: {}/{}".format(np.sum(y_train == 1), np.sum(y_train == 0), np.sum(y_test == 1), np.sum(y_test == 0)))
        print("Output written to {}".format(os.path.dirname(dtrain_file)))
