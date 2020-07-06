#!/bin/python
# This script will create summaries of lab data during pregnancy and create feature matrcies for xgboost prediction of preterm birth
#
#
#
# Abin Abraham
# created on: 2019-08-18 21:11:17
# updated on: 2019-09-05 08:58:05


import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime

import glob
import pickle

sys.path.append('/dors/capra_lab/users/abraha1/projects/PTB_phenotyping/scripts/rand_forest_ptb_classification')
from train_test_rf import load_labels


DATE = datetime.now().strftime('%Y-%m-%d')



# PATH
# DELIVERY_LABELS_FILE = "/dors/capra_lab/users/abraha1/projects/PTB_phenotyping/data/combined_cohorts_07_12_2018/full_dataset_characterization/est_delivery_date_at_least_one_icd_cpt_ega.tsv"
PKL_DELIVERY_LABELS_FILE="/dors/capra_lab/users/abraha1/projects/PTB_phenotyping/data/ptb_predict_machine_learning/feature_matrices/delivery_labels_from_load_labels.pickle"
PREG_LABS_DIR="/dors/capra_lab/users/abraha1/projects/PTB_phenotyping/data/ptb_predict_machine_learning/clinical_labs/davis_labs_survey"


# outputs
root_ = "/dors/capra_lab/users/abraha1/projects/PTB_phenotyping/data/ptb_predict_machine_learning/feature_matrices/clincal_labs/"
FEAT_OUTPUT_DIR = os.path.join(root_, "w_missing_values")
BINARY_FEAT_OUTPUT_DIR = os.path.join(root_, "binary_feat_matrix")

BY_TRI_FEAT_OUTPUT_DIR = os.path.join(root_, "by_tri_w_missing_values")
BINARY_BY_TRI_FEAT_OUTPUT_DIR = os.path.join(root_, "by_tri_binary_feat_matrix")


# -----------
# FUNCTIONS
# -----------

def get_stat(lab_df, first_delivery_date, stat):
    lab_df.delivery_date = pd.to_datetime(lab_df.delivery_date)
    lab_df['delivery_date_y_m'] = lab_df.delivery_date.dt.strftime('%Y-%m')


    # keep labs obtained during the first pregnancy
    lab_df['grid_delivery_date'] = lab_df.delivery_date_y_m + "_" + lab_df.grid
    keep_lab_df = lab_df.loc[lab_df.grid_delivery_date.isin(first_delivery_date.delivery_date_grid)].copy()

    if keep_lab_df.shape[0] == 0:
        return pd.DataFrame(), None, None

    short_name = keep_lab_df.lab_shortname.unique()[0]
    long_name = keep_lab_df.lab_longname.unique()[0]

    if stat == 'mean':
        gb_df = keep_lab_df.groupby('grid_delivery_date')['lab_value'].mean().reset_index()
    elif stat == 'median':
        gb_df = keep_lab_df.groupby('grid_delivery_date')['lab_value'].median().reset_index()
    elif stat == 'max':
        gb_df = keep_lab_df.groupby('grid_delivery_date')['lab_value'].max().reset_index()
    elif stat == 'min':
        gb_df = keep_lab_df.groupby('grid_delivery_date')['lab_value'].min().reset_index()


    return gb_df, short_name, long_name

def create_feat_mat(lab_files,  first_delivery_date, stat):

    # operate on first lab
    gb_df, short_name, long_name= get_stat(pd.read_csv(lab_files[0], sep="\t"),first_delivery_date, stat)
    gb_df.rename(columns={'lab_value':short_name}, inplace=True)

    # initialize first column
    stat_labs_df = pd.DataFrame()
    stat_labs_df = stat_labs_df.append(gb_df)

    # loop through all other lab files
    for counter, lab_file in enumerate(lab_files):

        if counter == 0:
            continue

        # print("{}: {}".format(counter, os.path.basename(lab_file)))

        try:
            lab_df = pd.read_csv(lab_file, sep="\t")
            lab_df.lab_longname.unique()[0]
            gb_df, short_name, long_name= get_stat(lab_df, first_delivery_date, stat)

            # if no labs for the first pregnancy...
            if gb_df.shape[0] == 0:
                continue

            gb_df.rename(columns={'lab_value':short_name}, inplace=True)

        except pd.errors.EmptyDataError:
            # print("EmptyDataError")
            continue

        stat_labs_df = pd.merge(gb_df, stat_labs_df, on='grid_delivery_date',how='outer')


    stat_labs_df['GRID'] = stat_labs_df.grid_delivery_date.apply(lambda x: x.split("_")[1])
    stat_labs_df.drop(['grid_delivery_date'], axis=1, inplace=True)

    cols = list(stat_labs_df.columns)
    cols.remove('GRID')
    stat_labs_df= stat_labs_df.loc[:, ['GRID'] + cols]


    return stat_labs_df

def convert_to_binary(stat_df):
    # convert feat matrix to binary with 'missing' NaN value converted to 0; all others to 1;

    return ((~stat_df.set_index('GRID').isnull())*1).reset_index()

def calc_tri_of_lab(lab_df):
    df = lab_df.copy()


    df.preg_start_date = pd.to_datetime(df.preg_start_date)
    df.lab_date = pd.to_datetime(df.lab_date)
    df['lab_trimester'] = (df.lab_date - df.preg_start_date).dt.days.apply(lambda x: np.digitize([x], bins=[0,84,182, 301], right=True)[0])

    return df

def create_feat_mat_by_tri(lab_files, final_labels_df, stat, trimester):

    stat_labs_df = pd.DataFrame()

    for counter, lab_file in enumerate(lab_files):

        # calc trimester of each lab
        try:
            lab_df = calc_tri_of_lab(pd.read_csv(lab_file, sep="\t"))
            tri_lab_df = lab_df.loc[lab_df['lab_trimester']==trimester].copy()
            gb_df, short_name, long_name= get_stat(tri_lab_df, final_labels_df, stat)

            if gb_df.shape[0] == 0:
                continue

            gb_df.rename(columns={'lab_value':short_name}, inplace=True)

            # initialize first column
            if stat_labs_df.shape[0]==0:
                stat_labs_df = stat_labs_df.append(gb_df)

            # merge
            else:
                stat_labs_df = pd.merge(gb_df, stat_labs_df, on='grid_delivery_date',how='outer')


        # if the lab_file is empty
        except pd.errors.EmptyDataError:
            continue


    stat_labs_df['GRID'] = stat_labs_df.grid_delivery_date.apply(lambda x: x.split("_")[1])
    stat_labs_df.drop(['grid_delivery_date'], axis=1, inplace=True)

    cols = list(stat_labs_df.columns)
    cols.remove('GRID')
    stat_labs_df= stat_labs_df.loc[:, ['GRID'] + cols]


    return stat_labs_df


# %%
# -----------
# MAIN
# -----------
if __name__ == '__main__':

    ##
    #   LOAD DELIVERY LABELS
    ##

    # final_labels_df = load_labels(DELIVERY_LABELS_FILE)
    final_labels_df = pickle.load( open(PKL_DELIVERY_LABELS_FILE, 'rb'))
    # pickle.dump( final_labels_df , open( '/dors/capra_lab/users/abraha1/projects/PTB_phenotyping/data/ptb_predict_machine_learning/feature_matrices/delivery_labels_from_load_labels.pickle', 'wb'))

    final_labels_df.delivery_date = pd.to_datetime(final_labels_df.delivery_date)
    final_labels_df['consensus_delivery_y_m'] = final_labels_df.delivery_date.dt.strftime('%Y-%m')
    final_labels_df['delivery_date_grid'] = final_labels_df.consensus_delivery_y_m + "_" + final_labels_df.GRID





    ###
    #   TAKE SUMMARY STATS OF CLINCIAL LABS
    ###
    lab_files = glob.glob(PREG_LABS_DIR +"/*.tsv")

    # # -----------  One feat matrix per summary stat  -----------
    # # note: the binary feat matrix will also only have one column per clinical lab
    #
    # for this_stat in ['mean','median','min','max']:
    #     stat_df = create_feat_mat(lab_files, final_labels_df, this_stat)
    #     binary_df = convert_to_binary(stat_df)
    #
    #     stat_df.to_csv(os.path.join(FEAT_OUTPUT_DIR, 'first_preg_{}_labs_feat_mat.tsv'.format(this_stat)), sep="\t", index=False, na_rep="NaN")
    #     binary_df.to_csv(os.path.join(BINARY_FEAT_OUTPUT_DIR, 'binary_present_labs_first_preg_{}_feat_mat.tsv'.format(this_stat)), sep="\t", index=False)
    #
    #
    #     ### create feat matrix by trimester
    #     for tri in [1, 2, 3]:
    #         tri_stat_df = create_feat_mat_by_tri(lab_files, final_labels_df, this_stat, tri)
    #         tri_binary_df = convert_to_binary(tri_stat_df)
    #
    #         tri_stat_df.to_csv(os.path.join(BY_TRI_FEAT_OUTPUT_DIR, 'first_preg_tri{}_{}_labs_feat_mat.tsv'.format(tri, this_stat)), sep="\t", index=False, na_rep="NaN")
    #         tri_binary_df.to_csv(os.path.join(BINARY_BY_TRI_FEAT_OUTPUT_DIR, 'binary_present_labs_first_preg_tri{}_{}_feat_mat.tsv'.format(tri, this_stat)), sep="\t", index=False)
    #

    # %%
    # -----------  One feat matrix with all of summary stats  -----------


    all_stats_df = pd.DataFrame()
    for counter, this_stat in enumerate(['mean','median','min','max']):

        stat_df = create_feat_mat(lab_files, final_labels_df, this_stat)
        # append stat to column names
        stat_df.columns = ['{}_{}'.format(x, this_stat) if x != 'GRID' else 'GRID' for x in stat_df.columns]

        if counter == 0:
            # initialize
            all_stats_df = all_stats_df.append(stat_df)
        else:
            all_stats_df = pd.merge(all_stats_df, stat_df, on='GRID', how='outer')

    # for binary matricies, we will only have one column per clinical labs since for example mean and median binary columns will be the same
    keep_one_stat_cols = [x for x in all_stats_df.columns if ((x == 'GRID') | ('mean' in x)) ]
    all_binary_df = convert_to_binary(all_stats_df.loc[:, keep_one_stat_cols])


    # all_stats_df.to_csv(os.path.join(FEAT_OUTPUT_DIR, 'first_preg_all_stats_labs_feat_mat.tsv'), sep="\t", index=False, na_rep="NaN")
    all_binary_df.to_csv(os.path.join(BINARY_FEAT_OUTPUT_DIR, 'binary_present_labs_first_preg_all_stats_feat_mat.tsv'), sep="\t", index=False)


    ### take all labs for each trimester...
    for tri in [1, 2, 3]:

        by_tri_all_stats_df = pd.DataFrame()
        for counter, this_stat in enumerate(['mean','median','min','max']):

            tri_stat_df = create_feat_mat_by_tri(lab_files, final_labels_df, this_stat, tri)
            # append stat to column names
            tri_stat_df.columns = ['{}_{}'.format(x, this_stat) if x != 'GRID' else 'GRID' for x in tri_stat_df.columns]

            if counter == 0:
                by_tri_all_stats_df = by_tri_all_stats_df.append(tri_stat_df)
            else:
                by_tri_all_stats_df = pd.merge(by_tri_all_stats_df, tri_stat_df, on='GRID', how='outer')

        keep_one_stat_cols_by_tri = [x for x in by_tri_all_stats_df.columns if ((x == 'GRID') | ('mean' in x)) ]
        by_tri_all_binary_df = convert_to_binary(all_stats_df.loc[:, keep_one_stat_cols_by_tri])


        # by_tri_all_stats_df.to_csv(os.path.join(BY_TRI_FEAT_OUTPUT_DIR, 'first_preg_tri{}_all_stats_labs_feat_mat.tsv'.format(tri)), sep="\t", index=False, na_rep="NaN")
        by_tri_all_binary_df.to_csv(os.path.join(BINARY_BY_TRI_FEAT_OUTPUT_DIR, 'binary_present_labs_first_preg_tri{}_all_stats_feat_mat.tsv'.format(tri)), sep="\t", index=False)
