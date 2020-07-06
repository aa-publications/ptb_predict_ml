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

sys.path.append('/dors/capra_lab/users/abraha1/projects/PTB_phenotyping/scripts/rand_forest_ptb_classification')
from rand_forest_helper_functions import label_targets, create_icd_feature_matrix, filter_icd_by_delivery_date, fllter_by_days_to_delivery, filter_by_days_from_pregnancy_start
from train_test_rf import load_labels, load_X_y, compute_metrics, metrics_to_df, plot_roc, plot_pr



DATE = datetime.now().strftime('%Y-%m-%d')



# PATH
DELIVERY_LABELS_FILE = "/dors/capra_lab/users/abraha1/projects/PTB_phenotyping/data/combined_cohorts_07_12_2018/full_dataset_characterization/est_delivery_date_at_least_one_icd_cpt_ega.tsv"
EGA_FILE="/dors/capra_lab/users/abraha1/projects/PTB_phenotyping/data/combined_cohorts_07_12_2018/full_dataset_characterization/expanded_ega/date_of_concep_from_closest_ega_within_20wks_of_delivery.tsv"


PKL_DELIVERY_LABELS_FILE="/dors/capra_lab/users/abraha1/projects/PTB_phenotyping/data/ptb_predict_machine_learning/feature_matrices/delivery_labels_from_load_labels.pickle"
PREG_LABS_DIR="/dors/capra_lab/users/abraha1/projects/PTB_phenotyping/data/ptb_predict_machine_learning/clinical_labs/davis_labs_survey"


# outputs
root_ = "/dors/capra_lab/users/abraha1/projects/PTB_phenotyping/data/ptb_predict_machine_learning/feature_matrices/clincal_labs/"
FEAT_OUTPUT_DIR = "/dors/capra_lab/users/abraha1/projects/PTB_phenotyping/data/ptb_predict_machine_learning/feature_matrices/clincal_labs/no_twins_since_concep_upto_28wk"

# -----------
# FUNCTIONS
# -----------

def get_preg_start_delivery_date(DELIVERY_LABELS_FILE, EGA_FILE):



    # load delivery labels
    final_labels_df = load_labels(DELIVERY_LABELS_FILE)


    # load EGA data  -- keep only first/earliest delivery
    ega_df = pd.read_csv(EGA_FILE, sep="\t")
    ega_df.delivery_date = pd.to_datetime(ega_df.delivery_date)
    ega_df.sort_values(['GRID','delivery_date'],inplace=True, ascending=True)
    earliest_ega_df = ega_df[~ega_df.duplicated(['GRID'], keep='first')].copy()

    # check that earliest EGA delivery identified refers to teh earliest delivery in the delivery labels dataset


    temp_first_label_df = final_labels_df.copy()
    temp_first_label_df['GRID_DDATE'] = temp_first_label_df.GRID +"_"+temp_first_label_df.delivery_date

    temp_early_ega_df = earliest_ega_df.copy()
    temp_early_ega_df['GRID_DDATE'] = temp_early_ega_df.GRID +"_"+ temp_early_ega_df.delivery_date.dt.strftime( "%Y-%m-%d")

    # align delivery dates
    keep_early_ega_df = temp_early_ega_df[temp_early_ega_df.GRID_DDATE.isin(temp_first_label_df.GRID_DDATE)].copy()

    delivery_date_dict = dict(zip(keep_early_ega_df.GRID, keep_early_ega_df.delivery_date.astype('str')))
    preg_start_date_dict = dict(zip(keep_early_ega_df.GRID, keep_early_ega_df.date_of_conception.astype('str')))


    return delivery_date_dict, preg_start_date_dict

def get_stat(keep_lab_df, stat):

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

def load_labs_before_28wks(lab_file, grid_dd_preg_start_dict):

    # load lab_file data and update preg_start_date
    temp_lab_df = pd.read_csv(lab_file, sep="\t").drop('preg_start_date', axis=1)

    # update preg_start
    temp_lab_df['grid_delivery_date'] = temp_lab_df['grid'] +"_"+ temp_lab_df['delivery_date']
    temp_lab_df['preg_start_date'] = temp_lab_df['grid_delivery_date'].map(grid_dd_preg_start_dict)

    # calculate days of gestation
    temp_lab_df['preg_start_date'] = pd.to_datetime(temp_lab_df['preg_start_date'])
    temp_lab_df['lab_date'] = pd.to_datetime(temp_lab_df['lab_date'])
    temp_lab_df['days_of_gestation'] =  temp_lab_df.lab_date - temp_lab_df['preg_start_date']
    temp_lab_df = temp_lab_df[temp_lab_df['days_of_gestation'] > np.timedelta64(0, 'D')].copy() # keep only psoitive gestational days
    final_lab_df  = temp_lab_df[temp_lab_df['days_of_gestation'] < np.timedelta64(28*7, 'D')].copy() # keep only labs collected before 28 weeks
    return final_lab_df


def create_feat_mat(lab_files,  grid_dd_preg_start_dict, stat):
    # will only keep labs during the first pregancny and BEFORE 28 weeks of gestation.

    temp_lab_df = load_labs_before_28wks(lab_files[0], grid_dd_preg_start_dict)


    # operate on first lab
    gb_df, short_name, long_name= get_stat(temp_lab_df, stat)
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
            lab_df = load_labs_before_28wks(lab_file, grid_dd_preg_start_dict)
            gb_df, short_name, long_name= get_stat(lab_df,  stat)

            # if no labs for the first pregnancy...
            if gb_df.shape[0] == 0:
                continue

            gb_df.rename(columns={'lab_value':short_name}, inplace=True)

        except pd.errors.EmptyDataError:
            # print("EmptyDataError")
            continue

        stat_labs_df = pd.merge(gb_df, stat_labs_df, on='grid_delivery_date',how='outer')


    stat_labs_df['GRID'] = stat_labs_df.grid_delivery_date.apply(lambda x: x.split("_")[0])
    stat_labs_df.drop(['grid_delivery_date'], axis=1, inplace=True)

    cols = list(stat_labs_df.columns)
    cols.remove('GRID')
    stat_labs_df= stat_labs_df.loc[:, ['GRID'] + cols]


    return stat_labs_df



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


    # lock delivery date and preganncy start dates
    # look only at the first delivery
    delivery_date_dict, preg_start_date_dict = get_preg_start_delivery_date(DELIVERY_LABELS_FILE, EGA_FILE)
    grid, dd = list(zip(*[ (grid,dd) for grid, dd in delivery_date_dict.items()]))
    grid_p, ps = list(zip(*[ (grid,dd) for grid, dd in preg_start_date_dict.items()]))
    dd_df = pd.DataFrame({'GRID':grid, 'delivery_date': dd})
    ps_df = pd.DataFrame({'GRID':grid, 'preg_start': ps})
    merge_df = pd.merge(dd_df, ps_df, on='GRID', how='inner')
    merge_df['GRID_delivery_date'] = merge_df['GRID'] +"_"+ merge_df['delivery_date']
    grid_dd_preg_start_dict = dict(zip(merge_df.GRID_delivery_date, merge_df.preg_start))


    ###
    #   TAKE SUMMARY STATS OF CLINCIAL LABS
    ###
    lab_files = glob.glob(PREG_LABS_DIR +"/*.tsv")


    # -----------  One feat matrix with all of summary stats  -----------


    all_stats_df = pd.DataFrame()
    for counter, this_stat in enumerate(['mean','median','min','max']):

        stat_df = create_feat_mat(lab_files, grid_dd_preg_start_dict, this_stat)
        # append stat to column names
        stat_df.columns = ['{}_{}'.format(x, this_stat) if x != 'GRID' else 'GRID' for x in stat_df.columns]

        if counter == 0:
            # initialize
            all_stats_df = all_stats_df.append(stat_df)
        else:
            all_stats_df = pd.merge(all_stats_df, stat_df, on='GRID', how='outer')

    # # for binary matricies, we will only have one column per clinical labs since for example mean and median binary columns will be the same
    # keep_one_stat_cols = [x for x in all_stats_df.columns if ((x == 'GRID') | ('mean' in x)) ]
    # all_binary_df = convert_to_binary(all_stats_df.loc[:, keep_one_stat_cols])


    all_stats_df.to_csv(os.path.join(FEAT_OUTPUT_DIR, 'first_preg_all_stats_labs_feat_mat.tsv'), sep="\t", index=False, na_rep="NaN")
    # all_binary_df.to_csv(os.path.join(BINARY_FEAT_OUTPUT_DIR, 'binary_present_labs_first_preg_all_stats_feat_mat.tsv'), sep="\t", index=False)


