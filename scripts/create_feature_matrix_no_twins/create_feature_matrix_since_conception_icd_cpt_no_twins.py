#!/bin/python
# This script will...
#       * use icd-9 and cpt codes to create a feature matrix that includes all codes up to X days after the start of pregnancy
#       * outputs a featrure matrix to the OUTPUT_DIR
# Abin Abraham


# created on: 2019-02-04 07:31:03



import os
import sys
import pickle
import numpy as np
import pandas as pd
import importlib
from datetime import datetime
sys.path.append('/dors/capra_lab/users/abraha1/projects/PTB_phenotyping/scripts/rand_forest_ptb_classification')
from rand_forest_helper_functions import label_targets, create_icd_feature_matrix, filter_icd_by_delivery_date, fllter_by_days_to_delivery, filter_by_days_from_pregnancy_start
from train_test_rf import load_labels, load_X_y, compute_metrics, metrics_to_df, plot_roc, plot_pr

sys.path.append('/dors/capra_lab/users/abraha1/projects/PTB_phenotyping/scripts/rand_forest_ptb_classification/to_dmatrix')
from create_dmatrix_func import convert_to_xgbDataset

sys.path.append('/dors/capra_lab/users/abraha1/projects/PTB_phenotyping/scripts/rand_forest_ptb_classification/create_feat_matrix/no_twins_datasets')
from helper_func import ascertainment_and_twin_codes_to_exclude, get_earliest_preg_start, get_mult_gest_and_twin_codes, keep_only_singletons

sys.path.append('/dors/capra_lab/users/abraha1/projects/PTB_phenotyping/scripts/rand_forest_ptb_classification/to_dmatrix')
from create_dmatrix_func import convert_to_dmatrix

DATE = datetime.now().strftime('%Y-%m-%d')


# -----------
# PATHS
# -----------

# output files


DELIVERY_LABELS_FILE = "/dors/capra_lab/users/abraha1/projects/PTB_phenotyping/data/combined_cohorts_07_12_2018/full_dataset_characterization/est_delivery_date_at_least_one_icd_cpt_ega.tsv"
EGA_FILE="/dors/capra_lab/users/abraha1/projects/PTB_phenotyping/data/combined_cohorts_07_12_2018/full_dataset_characterization/expanded_ega/date_of_concep_from_closest_ega_within_20wks_of_delivery.tsv"
# EGA_FILE = "/dors/capra_lab/users/abraha1/prelim_studies/crp_gest_age_assoc/data/EGA_w-in_3days_of_delivery.tsv"


ICD9_FILE ="/dors/capra_lab/users/abraha1/projects/PTB_phenotyping/data/combined_cohorts_07_12_2018/full_dataset/full_ICD9_cohort.tsv"
CPT_FILE= "/dors/capra_lab/users/abraha1/projects/PTB_phenotyping/data/combined_cohorts_07_12_2018/full_cpt_codes/full_CPT_cohort.tsv"

# output file names
OUTPUT_DIR="/dors/capra_lab/users/abraha1/projects/PTB_phenotyping/data/ptb_predict_machine_learning/feature_matrices/no_twins/time_since_preg_start_icd_cpt_no_twins_timeseries_v1"
feat_file = os.path.join(OUTPUT_DIR, 'up_to_{}_since_preg_start_icd9_cpt_no_twins_count_feat_mat.tsv')
dtrain_file = os.path.join(OUTPUT_DIR,  'up_to_{}_since_preg_start_icd9_cpt_no_twins_count_dtrain.dmatrix')
deval_file = os.path.join(OUTPUT_DIR,  'up_to_{}_since_preg_start_icd9_cpt_no_twins_count_deval.dmatrix')
dtrain_eval_file = os.path.join(OUTPUT_DIR,  'up_to_{}_since_preg_start_icd9_cpt_no_twins_count_dtrain_deval.dmatrix')
dtest_file = os.path.join(OUTPUT_DIR,  'up_to_{}_since_preg_start_icd9_cpt_no_twins_count_dtest.dmatrix')
annotated_file = os.path.join(OUTPUT_DIR, 'up_to_{}_since_preg_start_icd9_cpt_no_twins_count_annotated.tsv.pickle')
new_col_name_mapping_file = os.path.join(OUTPUT_DIR,  'up_to_{}_since_preg_start_icd9_cpt_no_twins_count_new_col_name_mapping.tsv')


# older paths: note: this ega based cohort uses CPT to find the closet EGA to delivery date
# EGA_FILE = "/dors/capra_lab/users/abraha1/prelim_studies/crp_gest_age_assoc/data/EGA_w-in_3days_of_delivery.tsv"
# OUTPUT_DIR = "/dors/capra_lab/users/abraha1/projects/PTB_phenotyping/data/ptb_predict_machine_learning/feature_matrices/time_since_preg_start_icd_cpt"



# %%
# -----------
# MAIN
# -----------



###
#       LOAD
###



# load delivery labels
final_labels_df = load_labels(DELIVERY_LABELS_FILE)


# load EGA data  -- keep only first/earliest delivery
ega_df = pd.read_csv(EGA_FILE, sep="\t")
delivery_date_dict, preg_start_date_dict = get_earliest_preg_start(final_labels_df, ega_df)


# %% read in icd-9 and cpt codes
icd9_df = pd.read_csv(ICD9_FILE, sep="\t", usecols=['GRID', 'ICD', 'Date'])
cpt_df = pd.read_csv(CPT_FILE, sep="\t", usecols=['GRID', 'CPT', 'Date'])

# remove identical icd9 and cpt codes & concat into one df
uniq_icd9 = set(icd9_df.ICD.unique())
uniq_cpt = set(cpt_df.CPT.unique())
codes_to_remove = uniq_icd9.intersection(uniq_cpt)

# concat icd9 and cpt codes
n_icd9_df = icd9_df[~icd9_df.ICD.isin(codes_to_remove)]
n_cpt_df = cpt_df[~cpt_df.CPT.isin(codes_to_remove)].rename(columns={'CPT':'ICD'})
concat_df = pd.concat([n_icd9_df, n_cpt_df])



# remove patients with twins + mult. gest.
mult_gest_codes = get_mult_gest_and_twin_codes()
singletons_icd_cpt_df = keep_only_singletons(mult_gest_codes, concat_df)

twins_grids_based_on_icd_cpt = set(concat_df.GRID.unique()) -  set(singletons_icd_cpt_df.GRID.unique())
# twin_grids_file="/dors/capra_lab/users/abraha1/projects/PTB_phenotyping/scripts/rand_forest_ptb_classification/create_feat_matrix/no_twins_datasets/twin_grids_based_on_icd_cpt.txt"
# with open(twin_grids_file, 'w') as fw:
#     for grid in twins_grids_based_on_icd_cpt:
#         fw.write(grid+"\n")


# remove codes used for ascertainment and twin/mult.gest codes
exclude_codes = ascertainment_and_twin_codes_to_exclude()
clean_singletons_icd_cpt_df = singletons_icd_cpt_df[~singletons_icd_cpt_df.ICD.isin(exclude_codes)].copy()


# %%
##
#       CREATE time-to-delivery CPT & ICD-9 FEATURE MATRI(X/ICES)
##



# filter long table to include only codes in specified time frame
# timeframes = { '28_weeks':28*7 }
# timeframes = {'0_weeks':0, '13_weeks':13*7, '28_weeks':28*7, '32_weeks':32*7, '35_weeks':35*7, '37_weeks':37*7 }
timeframes = {'0_weeks':0, '13_weeks':13*7, '28_weeks':28*7, '35_weeks':35*7, '37_weeks':35*7 }

keep_feat_mat = {}
for label, days_threshold in timeframes.items():
    print("**{}".format(label))

    dummy_ex_codes = set()

    #   * will only keep women and codes occuring before X gestational weeks ('timeframe')
    #   * this filtering will include an individual so long as they have AT LEAST ONE CODE that occured before X days since conception
    keep_df = filter_by_days_from_pregnancy_start( clean_singletons_icd_cpt_df, delivery_date_dict, preg_start_date_dict, days_from_preg_start=days_threshold)
    feat_mat = create_icd_feature_matrix(keep_df.loc[:,['GRID','ICD','Date']], delivery_date_dict, dummy_ex_codes, timeframe=None)

    keep_feat_mat[label] = feat_mat



# if doing multiple timepoints, make sure you harmonize the cohort to have the same number of individuals ...
# harmonize cohort
#   * for all time points, include women who have had at least one code before the start of conception
#   * note: the counts of mom won't be the exact same since mom who delivered before the end of the time frame will drop out
keep_grid = keep_feat_mat['0_weeks'].GRID.unique()




#
# convert to dmatrix.
#

for label, feat_df in keep_feat_mat.items():
    
    print(label)
    # feat_df_same_cohort_df = feat_df.copy()
    feat_df_same_cohort_df = feat_df[feat_df.GRID.isin(keep_grid)].copy()


    # write feature file
    this_feat_file = feat_file.format(label)
    feat_df_same_cohort_df.to_csv(this_feat_file, sep="\t", index=False)

    # convert to dmatrix
    xgb_train, xgb_eval, xgb_train_eval,  xgb_test, annotated_df, mapped_col_df = convert_to_xgbDataset(this_feat_file, final_labels_df)


    # twins_grids_based_on_icd_cpt
    
    # final_labels_df


    #write
    annotated_df.reset_index(drop=True).to_pickle(annotated_file.format(label))
    mapped_col_df.to_csv(new_col_name_mapping_file.format(label))
    print("num bytes in df = {}".format(annotated_df.memory_usage().sum()))
    xgb_train.save_binary(dtrain_file.format(label))
    xgb_eval.save_binary(deval_file.format(label))
    xgb_train_eval.save_binary(dtrain_eval_file.format(label))
    xgb_test.save_binary(dtest_file.format(label))
