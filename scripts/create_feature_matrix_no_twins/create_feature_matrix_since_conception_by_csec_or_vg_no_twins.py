#!/bin/python
# This script will...
#       * use icd-9 and cpt codes to create a feature matrix that includes all codes up to X days after the start of pregnancy
#       * outputs a featrure matrix to the OUTPUT_DIR
# Abin Abraham


# created on: 2019-02-04 07:31:03


# %%
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


CSECTION_ICD9_CODES =['669.7', '669.70', '669.71', '763.4', '74.0', '74.1', '74.2', '74.4', '74.9','74.99']
CSECTION_CPT = ['59510', '59514', '59515', '59618', '59620', '59622']
CSECTION_CODES =CSECTION_ICD9_CODES + CSECTION_CPT

VAGINAL_CPT =['59409', '59410', '59610', '59612', '59614']


# %% PATHS

DELIVERY_LABELS_FILE = "/dors/capra_lab/users/abraha1/projects/PTB_phenotyping/data/combined_cohorts_07_12_2018/full_dataset_characterization/est_delivery_date_at_least_one_icd_cpt_ega.tsv"
EGA_FILE="/dors/capra_lab/users/abraha1/projects/PTB_phenotyping/data/combined_cohorts_07_12_2018/full_dataset_characterization/expanded_ega/date_of_concep_from_closest_ega_within_20wks_of_delivery.tsv"

ICD9_FILE ="/dors/capra_lab/users/abraha1/projects/PTB_phenotyping/data/combined_cohorts_07_12_2018/full_dataset/full_ICD9_cohort.tsv"
CPT_FILE= "/dors/capra_lab/users/abraha1/projects/PTB_phenotyping/data/combined_cohorts_07_12_2018/full_cpt_codes/full_CPT_cohort.tsv"

OUTPUT_DIR="/dors/capra_lab/users/abraha1/projects/PTB_phenotyping/data/ptb_predict_machine_learning/feature_matrices/no_twins/time_since_preg_start_icd_cpt_csec_vaginal_no_twins"

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



# read in icd-9 and cpt codes
icd9_df = pd.read_csv(ICD9_FILE, sep="\t", usecols=['GRID', 'ICD', 'Date'])
cpt_df = pd.read_csv(CPT_FILE, sep="\t", usecols=['GRID', 'CPT', 'Date'])

# remove identical icd9 and cpt codes & concat into one df
uniq_icd9 = set(icd9_df.ICD.unique())
uniq_cpt = set(cpt_df.CPT.unique())
codes_to_remove = uniq_icd9.intersection(uniq_cpt)

n_icd9_df = icd9_df[~icd9_df.ICD.isin(codes_to_remove)]
n_cpt_df = cpt_df[~cpt_df.CPT.isin(codes_to_remove)].rename(columns={'CPT':'ICD'})
concat_df = pd.concat([n_icd9_df, n_cpt_df])



# remove patients with twins + mult. gest.
mult_gest_codes = get_mult_gest_and_twin_codes()
singletons_icd_cpt_df = keep_only_singletons(mult_gest_codes, concat_df)



filtered_concat_df = singletons_icd_cpt_df[singletons_icd_cpt_df.GRID.isin(final_labels_df.GRID)].copy()
filtered_concat_df.Date = pd.to_datetime(filtered_concat_df.Date)

filtered_concat_df['delivery_date'] = filtered_concat_df.GRID.map(delivery_date_dict)
filtered_concat_df['delivery_date'] = pd.to_datetime(filtered_concat_df['delivery_date'])
filtered_concat_df['since_delivery_abs_days'] = np.abs((filtered_concat_df.Date - filtered_concat_df.delivery_date).dt.days)

###
#   CREATE TWO COHORTs - C/S and VAGINAL
###


# 1)
# keep CPT or ICD codes occuring within 1
within_10_days_df = filtered_concat_df.loc[ filtered_concat_df['since_delivery_abs_days'] < 10 ].copy()
csection_bool = within_10_days_df.ICD.isin(CSECTION_CODES)
vaginal_bool = within_10_days_df.ICD.isin(VAGINAL_CPT)
only_csections_mask = csection_bool & ~vaginal_bool
only_vaginal_mask = vaginal_bool & ~csection_bool

csection_grids = set(within_10_days_df[only_csections_mask].GRID.unique())
vaginal_grids = set(within_10_days_df[only_vaginal_mask].GRID.unique())



# 2) c/s only grids
csection_concat_df = filtered_concat_df[filtered_concat_df.GRID.isin(csection_grids)].copy()

# 3) vaginal only grids
vaginal_concat_df = filtered_concat_df[filtered_concat_df.GRID.isin(vaginal_grids)].copy()



# remove codes used for ascertainment and twin/mult.gest codes
exclude_codes = ascertainment_and_twin_codes_to_exclude()
filt_csection_concat_df = csection_concat_df[~csection_concat_df.ICD.isin(exclude_codes)].copy()
filt_vaginal_concat_df = vaginal_concat_df[~vaginal_concat_df.ICD.isin(exclude_codes)].copy()





# %%
##
#       CREATE time-to-delivery CPT & ICD-9 FEATURE MATRI(X/ICES)
##



# filter long table to include only codes in specified time frame
#   this filtering will include an individual so long as they have AT LEAST ONE CODE that occured before X days since conception
# timeframes = {'0_weeks':0, '13_weeks':13*7, '28_weeks':28*7, '32_weeks':32*7, '35_weeks':35*7, '37_weeks':37*7 }
timeframes = {'28_weeks':28*7}


for label, days_threshold in timeframes.items():
    print("**{}".format(label))
    # here I break up icd-9 codes into before, and within [x] # of days from delivery
    csec_keep_df = filter_by_days_from_pregnancy_start( filt_csection_concat_df, delivery_date_dict, preg_start_date_dict, days_from_preg_start=days_threshold)
    vg_keep_df = filter_by_days_from_pregnancy_start( filt_vaginal_concat_df, delivery_date_dict, preg_start_date_dict, days_from_preg_start=days_threshold)

    dummy_ex_codes = set() # don't need to exclude codes since this is already done
    csec_feat_mat = create_icd_feature_matrix(csec_keep_df.loc[:,['GRID','ICD','Date']],delivery_date_dict, dummy_ex_codes, timeframe=None)
    vg_feat_mat = create_icd_feature_matrix(vg_keep_df.loc[:,['GRID','ICD','Date']],delivery_date_dict, dummy_ex_codes, timeframe=None)


    csec_feat_mat.to_csv(os.path.join(OUTPUT_DIR, 'csection_up_to_{}_since_preg_start_icd9_cpt_count_no_twins_feat_mat.tsv'.format(label)), sep="\t", index=False)
    vg_feat_mat.to_csv(os.path.join(OUTPUT_DIR, 'vaginal_delivery_up_to_{}_since_preg_start_icd9_cpt_count_no_twins_feat_mat.tsv'.format(label)), sep="\t", index=False)





for label, feat_df in zip(['csection','vaginal_delivery'], [csec_feat_mat, vg_feat_mat]):

    print(label)
    timepoint='28_weeks'
    this_feat_file = os.path.join(OUTPUT_DIR, f"{label}_up_to_{timepoint}_since_preg_start_icd9_cpt_count_no_twins_feat_mat.tsv")


    # convert to dmatrix
    xgb_train, xgb_eval, xgb_train_eval,  xgb_test, annotated_df, mapped_col_df = convert_to_xgbDataset(this_feat_file, final_labels_df)


    dtrain_file = os.path.join(OUTPUT_DIR,  f'{label}_up_to_{timepoint}_since_preg_start_icd9_cpt_no_twins_count_dtrain.dmatrix')
    deval_file = os.path.join(OUTPUT_DIR,  f'{label}_up_to_{timepoint}_since_preg_start_icd9_cpt_no_twins_count_deval.dmatrix')
    dtrain_eval_file = os.path.join(OUTPUT_DIR,  f'{label}_up_to_{timepoint}_since_preg_start_icd9_cpt_no_twins_count_dtrain_deval.dmatrix')
    dtest_file = os.path.join(OUTPUT_DIR,  f'{label}_up_to_{timepoint}_since_preg_start_icd9_cpt_no_twins_count_dtest.dmatrix')
    annotated_file = os.path.join(OUTPUT_DIR, f'{label}_up_to_{timepoint}_since_preg_start_icd9_cpt_no_twins_count_annotated.tsv.pickle')
    new_col_name_mapping_file = os.path.join(OUTPUT_DIR,  f'{label}_up_to_{timepoint}_since_preg_start_icd9_cpt_no_twins_count_new_col_name_mapping.tsv')


    #write
    annotated_df.reset_index(drop=True).to_pickle(annotated_file)
    mapped_col_df.to_csv(new_col_name_mapping_file)
    print("num bytes in df = {}".format(annotated_df.memory_usage().sum()))
    xgb_train.save_binary(dtrain_file)
    xgb_eval.save_binary(deval_file)
    xgb_train_eval.save_binary(dtrain_eval_file)
    xgb_test.save_binary(dtest_file)
