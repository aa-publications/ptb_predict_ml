#!/bin/python
# This script will ...
#
#
#
# Abin Abraham
# created on: 2020-01-22 12:53:55


import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime

DATE = datetime.now().strftime('%Y-%m-%d')

sys.path.append('/dors/capra_lab/users/abraha1/projects/PTB_phenotyping/scripts/rand_forest_ptb_classification')
from rand_forest_helper_functions import label_targets, create_icd_feature_matrix, filter_icd_by_delivery_date, fllter_by_days_to_delivery, filter_by_days_from_pregnancy_start
from train_test_rf import load_labels, load_X_y, compute_metrics, metrics_to_df, plot_roc, plot_pr

sys.path.append('/dors/capra_lab/users/abraha1/projects/PTB_phenotyping/scripts/rand_forest_ptb_classification/create_feat_matrix/no_twins_datasets')
from helper_func import ascertainment_and_twin_codes_to_exclude, get_earliest_preg_start, get_mult_gest_and_twin_codes, keep_only_singletons

# PATHS: 
DELIVERY_LABELS_FILE = "/dors/capra_lab/users/abraha1/projects/PTB_phenotyping/data/combined_cohorts_07_12_2018/full_dataset_characterization/est_delivery_date_at_least_one_icd_cpt_ega.tsv"
TWINS_FILE="/dors/capra_lab/users/abraha1/projects/PTB_phenotyping/scripts/rand_forest_ptb_classification/create_feat_matrix/no_twins_datasets/twin_grids_based_on_icd_cpt.txt"

ICD9_FILE ="/dors/capra_lab/users/abraha1/projects/PTB_phenotyping/data/combined_cohorts_07_12_2018/full_dataset/full_ICD9_cohort.tsv"
CPT_FILE= "/dors/capra_lab/users/abraha1/projects/PTB_phenotyping/data/combined_cohorts_07_12_2018/full_cpt_codes/full_CPT_cohort.tsv"


ICD9_AND_CPT_FILE="/dors/capra_lab/users/abraha1/projects/PTB_phenotyping/results/ptb_predict_machine_learning/2019-01-16_xgboost_hyperopt_icd_cpt_raw_counts/input_data/input_data_all_icd9_cpt_count_subset-2019-01-26.tsv"


# %%
###
### main
###

# load delivery labels
final_labels_df = load_labels(DELIVERY_LABELS_FILE)

final_labels_df.GRID.nunique()
labeled_grids = set(final_labels_df.GRID.unique())
final_labels_df.shape
final_labels_df

with open(TWINS_FILE, 'r') as ffile: 
    content = [x.splitlines()[0] for x in ffile.readlines()]


mult_grids = set(content)        

no_twins_grids_n  = len(set(final_labels_df.GRID.unique()) - mult_grids)


n_twins_rm = final_labels_df.GRID.nunique() - no_twins_grids_n
(n_twins_rm/final_labels_df.GRID.nunique())*100
n_twins_rm


# %%
# load icd and cpt data and restrict to 

#  read in icd-9 and cpt codes
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

concat_df.GRID.unique()
concat_df.GRID.nunique()

icd_cpt_grids = set(concat_df.GRID.unique())
len(labeled_grids.intersection(icd_cpt_grids))



# remove patients with twins + mult. gest.
mult_gest_codes = get_mult_gest_and_twin_codes()
singletons_icd_cpt_df = keep_only_singletons(mult_gest_codes, concat_df)


icd9_and_cpt_df = pd.read_csv(ICD9_AND_CPT_FILE, sep='\t')
icd9cpt_grids = set(icd9_and_cpt_df.GRID.unique())


len(icd9cpt_grids)
len(icd9cpt_grids - mult_grids) 
len(mult_grids)/len(icd9cpt_grids)*100
35281 - 33928