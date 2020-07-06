#!/bin/python
# This script will see how much overlap there is between existing chart reivew grid sets and training and test set.
#
#
#
# Abin Abraham
# created on 2019-05-24 08:23:02


import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime

DATE = datetime.now().strftime('%Y-%m-%d')


# PATHS
chart_cohorts_dir = "/dors/capra_lab/users/abraha1/projects/PTB_phenotyping/data/chart_review_cohort/age18-64_union_icd_cpt_06_28_2018"
icd_cpt_input_file="/dors/capra_lab/users/abraha1/projects/PTB_phenotyping/results/ptb_predict_machine_learning/2019-01-16_xgboost_hyperopt_icd_cpt_raw_counts/input_data"

output_dir="/dors/capra_lab/users/abraha1/projects/PTB_phenotyping/scripts/rand_forest_ptb_classification/manuscript/chart_review/"
# %%


# load all grid in random chart reivew set
all_cohorts  = pd.DataFrame()
for cohort_num in np.arange(1,7):


    with open(os.path.join(chart_cohorts_dir, 'random_cohort_{}'.format(cohort_num)), 'r') as fopen:
            all_of_it = fopen.read()

    df = pd.DataFrame({'cohort_num': [cohort_num]*100, 'GRID': [all_of_it.splitlines()][0]})

    all_cohorts = all_cohorts.append(df)


# load grids in random forest training
rf_df = pd.read_csv(os.path.join(icd_cpt_input_file, 'input_data_all_icd9_count_subset-2019-01-25.tsv'), sep="\t", usecols=['GRID'])
rf_df['source'] = 'rf'


rf_df.head()
merged_df = pd.merge(all_cohorts, rf_df, on="GRID", how='inner')


merged_df.shape
merged_df.head()

merged_df.cohort_num.unique()


# write
merged_df.to_csv(os.path.join(output_dir, 'grids_shared_in_icd_cpt_model_and_random_chart_review.tsv'), sep="\t", index=False)
