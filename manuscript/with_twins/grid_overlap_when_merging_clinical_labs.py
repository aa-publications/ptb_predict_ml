#!/bin/python
# This script will create feature matrices to compare how adding clinical labs to billing data (icd and cpt) will affect prediction performance.
#
#
#
# Abin Abraham
# created on: 2019-09-04 10:35:28



import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime

import glob
from collections import OrderedDict


sys.path.append('/dors/capra_lab/users/abraha1/projects/PTB_phenotyping/scripts/rand_forest_ptb_classification/clinical_labs')
from clin_labs_create_feat_matrix import create_feat_mat

sys.path.append('/dors/capra_lab/users/abraha1/projects/PTB_phenotyping/scripts/rand_forest_ptb_classification/manuscript/')
from funcs_for_merging_datasets import check_for_col_collisions, create_pairwise_datasets,merge_two_df


DATE = datetime.now().strftime('%Y-%m-%d')



# PATHS
DELIVERY_LABELS_FILE = "/dors/capra_lab/users/abraha1/projects/PTB_phenotyping/data/combined_cohorts_07_12_2018/full_dataset_characterization/est_delivery_date_at_least_one_icd_cpt_ega.tsv"
PREG_LABS_DIR="/dors/capra_lab/users/abraha1/projects/PTB_phenotyping/data/ptb_predict_machine_learning/clinical_labs/davis_labs_survey"

root_="/dors/capra_lab/users/abraha1/projects/PTB_phenotyping/data/ptb_predict_machine_learning/feature_matrices/"

FILES = {'all_cpt_raw_count_feat_mat.tsv':os.path.join(root_,"cpt_codes/all_cpt_raw_count_feat_mat.tsv"),
         'all_icd9_raw_count_feat_mat.tsv':os.path.join(root_,"raw_counts/all_icd9_raw_count_feat_mat.tsv"),
         'clin_labs': os.path.join(root_, 'clincal_labs/w_missing_values/first_preg_all_stats_labs_feat_mat.tsv'),
         'clin_labs_binary': os.path.join(root_, 'clincal_labs/binary_feat_matrix/binary_present_labs_first_preg_all_stats_feat_mat.tsv')}

OUTPUT_DIR="/dors/capra_lab/users/abraha1/projects/PTB_phenotyping/data/ptb_predict_machine_learning/feature_matrices/add_clin_labs"



# %%
# -----------
# FUNCTIONS
# -----------


# %%
# -----------
# MAIN
# -----------

### load icd and cpt datasets
print("loading icd and cpt...")
icd_cpt_df = merge_two_df( pd.read_csv(FILES['all_icd9_raw_count_feat_mat.tsv'], sep="\t"), pd.read_csv(FILES['all_cpt_raw_count_feat_mat.tsv'], sep="\t"))


### create merged datasets
print("creating pairwise feat matrices")

clin_df = pd.read_csv(FILES['clin_labs'], sep="\t")
binary_df = pd.read_csv(FILES['clin_labs_binary'], sep="\t")

vs_clin_labs_icd_cpt, vs_clin_labs_clin_labs, vs_clin_labs_all = create_pairwise_datasets(icd_cpt_df, clin_df, set(icd_cpt_df.GRID.values), set(clin_df.GRID.values))
vs_bin_labs_icd_cpt, vs_bin_labs_clin_labs, vs_bin_labs_all = create_pairwise_datasets(icd_cpt_df,binary_df, set(icd_cpt_df.GRID.values), set(binary_df.GRID.values))



### write
print("writing...")
vs_clin_labs_icd_cpt.to_csv(os.path.join(OUTPUT_DIR, 'vs_clin_labs_icd_cpt_feat_mat.tsv'), sep="\t", index=False)
vs_clin_labs_clin_labs.to_csv(os.path.join(OUTPUT_DIR, 'vs_clin_labs_clin_labs_feat_mat.tsv'), sep="\t", index=False)
vs_clin_labs_all.to_csv(os.path.join(OUTPUT_DIR, 'vs_clin_labs_all_feat_mat.tsv'), sep="\t", index=False)


vs_bin_labs_icd_cpt.to_csv(os.path.join(OUTPUT_DIR, 'vs_bin_labs_icd_cpt_feat_mat.tsv'), sep="\t", index=False)
vs_bin_labs_clin_labs.to_csv(os.path.join(OUTPUT_DIR, 'vs_bin_labs_bin_labs_feat_mat.tsv'), sep="\t", index=False)
vs_bin_labs_all.to_csv(os.path.join(OUTPUT_DIR, 'vs_bin_labs_all_feat_mat.tsv'), sep="\t", index=False)


