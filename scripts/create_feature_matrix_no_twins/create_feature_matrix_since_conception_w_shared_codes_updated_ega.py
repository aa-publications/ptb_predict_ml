#!/bin/python
# This script will...
#       * ega to determine the 'start' of pregnancy (conception) and inclue feature up to X days after conception
#       * outputs a featrure matrix to the OUTPUT_DIR
# Abin Abraham


# created on: 2019-02-04 07:31:03




# %%
import os
import sys
from time import time
import pickle
import numpy as np
import pandas as pd
import importlib
from datetime import datetime
sys.path.append('/dors/capra_lab/users/abraha1/projects/PTB_phenotyping/scripts/rand_forest_ptb_classification')
from rand_forest_helper_functions import label_targets, create_icd_feature_matrix, filter_icd_by_delivery_date, fllter_by_days_to_delivery, filter_by_days_from_pregnancy_start
from train_test_rf import load_labels, load_X_y, compute_metrics, metrics_to_df, plot_roc, plot_pr

sys.path.append('/dors/capra_lab/users/abraha1/projects/PTB_phenotyping/scripts/rand_forest_ptb_classification/to_dmatrix')
from create_dmatrix_func import convert_to_dmatrix, convert_to_xgbDataset

sys.path.append('/dors/capra_lab/users/abraha1/projects/PTB_phenotyping/scripts/rand_forest_ptb_classification/create_feat_matrix/no_twins_datasets')
from helper_func import ascertainment_and_twin_codes_to_exclude, get_earliest_preg_start

DATE = datetime.now().strftime('%Y-%m-%d')

# uncomment if doing multiple timepoints
# import argparse
# parser = argparse.ArgumentParser(description='Example with nonoptional arguments')
# parser.add_argument('timepoint', action='store', type=str)
#
# # retrieve passed arguments
# results = parser.parse_args()
# timepoint = results.timepoint



# -----------
# PATHS
# -----------

OUTPUT_FEAT_DIR = "/dors/capra_lab/users/abraha1/projects/PTB_phenotyping/data/ptb_predict_machine_learning/feature_matrices/time_since_preg_start_icd9_10_phe_vu_uscf_shared_codes_no_twins_updated_ega/"
OUTPUT_LONG_DIR = "/dors/capra_lab/users/abraha1/projects/PTB_phenotyping/data/ptb_predict_machine_learning/feature_matrices/time_since_preg_start_icd9_10_phe_vu_uscf_shared_codes_no_twins_updated_ega"

DATA_ROOT = "/dors/capra_lab/users/abraha1/projects/PTB_phenotyping/data/combined_cohorts_07_12_2018/"
ICD9_FILE = os.path.join(DATA_ROOT, "full_dataset/full_ICD9_cohort.tsv")
ICD10_FILE = os.path.join(DATA_ROOT, "full_dataset/full_ICD10_cohort.tsv")


ICD9_TO_PHE_FILE = "/scratch/abraha1/ptb_predict/phecode_icd9_rolled.csv"
ICD10_TO_PHE_FILE = "/scratch/abraha1/ptb_predict/Phecode_map_v1_2_icd10cm_beta.csv"

UCSF_ICD9_FILE="/dors/capra_lab/users/abraha1/projects/PTB_phenotyping/data/ptb_predict_machine_learning/feature_matrices/time_since_preg_start_icd9_10_phe_vu_uscf_shared_codes/icd9_10_since_conception_long_lists_shared_codes/ICD9_codes_UCSF.csv"
UCSF_ICD10_FILE="/dors/capra_lab/users/abraha1/projects/PTB_phenotyping/data/ptb_predict_machine_learning/feature_matrices/time_since_preg_start_icd9_10_phe_vu_uscf_shared_codes/icd9_10_since_conception_long_lists_shared_codes/ICD10_codes_UCSF.csv"

DELIVERY_LABELS_FILE = "/dors/capra_lab/users/abraha1/projects/PTB_phenotyping/data/combined_cohorts_07_12_2018/full_dataset_characterization/est_delivery_date_at_least_one_icd_cpt_ega.tsv"
EGA_FILE = "/dors/capra_lab/users/abraha1/projects/PTB_phenotyping/data/combined_cohorts_07_12_2018/full_dataset_characterization/expanded_ega/date_of_concep_from_closest_ega_within_20wks_of_delivery.tsv"




# -----------
# FUNCTIONS
# -----------


def remove_codes( exclude_codes, df):

    clean_df = df[~df.ICD.isin(exclude_codes)].copy()

    return clean_df


def keep_only_singletons(mult_gest_codes, df):
    # df should be a long dataframe. one row per GRID-ICD-DATE comobo
    # return a df excluding GRIDs with ≥ 1 code indicating multiple gestations.
    mult_gest_grids = df.loc[df.ICD.isin(mult_gest_codes), 'GRID'].unique()
    singletons_df = df.loc[~df.GRID.isin(mult_gest_grids)].copy()
    print(f"Removed {len(mult_gest_grids):,} out of {df.GRID.nunique():,} women due to multiple pregnancies.")

    return singletons_df

def intersect_ucsf_codes(clean_df, ucsf_df):

    n_ucsf = ucsf_df.ICD.nunique()
    n_vu = clean_df.ICD.nunique()
    n_inter = len(set(clean_df.ICD.unique()).intersection(set(ucsf_df.ICD.values)))

    print(f"{n_inter:,} codes kept out of {n_ucsf:,} in UCSF and {n_vu:,} in VU.")

    clean_inter_df = clean_df[clean_df.ICD.isin(ucsf_df.ICD)].copy()

    return clean_inter_df




# %%
# -----------
# MAIN
# -----------


###
###    load
###

# load delivery labels
final_labels_df = load_labels(DELIVERY_LABELS_FILE)
ega_df = pd.read_csv(EGA_FILE, sep="\t")

def get_earliest_preg_start(final_labels_df, ega_df):

    # load EGA data  -- keep only first/earliest delivery
    ega_df.delivery_date = pd.to_datetime(ega_df.delivery_date)
    ega_df.sort_values(['GRID','delivery_date'],inplace=True, ascending=True)
    earliest_ega_df = ega_df[~ega_df.duplicated(['GRID'], keep='first')].copy()

    #
    # keep only ega values for delivery date matching in final_labels_df
    #

    temp_first_label_df = final_labels_df.copy()
    temp_first_label_df['GRID_DDATE'] = temp_first_label_df.GRID +"_"+temp_first_label_df.delivery_date

    temp_early_ega_df = earliest_ega_df.copy()
    temp_early_ega_df['GRID_DDATE'] = temp_early_ega_df.GRID +"_"+ temp_early_ega_df.delivery_date.dt.strftime( "%Y-%m-%d")

    # align delivery dates
    keep_early_ega_df = temp_early_ega_df[temp_early_ega_df.GRID_DDATE.isin(temp_first_label_df.GRID_DDATE)].copy()


    delivery_date_dict = dict(zip(keep_early_ega_df.GRID, keep_early_ega_df.delivery_date.astype('str')))
    preg_start_date_dict = dict(zip(keep_early_ega_df.GRID, keep_early_ega_df.date_of_conception.astype('str')))

    return delivery_date_dict, preg_start_date_dict

#
# load icd9 data
#


# #ucsf codes
ucsf_icd9_df = pd.read_csv(UCSF_ICD9_FILE, sep=",", names=['_', 'ICD', 'freq'])

# load icds
icd9_df = pd.read_csv(ICD9_FILE, sep="\t", usecols=['GRID', 'ICD', 'Date'])

# remove code used for ascertainment...
exclude_codes = ascertainment_and_twin_codes_to_exclude()
clean_all_icd9_df = remove_codes(exclude_codes, icd9_df)

# keep icd9 codes common to both ucsf and vu
clean_icd9_df = intersect_ucsf_codes(clean_all_icd9_df, ucsf_icd9_df)

# get multiple gestation codes
mult_gest_codes = get_mult_gest_codes()



###
###    filter icd codes to include codes before X days after 'CONCEPTION'
###


# filter long table to include only codes in specified time frame
#   this filtering will include an individual so long as they have AT LEAST ONE CODE that occured before X days since conception
timeframes = {'0_weeks':0, '13_weeks':13*7, '28_weeks':28*7, '32_weeks':32*7, '35_weeks':35*7, '37_weeks':37*7 }

# ---
# only RUN 28 weeks
# ---
timepoint='28_weeks'

label=timepoint
days_threshold=timeframes[timepoint]

# for label, days_threshold in timeframes.items():


start = time()
print(">>> Keeping codes occuring before {} after conception.".format(label))
#  * individuals who have already deliveried (i.e. out of the 'X days into pregnancy') are removed
keep_icd9_df = filter_by_days_from_pregnancy_start( clean_icd9_df, delivery_date_dict, preg_start_date_dict, days_from_preg_start=days_threshold)



# remove women with multiple gestations before the devliery date
singletons_icd9_df = keep_only_singletons(mult_gest_codes, keep_icd9_df)


print(f"{singletons_icd9_df.GRID.nunique():,} women remain in ICD9 dataset")
# print(f"{singletons_icd10_df.GRID.nunique():,} women remain in ICD9 dataset")


# write the list with timestamps
singletons_icd9_df.to_csv(os.path.join(OUTPUT_LONG_DIR, 'icd9_upto_{}_after_conception.tsv'.format(label)), sep="\t", index=False)


# convert to feature meatrics
exclude_codes_temp = set()
feat_mat_icd9 = create_icd_feature_matrix(singletons_icd9_df.loc[:,['GRID','ICD','Date']], delivery_date_dict, exclude_codes_temp, timeframe=None)

# write feature matrix
fstart= time()
feat_file=os.path.join(OUTPUT_FEAT_DIR, f'icd9_counts_upto_{label}_after_conception_no_twins_feat_mat.tsv')
feat_mat_icd9.to_csv(feat_file, sep="\t", index=False)

print(f">>> Wrote feature matrices of icd and phecodes. Took { (time()-fstart)/60:.2f}")




# --
#   --
# --

# harmonize cohort
#   for all time points, include women who have had at least one code before the start of conception
#   note: the counts of mom won't be the exact same since mom who delivered before the end of the time frame will drop out
# # keep_grid = keep_feat_mat['0_weeks'].GRID.unique()
#
# # for label, feat_df in keep_feat_mat.items():
#
#         print(label)
#         feat_df_same_cohort_df = feat_df[feat_df.GRID.isin(keep_grid)].copy()
#         feat_df_same_cohort_df.to_csv(os.path.join(OUTPUT_DIR, 'up_to_{}_since_preg_start_icd9_cpt_count_feat_mat.tsv'.format(label)), sep="\t", index=False)
#
#


# convert to dmatrix

# dtrain, dtest, annotated_df, mapped_col_df = convert_to_dmatrix(feat_file, final_labels_df)
xgb_train, xgb_eval, xgb_train_eval,  xgb_test, annotated_df, mapped_col_df = convert_to_xgbDataset(feat_file, final_labels_df)


dtrain_file = os.path.join(OUTPUT_FEAT_DIR,f'icd9_counts_upto_{label}_after_conception_no_twins_dtrain.dmatrix')
deval_file = os.path.join(OUTPUT_FEAT_DIR,f'icd9_counts_upto_{label}_after_conception_no_twins_dteval.dmatrix')
dtrain_eval_file = os.path.join(OUTPUT_FEAT_DIR,f'icd9_counts_upto_{label}_after_conception_no_twins_dtrain_eval.dmatrix')
dtest_file = os.path.join(OUTPUT_FEAT_DIR,f'icd9_counts_upto_{label}_after_conception_no_twins_dtest.dmatrix')
annotated_file = os.path.join(OUTPUT_FEAT_DIR,f'icd9_counts_upto_{label}_after_conception_no_twins_annotated.tsv.pickle')
new_col_name_mapping_file = os.path.join(OUTPUT_FEAT_DIR,f'icd9_counts_upto_{label}_after_conception_no_twins_new_col_name_mapping.tsv')

#write
annotated_df.reset_index(drop=True).to_pickle(annotated_file)
mapped_col_df.to_csv(new_col_name_mapping_file)
print("num bytes in df = {}".format(annotated_df.memory_usage().sum()))
xgb_train.save_binary(dtrain_file)
xgb_eval.save_binary(deval_file)
xgb_train_eval.save_binary(dtrain_eval_file)
xgb_test.save_binary(dtest_file)
