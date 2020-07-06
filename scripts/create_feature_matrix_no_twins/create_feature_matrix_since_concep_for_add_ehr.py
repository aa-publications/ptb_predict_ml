#!/bin/python
# This script will...
#       * use icd-9 and cpt codes to create a feature matrix that includes all codes up to X days after the start of pregnancy
#       * outputs a featrure matrix to the OUTPUT_DIR
# Abin Abraham


# create


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


sys.path.append('/dors/capra_lab/users/abraha1/projects/PTB_phenotyping/scripts/rand_forest_ptb_classification/create_feat_matrix/no_twins_datasets')
from helper_func import ascertainment_and_twin_codes_to_exclude, get_earliest_preg_start, get_mult_gest_and_twin_codes, keep_only_singletons



DATE = datetime.now().strftime('%Y-%m-%d')


# -----------
# PATHS
# -----------

# output files


DELIVERY_LABELS_FILE = "/dors/capra_lab/users/abraha1/projects/PTB_phenotyping/data/combined_cohorts_07_12_2018/full_dataset_characterization/est_delivery_date_at_least_one_icd_cpt_ega.tsv"
EGA_FILE="/dors/capra_lab/users/abraha1/projects/PTB_phenotyping/data/combined_cohorts_07_12_2018/full_dataset_characterization/expanded_ega/date_of_concep_from_closest_ega_within_20wks_of_delivery.tsv"

ICD9_FILE ="/dors/capra_lab/users/abraha1/projects/PTB_phenotyping/data/combined_cohorts_07_12_2018/full_dataset/full_ICD9_cohort.tsv"
CPT_FILE= "/dors/capra_lab/users/abraha1/projects/PTB_phenotyping/data/combined_cohorts_07_12_2018/full_cpt_codes/full_CPT_cohort.tsv"

# output file names
OUTPUT_DIR_ICD_CPT="/dors/capra_lab/users/abraha1/projects/PTB_phenotyping/data/ptb_predict_machine_learning/feature_matrices/no_twins/add_ehr_since_preg_start_28wks_v1"


root_path="/dors/capra_lab/users/"
short_hand_files = {'GRID_PRS.tsv':os.path.join(root_path,"abraha1/projects/PTB_phenotyping/scripts/polygenic_risk_scores/zhang_with_covars/GRID_PRS.tsv"),
                    'years_at_delivery_matrix.tsv':"/dors/capra_lab/users/abraha1/projects/PTB_phenotyping/data/ptb_predict_machine_learning/demographics/years_at_delivery_matrix.tsv",
                    'demographics_matrix.tsv':os.path.join(root_path,"abraha1/projects/PTB_phenotyping/data/ptb_predict_machine_learning/demographics/demographics_matrix.tsv"),
                    'unstruct_':"/dors/capra_lab/users/abraha1/projects/PTB_phenotyping/data/ptb_predict_machine_learning/feature_matrices/ob_notes_unstruct/since_concep_upto_28_weeks/filtered_cui_counts_ob_notes_since_concep_up_to_28wk_feat_mat.tsv",
                    'struct':"/dors/capra_lab/users/abraha1/projects/PTB_phenotyping/data/ptb_predict_machine_learning/ob_notes_variables/ob_notes_struc_since_concep_up_to_28wk.tsv",
                    'clin_labs':"/dors/capra_lab/users/abraha1/projects/PTB_phenotyping/data/ptb_predict_machine_learning/feature_matrices/clincal_labs/no_twins_since_concep_upto_28wk/first_preg_all_stats_labs_feat_mat.tsv"}





# -----------
# ICD CODES TO DEFINE DELIVERY TYPES
# -----------

PRETERM_ICD9_CODES = ['644.2', '644.20', '644.21']
PRETERM_ICD10_CODES = ['O60.1', 'O60.10', 'O60.10X0', 'O60.10X1', 'O60.10X2', 'O60.10X3', 'O60.10X4',
                       'O60.10X5', 'O60.10X9', 'O60.12', 'O60.12X0', 'O60.12X1', 'O60.12X2', 'O60.12X3',
                       'O60.12X4', 'O60.12X5', 'O60.12X9', 'O60.13', 'O60.13X0', 'O60.13X1', 'O60.13X2',
                       'O60.13X3', 'O60.13X4', 'O60.13X5', 'O60.13X9', 'O60.14', 'O60.14X0', 'O60.14X1',
                       'O60.14X2', 'O60.14X3', 'O60.14X4', 'O60.14X5', 'O60.14X9']

TERM_ICD9_CODES = ['650', '645.1', '645.10', '645.11', '645.13', '649.8', '649.81', '649.82']
TERM_ICD10_CODES = ['O60.20', 'O60.20X0', 'O60.20X1', 'O60.20X2', 'O60.20X3', 'O60.20X4', 'O60.20X5',
                    'O60.20X9', 'O60.22', 'O60.22X0', 'O60.22X1', 'O60.22X2', 'O60.22X3', 'O60.22X4',
                    'O60.22X5', 'O60.22X9', 'O60.23', 'O60.23X0', 'O60.23X1', 'O60.23X2', 'O60.23X3',
                    'O60.23X4', 'O60.23X5', 'O60.23X9', 'O80', 'O48.0', '650', '645.1', '645.10',
                    '645.11', '645.13', '649.8', '649.81', '649.82']

POSTTERM_ICD9_CODES = ['645.2', '645.20', '645.21', '645.23', '645.00', '645.01', '645.03']
POSTTERM_ICD10_CODES = ['O48.1']

CPT_DELIVERY_CODES = ['59400', '59409', '59410', '59414', '59510', '59514',
                 '59515', '59525', '59610', '59612', '59614', '59618', '59620', '59622']

# ZA3 codes sepereated by gestational age (icd10 codes)
LESS_20WK_CODES = ['Z3A.0', 'Z3A.00', 'Z3A.01', 'Z3A.08', 'Z3A.09', 'Z3A.1', 'Z3A.10', 'Z3A.11',
                   'Z3A.12', 'Z3A.13', 'Z3A.14', 'Z3A.15', 'Z3A.16', 'Z3A.17', 'Z3A.18', 'Z3A.19']
BW_20_37WK_CODES = ['Z3A.2', 'Z3A.20', 'Z3A.21', 'Z3A.22', 'Z3A.23', 'Z3A.24', 'Z3A.25', 'Z3A.26',
                    'Z3A.27', 'Z3A.28', 'Z3A.29', 'Z3A.3', 'Z3A.30', 'Z3A.31', 'Z3A.32', 'Z3A.33',
                    'Z3A.34', 'Z3A.35', 'Z3A.36', 'Z3A.37']
BW_37_42WK_CODES = ['Z3A.38', 'Z3A.39', 'Z3A.4', 'Z3A.40', 'Z3A.41']
BW_42_HIGHER_CODES = ['Z3A.42', 'Z3A.49']

ICD9_MULT_GEST = ['651','651.7','651.70','651.71','651.8','651.81','651.83','651.9','651.91','651.93','652.6','652.60','652.61','652.63','V91','V91.9','V91.90','V91.91','V91.92','V91.99']
MORE_ICD9_TWINS_CODES=["651","651.0","651.00","651.01","651.03","651.1","651.10","651.11","651.13","651.2","651.20","651.21","651.23","651.3","651.30","651.31","651.33","651.4","651.40","651.41","651.43","651.5","651.50","651.51", "651.53"]
COMPLETE_VCODES_TWINS_ICD9 =  ["V91","V91.0","V91.00","V91.01","V91.02","V91.03","V91.09","V91.1","V91.10","V91.11","V91.12","V91.19","V91.2","V91.20","V91.21","V91.22","V91.29","V91.9","V91.90","V91.91","V91.92", "V91.99"]
CPT_TWINS_CODES = ["74713","76802","76810","76812","76814"]
ICD10_MULT_GEST = ['BY4BZZZ','BY4DZZZ','BY4GZZZ','O30.801','O30.802','O30.803','O30.809','O30.811','O30.812','O30.813','O30.819','O30.821','O30.822','O30.823','O30.829','O30.891','O30.892','O30.893','O30.899','O30.91','O30.92','O30.93','O31.BX10','O31.BX11','O31.BX12','O31.BX13','O31.BX14','O31.BX15','O31.BX19','O31.BX20','O31.BX21','O31.BX22','O31.BX23','O31.BX24','O31.BX25','O31.BX29','O31.BX30','O31.BX31','O31.BX32','O31.BX33','O31.BX34','O31.BX35','O31.BX39','O31.BX90','O31.BX91','O31.BX92','O31.BX93','O31.BX94','O31.BX95','O31.BX99']


# added twins
exclude_codes = PRETERM_ICD9_CODES + PRETERM_ICD10_CODES + TERM_ICD9_CODES + TERM_ICD10_CODES + POSTTERM_ICD9_CODES + POSTTERM_ICD10_CODES + LESS_20WK_CODES + BW_20_37WK_CODES + BW_37_42WK_CODES + BW_42_HIGHER_CODES + BW_42_HIGHER_CODES + CPT_DELIVERY_CODES +ICD9_MULT_GEST+ MORE_ICD9_TWINS_CODES+ COMPLETE_VCODES_TWINS_ICD9+ CPT_TWINS_CODES+ ICD10_MULT_GEST




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


# remove codes used for ascertainment and twin/mult.gest codes
exclude_codes = ascertainment_and_twin_codes_to_exclude()
clean_singletons_icd_cpt_df = singletons_icd_cpt_df[~singletons_icd_cpt_df.ICD.isin(exclude_codes)].copy()





# %%
##
#       CREATE time-to-delivery CPT & ICD-9 FEATURE MATRI(X/ICES)
##



# filter long table to include only codes in specified time frame
#   * this filtering will include an individual so long as they have AT LEAST ONE CODE that occured before X days since conception
# timeframes = {'0_weeks':0, '13_weeks':13*7, '28_weeks':28*7, '32_weeks':32*7, '35_weeks':35*7, '37_weeks':37*7 }
timeframes = {'28_weeks':28*7}


keep_feat_mat = {}
for label, days_threshold in timeframes.items():
    print("**{}".format(label))
    # here I break up icd-9 codes into before, and within [x] # of days from delivery
    #   * will only keep women with a pregnancy start date
    keep_df = filter_by_days_from_pregnancy_start( clean_singletons_icd_cpt_df, delivery_date_dict, preg_start_date_dict, days_from_preg_start=days_threshold)


    feat_mat = create_icd_feature_matrix(keep_df.loc[:,['GRID','ICD','Date']],
                                              delivery_date_dict, exclude_codes, timeframe=None)

    keep_feat_mat[label] = feat_mat



feat_mat.to_csv(os.path.join(OUTPUT_DIR_ICD_CPT, 'up_to_28_weeks_since_preg_start_icd9_cpt_no_twins_count_feat_mat_for_add_ehr.tsv'), sep="\t", index=False)