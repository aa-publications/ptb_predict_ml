#!/bin/python
# This script will ...
#
#
#
# Abin Abraham
# created on:

# coding: utf-8




import os, sys
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import glob

from datetime import datetime
DATE = datetime.now().strftime('%Y-%m-%d')





sys.path.append('/dors/capra_lab/users/abraha1/projects/PTB_phenotyping/scripts/rand_forest_ptb_classification')
from train_test_rf import load_labels






# PATH
DELIVERY_LABELS_FILE = "/dors/capra_lab/users/abraha1/projects/PTB_phenotyping/data/combined_cohorts_07_12_2018/full_dataset_characterization/est_delivery_date_at_least_one_icd_cpt_ega.tsv"
RACE_FILE = "/dors/capra_lab/users/abraha1/projects/PTB_phenotyping/data/ptb_predict_machine_learning/demographics/demographics_matrix.tsv"
AGE_FILE="/dors/capra_lab/users/abraha1/projects/PTB_phenotyping/data/ptb_predict_machine_learning/demographics/years_at_delivery_matrix.tsv"
EGA_FILE = "/dors/capra_lab/users/abraha1/projects/PTB_phenotyping/data/combined_cohorts_07_12_2018/full_dataset_characterization/expanded_ega/date_of_concep_from_closest_ega_within_20wks_of_delivery.tsv"

OUTPUT_DIR="/dors/capra_lab/users/abraha1/projects/PTB_phenotyping/scripts/rand_forest_ptb_classification/manuscript/no_twins_analyses/28wk_ucsf_replication"

FEAT_FILE="/dors/capra_lab/users/abraha1/projects/PTB_phenotyping/results/ptb_predict_machine_learning/2020_05_26_time_since_conception_icd9_10_phecode_shared_ucsf_vu_codes_no_twins/icd9_28_weeks_since_concep_shared_codes_no_twins/feature_importance_icd9_counts_upto_28_weeks_after_conception_shared_codes_no_twins-2020-05-26.tsv"
FEAT_ANOT_FILE="/dors/capra_lab/users/abraha1/projects/PTB_phenotyping/data/ptb_predict_machine_learning/feature_matrices/time_since_preg_start_icd9_10_phe_vu_uscf_shared_codes/feat_matrix_no_twins/icd9_counts_upto_28_weeks_after_conception_annotated.tsv.feather"




###
###    functions
###
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
###
###    main
###


# load the first delivery label
final_labels_df = load_labels(DELIVERY_LABELS_FILE)
ega_df = pd.read_csv(EGA_FILE, sep="\t")
delivery_date_dict, preg_start_date_dict = get_earliest_preg_start(final_labels_df, ega_df)

anno_df = pd.read_feather(FEAT_ANOT_FILE)
anno_df = anno_df.loc[:, ['GRID','label']].copy()
label_dict = dict(zip(anno_df.GRID, anno_df.label))
anno_df.shape
anno_df.query('label == "preterm"').shape
# load age and race df
age_df = pd.read_csv(AGE_FILE, sep="\t")
race_df = pd.read_csv(RACE_FILE, sep="\t")


from scipy import stats
np.random.seed(12345678)


tstat, pval = stats.ttest_ind(keep_age_df.loc[keep_age_df['label']=='preterm', 'years_at_delivery'].values,
                keep_age_df.loc[keep_age_df['label']!='preterm', 'years_at_delivery'].values)
pval
# keep only GRIDS that were in the above group
keep_age_df = age_df[age_df.GRID.isin(anno_df.GRID)].copy()
keep_age_df['label'] = keep_age_df.GRID.map(label_dict)

keep_age_df.loc[keep_age_df['label']=='preterm', 'years_at_delivery'].mean()
keep_age_df.loc[keep_age_df['label']=='preterm', 'years_at_delivery'].std()
keep_age_df.loc[keep_age_df['label']=='preterm', 'years_at_delivery'].shape

keep_age_df.loc[keep_age_df['label']!='preterm', 'years_at_delivery'].mean()
keep_age_df.loc[keep_age_df['label']!='preterm', 'years_at_delivery'].std()
keep_age_df.loc[keep_age_df['label']!='preterm', 'years_at_delivery'].shape



# race
race_df=pd.read_csv(RACE_FILE, sep="\t")
keep_race_df = race_df[race_df.GRID.isin(anno_df.GRID)].copy()
keep_race_df['label'] = keep_race_df.GRID.map(label_dict)

# preterm
ptb_race_df = keep_race_df.loc[keep_race_df['label']=='preterm'].copy()
no_ptb_race_df = keep_race_df.loc[keep_race_df['label']!='preterm'].copy()

ptb_race_df.set_index('GRID', inplace=True, drop=True)
ptb_race_df.drop('label', axis=1, inplace=True)
no_ptb_race_df.set_index('GRID', inplace=True, drop=True)
no_ptb_race_df.drop('label', axis=1, inplace=True)


no_ptb_race_df


rename_races = { 'RACE_LIST_AFRICAN_AMERICAN':'African American',
                 'RACE_LIST_ASIAN':'Asian',
                 'RACE_LIST_CAUCASIAN': "White",
                 'RACE_LIST_HISPANIC': "Hispanic",
                 'RACE_LIST_NATIVE_AMERICAN':"Native American",
                 'RACE_LIST_OTHER':"Other"}

no_ptb_race_df.sum().reset_index()
ptb_race_df.sum().reset_index()


merged_df = pd.merge(no_ptb_race_df.sum().reset_index(), ptb_race_df.sum().reset_index(), on='index', suffixes=("_no_ptb", '_ptb'))
merged_df



chi2, p, dof, ex = stats.chi2_contingency([merged_df.iloc[:, 1].values, merged_df.iloc[:, 2].values])
p
dof
