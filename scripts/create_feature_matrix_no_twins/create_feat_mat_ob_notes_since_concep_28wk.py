#!/bin/python
# This script will
#
#
#
# Abin Abraham
# created on: 2020-06-02 00:17:37

import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime

DATE = datetime.now().strftime('%Y-%m-%d')

struc_file="/dors/capra_lab/users/abraha1/projects/PTB_phenotyping/data/ptb_predict_machine_learning/ob_notes_variables/ob_notes_vars_summarized.tsv"

DELIVERY_LABELS_FILE = "/dors/capra_lab/users/abraha1/projects/PTB_phenotyping/data/combined_cohorts_07_12_2018/full_dataset_characterization/est_delivery_date_at_least_one_icd_cpt_ega.tsv"
EGA_FILE="/dors/capra_lab/users/abraha1/projects/PTB_phenotyping/data/combined_cohorts_07_12_2018/full_dataset_characterization/expanded_ega/date_of_concep_from_closest_ega_within_20wks_of_delivery.tsv"
OUTPUT_DIR="/dors/capra_lab/users/abraha1/projects/PTB_phenotyping/data/ptb_predict_machine_learning/ob_notes_variables"


sys.path.append('/dors/capra_lab/users/abraha1/projects/PTB_phenotyping/scripts/rand_forest_ptb_classification')
from rand_forest_helper_functions import label_targets, create_icd_feature_matrix, filter_icd_by_delivery_date, fllter_by_days_to_delivery, filter_by_days_from_pregnancy_start
from train_test_rf import load_labels, load_X_y, compute_metrics, metrics_to_df, plot_roc, plot_pr



# %%
###
###    MAIN
###

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


# GRIDS with delivery label but w/o a corresponding EGA
final_labels_df[~final_labels_df.GRID.isin(keep_early_ega_df.GRID)].shape


delivery_date_dict = dict(zip(keep_early_ega_df.GRID, keep_early_ega_df.delivery_date.astype('str')))
preg_start_date_dict = dict(zip(keep_early_ega_df.GRID, keep_early_ega_df.date_of_conception.astype('str')))

len(preg_start_date_dict)

# %%

df = pd.read_csv(struc_file, sep="\t")


# strip quotation marks
df.loc[df['Values'] == "'No'", 'Values']  = 'No'
df.loc[df['Values'] == "'Yes'", 'Values']  = 'Yes'

# keep only features occuring before 28 weeks of gestation
df.shape
df.head()

df['preg_start'] = df.GRID.map(preg_start_date_dict)
df.DATE = pd.to_datetime(df.DATE)
df.preg_start = pd.to_datetime(df.preg_start)
df['day_since_concep'] = df.DATE - df.preg_start
df['within_28wks'] = (df['day_since_concep'] > np.timedelta64(0,'D')) & (df['day_since_concep'] < np.timedelta64(28*7,'D'))

df.day_since_concep.unique()



df['within_28wks'].sum()
filt_df = df.query("within_28wks == True").copy()
df.shape
filt_df.shape

# pivot
pivot_df = filt_df.pivot(index='GRID', columns='FIELDLABEL', values='Values')
obs_df = np.sum(~pivot_df.isnull(), 0).reset_index()
pts_df = np.sum(~pivot_df.isnull(), 1).reset_index()


# calculate # missing column and row wise ...
feat_df = obs_df.rename(columns={0: 'num_obs'}).sort_values('num_obs', ascending=False)
feat_df['total_obs'] = pivot_df.shape[0]
feat_df['percent_obs'] = (feat_df.num_obs/pivot_df.shape[0]*100).apply(lambda x: round(x,2))

grids_df = pts_df.rename(columns={0: 'num_obs'}).sort_values('num_obs', ascending=False)
grids_df['total_obs'] = pivot_df.shape[1]
grids_df['percent_obs'] = (grids_df.num_obs/pivot_df.shape[1]*100).apply(lambda x: round(x,2))

# take about 85% to get 2734 number of people  --. best split
grids_to_keep = grids_df[grids_df['percent_obs'] > 85].GRID

# conver to a wdie form and write feature
subset_df = pivot_df.loc[pivot_df.index.isin(grids_to_keep),:].copy()
subset_df.reset_index(inplace=True)

subset_df.head()
# write

subset_df
subset_df.to_csv(os.path.join(OUTPUT_DIR, 'ob_notes_struc_since_concep_up_to_28wk.tsv'), index=False, sep="\t", na_rep="nan")

