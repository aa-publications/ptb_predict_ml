#!/bin/python
# This script will ...
#
#
#
# Abin Abraham
# created on: 2020-04-21 13:50:06


import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime

DATE = datetime.now().strftime('%Y-%m-%d')  


# PATHS
FEAT_DIR = "/dors/capra_lab/users/abraha1/projects/PTB_phenotyping/data/ptb_predict_machine_learning/feature_matrices/time_since_preg_start_icd9_10_phe_vu_uscf_shared_codes"
EGA_FILE="/dors/capra_lab/users/abraha1/projects/PTB_phenotyping/data/combined_cohorts_07_12_2018/full_dataset_characterization/expanded_ega/date_of_conception_w_ega.tsv"


###
### main
###

eg_df = pd.read_csv(EGA_FILE, sep="\t")

ffile=os.path.join(FEAT_DIR,"icd9_counts_upto_28_weeks_after_conception_shared_codes_feat_mat.tsv")
feat_df = pd.read_csv(ffile, sep="\t")

eg_df.head()
feat_df.head()

eg_df.delivery_date = pd.to_datetime(eg_df.delivery_date)
eg_df.closest_ega_DATE = pd.to_datetime(eg_df.closest_ega_DATE)

eg_df['delivery_minus_ega_date']  = eg_df.delivery_date - eg_df.closest_ega_DATE
eg_df['delivery_minus_ega_date_days'] = eg_df['delivery_minus_ega_date'].apply(lambda x: np.abs(x.days))


first_eg_df = eg_df[~eg_df.sort_values(['GRID','delivery_date']).duplicated(keep='first')].copy()

bad_first_eg_df = first_eg_df.loc[first_eg_df['delivery_minus_ega_date_days'] > 280].copy()

bad_first_eg_df

feat_df.shape
feat_df[feat_df.GRID.isin(bad_first_eg_df.GRID)].shape
