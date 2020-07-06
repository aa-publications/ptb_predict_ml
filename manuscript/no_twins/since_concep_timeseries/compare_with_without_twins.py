#!/bin/python
# This script will ...
#
#
#
# Abin Abraham
# created on: 2020-03-27 13:09:56


import os, sys
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pickle

from glob import glob

from scipy.interpolate import interp1d
from datetime import datetime

sys.path.append('/dors/capra_lab/users/abraha1/projects/PTB_phenotyping/scripts/rand_forest_ptb_classification')
from hyperparam_tune import validate_best_model

sys.path.append("/dors/capra_lab/users/abraha1/projects/PTB_phenotyping/scripts/rand_forest_ptb_classification/manuscript/0_helper_func")
from manip_trained_models_funcs import unpack_input_data, get_preds

import matplotlib.font_manager as fm
fpath='/dors/capra_lab/users/abraha1/conda/envs/py36_r_ml/lib/python3.6/site-packages/matplotlib/mpl-data/fonts/ttf/Arial.ttf'
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
%matplotlib inline

DATE = datetime.now().strftime('%Y-%m-%d')

###
###    PATHS
###

TWINS_FILE="/dors/capra_lab/users/abraha1/projects/PTB_phenotyping/scripts/rand_forest_ptb_classification/create_feat_matrix/no_twins_datasets/twin_grids_based_on_icd_cpt.txt"

# results dir
results_dir="/dors/capra_lab/users/abraha1/projects/PTB_phenotyping/results/ptb_predict_machine_learning"
no_twins_dir=os.path.join(results_dir, '2020_06_14_since_conception_icd9_cpt_no_twins_timeseries_v1/')
twins_dir=os.path.join(results_dir, '2020_06_14_since_conception_icd9_cpt_no_twins_timeseries')
bad_ega_dir=os.path.join(results_dir, '2019_05_20_time_since_conception')

# feature matrix dir
feature_dir="/dors/capra_lab/users/abraha1/projects/PTB_phenotyping/data/ptb_predict_machine_learning/feature_matrices"
bad_ega_annotated_files=os.path.join(feature_dir, "time_since_preg_start_icd_cpt", "up_to_{}_*.feather") #'0_weeks'
no_twins_annotated_files=os.path.join(feature_dir, "no_twins/time_since_preg_start_icd_cpt_no_twins_timeseries_v1", "up_to_{}_*.pickle") #'0_weeks'
twins_annotated_files=os.path.join(feature_dir, "no_twins/time_since_preg_start_icd_cpt_no_twins_timeseries_v0", "up_to_{}_*.pickle") #'0_weeks'

OUPUT_DIR="/dors/capra_lab/users/abraha1/projects/PTB_phenotyping/scripts/rand_forest_ptb_classification/manuscript/no_twins_analyses/since_concep_timeseries"


# label files used
og_ega_file =  "/dors/capra_lab/users/abraha1/prelim_studies/crp_gest_age_assoc/data/EGA_w-in_3days_of_delivery.tsv"
new_ega_file ="/dors/capra_lab/users/abraha1/projects/PTB_phenotyping/data/combined_cohorts_07_12_2018/full_dataset_characterization/updated_ega_2020_06_16/earliest_delivery_with_date_of_conception_w_ega_updated_EGA_2020_06_17.tsv"



###
### function
###
def get_delivery_label(x):

    if (x.label == 'preterm')  & (x.twin_bool == True):
        return 'preterm_twins'
    elif (x.label == 'preterm')  & (x.twin_bool == False):
        return 'preterm_no_twins'
    elif (x.label != 'preterm')  & (x.twin_bool == True):
        return 'not_preterm_twins'
    elif (x.label != 'preterm')  & (x.twin_bool == False):
        return 'not_preterm_no_twins'

    else:
        print("didn't meet criteria")
        return np.nan




# %%
###
###    MAIN
###

og_ega_df = pd.read_csv(og_ega_file, sep="\t")
new_ega_df = pd.read_csv(new_ega_file, sep="\t")

og_ega_df.shape
new_ega_df.shape

og_ega_df.GRID.nunique()
new_ega_df.GRID.nunique()

og_ega_df.head()
new_ega_df.head()

# check og_ega_df




# load inputs GRIDS w/ labels

timeseries = ['0_weeks', '13_weeks','28_weeks', '35_weeks', '37_weeks']
label_filt = lambda df: df.loc[df['partition']=='held_out', ['GRID', 'label']].copy()
label_df_dicts = dict()
for timepoint in timeseries:
    print("timepoint")
    bad_ega_feat_file = glob(bad_ega_annotated_files.format(timepoint))[0]
    bad_df = label_filt(pd.read_feather(bad_ega_feat_file))


    no_twins_feat_file = glob(no_twins_annotated_files.format(timepoint))[0]
    no_twins_df = label_filt(pd.read_pickle(no_twins_feat_file))

    twins_feat_file = glob(twins_annotated_files.format(timepoint))[0]
    twins_df = label_filt(pd.read_pickle(twins_feat_file))
    print("Done with twin dataset")

    label_df_dicts[timepoint] = {'bad_df': bad_df,'no_twins_df': no_twins_df,'twins_df': twins_df}






# load twins
with open(TWINS_FILE, 'r') as fo:
    raw_twin_grids = fo.readlines()
twin_grids = [twin_grid.splitlines()[0] for twin_grid in raw_twin_grids]


all_counts_df = pd.DataFrame()
for this_timepoint in timeseries:
    for this_dataset, this_key in zip(['bad_ega_w_twins', 'no_twins', 'twins'], ['bad_df', 'no_twins_df', 'twins_df'] ):

        this_df  = label_df_dicts[this_timepoint][this_key].copy()
        this_df['twin_bool'] = False
        this_df.loc[this_df.GRID.isin(twin_grids), 'twin_bool']= True

        this_df['delivery_label'] = this_df.apply(lambda x: get_delivery_label(x),axis=1)


        count_df = pd.value_counts(this_df.delivery_label).reset_index()
        count_df['timepoint'] = this_timepoint
        count_df['dataset'] = this_dataset
        all_counts_df = all_counts_df.append(count_df)



###
###    analyze
###
all_wide_df = pd.DataFrame()
for this_dataset in ['bad_ega_w_twins', 'no_twins', 'twins']:
    plot_df = all_counts_df.loc[all_counts_df['dataset']==this_dataset].copy()
    wide_df = plot_df.pivot(index='index', columns='timepoint', values='delivery_label')

    wide_df['dataset'] = this_dataset
    all_wide_df = all_wide_df.append(wide_df)


# write
all_wide_df.to_csv(os.path.join(OUPUT_DIR, 'wide_counts_held_out.tsv'), sep="\t", index=True)
# %%
wide_df
plot_df = all_counts_df.loc[all_counts_df['dataset']=='bad_ega_w_twins'].copy()
wide_df = plot_df.pivot(index='index', columns='timepoint', values='delivery_label')
wide_df.index.name=''
wide_df


# Values of each group
not_preterm_no_twins = wide_df.values[0, :]
not_preterm_twins = wide_df.values[1, :]
preterm_no_twins = wide_df.values[2, :]
preterm_twins = wide_df.values[3, :]


# The position of the bars on the x-axis
r = [0,1,2,3,4]

# Names of group and bar width
names = ['A','B','C','D','E']
barWidth = 1

# Create brown bars
plt.bar(r, not_preterm_no_twins, color='#7f6d5f', edgecolor='white', width=barWidth)
# Create green bars (middle), on top of the firs ones
plt.bar(r, not_preterm_twins, bottom=not_preterm_no_twins, color='#557f2d', edgecolor='white', width=barWidth)
# Create green bars (top)
plt.bar(r, preterm_no_twins, bottom= np.add(not_preterm_no_twins,not_preterm_twins), color='#2d7f5e', edgecolor='white', width=barWidth)
# Create green bars (top)
plt.bar(r, preterm_twins, bottom= np.add(not_preterm_no_twins,not_preterm_twins,preterm_no_twins), color='black', edgecolor='white', width=barWidth)
