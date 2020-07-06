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

from scipy.cluster.hierarchy import linkage,leaves_list
from scipy.interpolate import interp1d
from cycler import cycler
from collections import OrderedDict

from datetime import datetime

sys.path.append('/dors/capra_lab/users/abraha1/projects/PTB_phenotyping/scripts/rand_forest_ptb_classification')
from hyperparam_tune import validate_best_model

sys.path.append("/dors/capra_lab/users/abraha1/projects/PTB_phenotyping/scripts/rand_forest_ptb_classification/manuscript/0_helper_func")
from manip_trained_models_funcs import unpack_input_data, get_preds


sys.path.append("/dors/capra_lab/users/abraha1/projects/PTB_phenotyping/scripts/rand_forest_ptb_classification/manuscript/0_helper_func")
from manip_trained_models_funcs import unpack_input_data, get_preds


import matplotlib.font_manager as fm
fpath='/dors/capra_lab/users/abraha1/conda/envs/py36_r_ml/lib/python3.6/site-packages/matplotlib/mpl-data/fonts/ttf/Arial.ttf'
bprop = fm.FontProperties(fname=fpath, size=10)
sprop = fm.FontProperties(fname=fpath, size=8)

import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

sys.path.append('/dors/capra_lab/users/abraha1/projects/PTB_phenotyping/scripts/rand_forest_ptb_classification/manuscript/no_twins_analyses')
from helper_manu_figs import set_up_manu_roc, manu_roc_format

# %matplotlib inline

###
###    PATHS
###
DATE = datetime.now().strftime('%Y-%m-%d')


ROOT_DATA_DIR = "/dors/capra_lab/users/abraha1/projects/PTB_phenotyping/results/ptb_predict_machine_learning/2020_06_05_second_ptb_no_twins"
OUTPUT_DIR =    "/dors/capra_lab/users/abraha1/projects/PTB_phenotyping/scripts/rand_forest_ptb_classification/manuscript/no_twins_analyses/second_ptb_no_twins/figures"

###
###    FUNCTIONS
###

def get_auroc_coords(metric_results):

    # unpack data
    metrics_results = metric_results
    fpr = metrics_results['fpr']
    tpr = metrics_results['tpr']
    auc = metrics_results['roc_auc']

    # calc
    lin_fx = interp1d(fpr, tpr, kind='linear', assume_sorted=True)
    interp_fpr = np.linspace(0, 1, 100)
    interp_tpr = lin_fx(interp_fpr)

    # force a 0,0 start
    interp_fpr = np.hstack((np.array([0]), interp_fpr))
    interp_tpr = np.hstack((np.array([0]), interp_tpr))

    return interp_fpr, interp_tpr, auc


def get_pr_coord(metrics_results, y_test):

    pr = metrics_results['pr_curve']
    rc = metrics_results['rc_curve']
    pr_auc = metrics_results['avg_prec']
    pos_prop=np.sum(y_test)/len(y_test)


    # calc
    lin_fx = interp1d(rc, pr, kind='linear')
    interp_rc = np.linspace(0, 1, 100)
    interp_pr = lin_fx(interp_rc)

    # force a 1,pos_prop end
    interp_rc = np.hstack((interp_rc, np.array([1])))
    interp_pr = np.hstack((interp_pr, np.array([pos_prop])))

    return interp_rc, interp_pr, pr_auc, pos_prop

# %%
# -----------
# MAIN
# -----------

# load input and model fiel
up_to_dataset_dict = OrderedDict()
for num_weeks in ['10', '30', '60']:

    input_file = glob(os.path.join(ROOT_DATA_DIR,'{}_days_bf_2nd_del_ptbhx_no_twins/input_data_eq_samp_size_raw_counts_icd_cpt_up_to_{}_days_before_second_delivery_no_twins-2020-*.tsv'.format(num_weeks,num_weeks)))[0]
    model_file = glob(os.path.join(ROOT_DATA_DIR,'{}_days_bf_2nd_del_ptbhx_no_twins/best_xgb_model_eq_samp_size_raw_counts_icd_cpt_up_to_{}_days_before_second_delivery_no_twins-2020-*.pickle'.format(num_weeks,num_weeks)))[0]
    _, _, X_test, y_test, xgb_model, input_data = unpack_input_data(input_file, model_file)
    metrics_results, _, _ = validate_best_model(xgb_model, X_test, y_test)
    interp_rc, interp_pr, pr_auc, pos_prop = get_pr_coord(metrics_results, y_test)
    interp_fpr, interp_tpr, auc = get_auroc_coords(metrics_results)


    up_to_dataset_dict['{}_days'.format(num_weeks)] =  {'interp_rc': interp_rc, 'interp_pr': interp_pr, 'pr_auc':pr_auc, 'pos_prop':pos_prop, 'rocauc':auc, 'interp_fpr':interp_fpr, 'interp_tpr':interp_tpr}




# %%
###
###    plot
###



# plot - PR

ax,sprop = set_up_manu_roc(fpath, figsize=(2.25,2.25) )

# colors
# hex_codes = [ '#%02x%02x%02x' % tuple((np.array(x)*255).astype(int)) for x in sns.color_palette("cubehelix", 3)]
hex_codes = ['#77AADD','#44BB99','#AAAA00','#EE8866','#FFAABB']
for index, timepoint in enumerate(up_to_dataset_dict.keys()):
    rc_coords = up_to_dataset_dict[timepoint]['interp_rc']
    pr_coords = up_to_dataset_dict[timepoint]['interp_pr']
    pr_auc = up_to_dataset_dict[timepoint]['pr_auc']
    pos_prop = up_to_dataset_dict[timepoint]['pos_prop']

    if pr_coords[0]==0: 
        pr_coords[0] = pr_coords[1]
    _ = ax.plot(rc_coords, pr_coords, lw=1.5, alpha=1, linestyle='-',  color=hex_codes[index], label="{} ({:.2f})".format(timepoint.replace('_', ' '), pr_auc))




_ = ax.plot([0, 1], [pos_prop, pos_prop], linestyle='-', lw=0.5,   color='#CD5C5C', label='Chance ({:.2f})'.format(pos_prop), alpha=1)

ax = manu_roc_format(ax, sprop,  'pr', 'lower right', lg_title='Days Before Delviery (PR-AUC)')
# plt.savefig(os.path.join(OUTPUT_DIR, f'{DATE}_pr_days_before_second_delivery_no_twins.pdf'), pad_inches=0,  transparent=True)


# %%
# plot ROC

ax,sprop = set_up_manu_roc(fpath, figsize=(2.25,2.25) )

up_to_dataset_dict['10_days'].keys()
# colors
# hex_codes = [ '#%02x%02x%02x' % tuple((np.array(x)*255).astype(int)) for x in sns.color_palette("cubehelix", 3)]
hex_codes =['#228833', '#4477AA', '#CCBB44']

for index, timepoint in enumerate(up_to_dataset_dict.keys()):
    interp_fpr = up_to_dataset_dict[timepoint]['interp_fpr']
    interp_tpr = up_to_dataset_dict[timepoint]['interp_tpr']
    roc_auc = up_to_dataset_dict[timepoint]['rocauc']

    _ = ax.plot(interp_fpr, interp_tpr, lw=1.5, alpha=1, linestyle='-',  color=hex_codes[index], label="{} ({:.2f})".format(timepoint.replace('_', ' '), roc_auc))


_ = ax.plot([0, 1], [0, 1], linestyle='-', lw=0.5, color='indianred', label='Chance', alpha=.8)

ax = manu_roc_format(ax, sprop,  'roc', 'lower right', lg_title='Days Before Delviery (ROC-AUC)')
plt.savefig(os.path.join(OUTPUT_DIR, f'{DATE}_roc_days_before_second_delivery_no_twins.pdf'), pad_inches=0,  transparent=True)

