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
import glob
import pickle


from scipy.cluster.hierarchy import linkage,leaves_list
from scipy.interpolate import interp1d
from cycler import cycler
from collections import OrderedDict

from datetime import datetime

sys.path.append('/dors/capra_lab/users/abraha1/projects/PTB_phenotyping/scripts/rand_forest_ptb_classification')
from hyperparam_tune import validate_best_model

sys.path.append("/dors/capra_lab/users/abraha1/projects/PTB_phenotyping/scripts/rand_forest_ptb_classification/manuscript/0_helper_func")
from manip_trained_models_funcs import unpack_input_data, get_preds

sys.path.append('/dors/capra_lab/users/abraha1/projects/PTB_phenotyping/scripts/rand_forest_ptb_classification/manuscript/no_twins_analyses')
from helper_manu_figs import set_up_manu_roc, manu_roc_format




import matplotlib.font_manager as fm
fpath='/dors/capra_lab/users/abraha1/conda/envs/py36_r_ml/lib/python3.6/site-packages/matplotlib/mpl-data/fonts/ttf/Arial.ttf'
bprop = fm.FontProperties(fname=fpath, size=10)
sprop = fm.FontProperties(fname=fpath, size=8)

import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42


###
###    PATHS
###
DATE = datetime.now().strftime('%Y-%m-%d')

BY_TYPE_SINCE_CONCEP_DIR="/dors/capra_lab/users/abraha1/projects/PTB_phenotyping/results/ptb_predict_machine_learning/2020_06_03_since_concep_cs_vg_delivery_no_twins"
CSEC_DIR=os.path.join(BY_TYPE_SINCE_CONCEP_DIR, 'since_concep_28_weeks_no_twins_csection')
VG_DIR=os.path.join(BY_TYPE_SINCE_CONCEP_DIR, 'since_concep_28_weeks_no_twins_vaginal_delivery')

# RF_DIR = "/dors/capra_lab/users/abraha1/projects/PTB_phenotyping/results/ptb_predict_machine_learning/2019_05_20_time_since_conception/up_to_28_weeks_since_preg_start_icd9_cpt_count"
# CLIN_RISK_DIR = "/dors/capra_lab/users/abraha1/projects/PTB_phenotyping/results/ptb_predict_machine_learning/2020_03_24_time_since_concep_up_to_28wks_w_riskfx"
OUTPUT_FIG_DIR = "/dors/capra_lab/users/abraha1/projects/PTB_phenotyping/scripts/rand_forest_ptb_classification/manuscript/no_twins_analyses/28wk_csec_vg_delivery/figures"



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


### load
# define file paths
csec_input_file=os.path.join(CSEC_DIR,'input_data_csection_up_to_28_weeks_since_preg_start_icd9_cpt_no_twins_count-2020-06-04.tsv')
vg_input_file=os.path.join(VG_DIR,'input_data_vaginal_delivery_up_to_28_weeks_since_preg_start_icd9_cpt_no_twins_count-2020-06-04.tsv')

csec_model_file=os.path.join(CSEC_DIR,'best_xgb_model_csection_up_to_28_weeks_since_preg_start_icd9_cpt_no_twins_count-2020-06-04.pickle')
vg_model_file=os.path.join(VG_DIR,'best_xgb_model_vaginal_delivery_up_to_28_weeks_since_preg_start_icd9_cpt_no_twins_count-2020-06-04.pickle')

# load models and input files
_, _, csec_X_test, csec_y_test, csec_xgb_model, csec_input_data = unpack_input_data(csec_input_file, csec_model_file)
_, _, vg_X_test, vg_y_test, vg_xgb_model, vg_input_data = unpack_input_data(vg_input_file, vg_model_file)


csec_metrics_results, csec_metrics_df, _ = validate_best_model(csec_xgb_model, csec_X_test, csec_y_test)
vg_metrics_results, vg_metrics_df, _ = validate_best_model(vg_xgb_model, vg_X_test, vg_y_test)


###
###    plot
###
# %%
# fig paramaters


# %% - ROC


# plot


csec_interp_fpr, csec_interp_tpr, csec_auc = get_auroc_coords(csec_metrics_results)
vg_interp_fpr, vg_interp_tpr, vg_auc = get_auroc_coords(vg_metrics_results)


ax,sprop = set_up_manu_roc(fpath, figsize=(2.25,2.25) )
#
#
_ = ax.plot(csec_interp_fpr, csec_interp_tpr, lw=1.5, alpha=1, linestyle='-',  color='#483D8B', label="C-section ({:.2f})".format(csec_auc))
_ = ax.plot(vg_interp_fpr, vg_interp_tpr, lw=1.5, alpha=1, linestyle='-', color='#00A79D',  label="Vaginal ({:.2f})".format(vg_auc))
_ = ax.plot([0, 1], [0, 1], linestyle='-', lw=0.5, color='indianred', label='Chance', alpha=.8)

ax = manu_roc_format(ax, sprop,  'roc', 'lower right', lg_title='Delivery (ROC-AUC)')

plt.savefig(os.path.join(OUTPUT_FIG_DIR, f'{DATE}_since_concep_28wks_auroc_csec_vg_delivery_no_twins.pdf'), pad_inches=0,  transparent=True)


# %% - PR

# plot


csec_interp_rc, csec_interp_pr, csec_pr_auc, csec_pos_prop = get_pr_coord(csec_metrics_results, csec_y_test)
vg_interp_rc, vg_interp_pr, vg_pr_auc, vg_pos_prop = get_pr_coord(vg_metrics_results, vg_y_test)
vg_interp_pr[0] = vg_interp_pr[1]

ax,sprop = set_up_manu_roc(fpath, figsize=(2.25,2.25) )
_ = ax.plot(csec_interp_rc, csec_interp_pr, lw=1.5, alpha=1, linestyle='-',  color='#483D8B', label="C-section ({:.2f})".format(csec_pr_auc))
_ = ax.plot(vg_interp_rc, vg_interp_pr, lw=1.5, alpha=1, linestyle='-', color='#00A79D',  label="Vaginal ({:.2f})".format(vg_pr_auc))
_ = ax.plot([0, 1], [csec_pos_prop, csec_pos_prop], linestyle='--', lw=0.5, color='#483D8B', label='Chance ({:.2f})'.format(csec_pos_prop), alpha=1)
_ = ax.plot([0, 1], [vg_pos_prop, vg_pos_prop], linestyle='--', lw=0.5, color='#00A79D', label='Chance ({:.2f})'.format(vg_pos_prop), alpha=1)
ax = manu_roc_format(ax, sprop,  'pr', 'upper right', lg_title='Delivery (PR-AUC)')


# plt.subplots_adjust(left=0.15,right=0.9, top=0.9, bottom=0.15)
plt.savefig(os.path.join(OUTPUT_FIG_DIR, f'{DATE}_since_concep_28wks_prroc_csec_vg_delivery_no_twins.pdf'), pad_inches=0,  transparent=True)