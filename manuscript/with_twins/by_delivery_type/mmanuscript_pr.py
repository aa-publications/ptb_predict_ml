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

BY_TYPE_SINCE_CONCEP_DIR="/dors/capra_lab/users/abraha1/projects/PTB_phenotyping/results/ptb_predict_machine_learning/2020_04_12_csection_and_vaginal_delivery"
CSEC_DIR=os.path.join(BY_TYPE_SINCE_CONCEP_DIR, 'csection')
VG_DIR=os.path.join(BY_TYPE_SINCE_CONCEP_DIR, 'vaginal')

# RF_DIR = "/dors/capra_lab/users/abraha1/projects/PTB_phenotyping/results/ptb_predict_machine_learning/2019_05_20_time_since_conception/up_to_28_weeks_since_preg_start_icd9_cpt_count"
# CLIN_RISK_DIR = "/dors/capra_lab/users/abraha1/projects/PTB_phenotyping/results/ptb_predict_machine_learning/2020_03_24_time_since_concep_up_to_28wks_w_riskfx"
OUTPUT_FIG_DIR = "/dors/capra_lab/users/abraha1/projects/PTB_phenotyping/scripts/rand_forest_ptb_classification/manuscript/time_since_conception/figures"


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
ehr_input_file=os.path.join(RF_DIR,'input_data_up_to_28_weeks_since_preg_start_icd9_cpt_count-2019-06-19.tsv')
riskfx_input_file=os.path.join(CLIN_RISK_DIR,'input_data_up_to_28_weeks_since_preg_start_risk_fx-2020-03-24.tsv')

ehr_model_file=os.path.join(RF_DIR,'best_xgb_model_up_to_28_weeks_since_preg_start_icd9_cpt_count-2019-06-19.pickle')
riskfx_model_file=os.path.join(CLIN_RISK_DIR,'best_xgb_model_up_to_28_weeks_since_preg_start_risk_fx-2020-03-24.pickle')

# load models and input files
_, _, ehr_X_test, ehr_y_test, ehr_xgb_model, ehr_input_data = unpack_input_data(ehr_input_file, ehr_model_file)
_, _, riskfac_X_test, riskfac_y_test, riskfac_xgb_model, riskfac_input_data = unpack_input_data(riskfx_input_file, riskfx_model_file)


ehr_metrics_results, ehr_metrics_df, _ = validate_best_model(ehr_xgb_model, ehr_X_test, ehr_y_test)
riskfac_metrics_results, riskfac_metrics_df, _ = validate_best_model(riskfac_xgb_model, riskfac_X_test, riskfac_y_test)


###
###    plot
###
# %%
# fig paramaters
sns.set( style='whitegrid',  font_scale=1.5, rc={'figure.figsize':(6,6)} )
sns.set_style( {'axes.grid': True, 'axes.edgecolor': 'k',  'grid.color': '#e1e1e1'})
fsize=14
leg_fsize=14

# %% - ROC


# plot
fig ,ax = plt.subplots()

ehr_interp_fpr, ehr_interp_tpr, ehr_auc = get_auroc_coords(ehr_metrics_results)
risk_interp_fpr, risk_interp_tpr, risk_auc = get_auroc_coords(riskfac_metrics_results)

_ = ax.plot(ehr_interp_fpr, ehr_interp_tpr, lw=2, alpha=1, linestyle='-',  color='darkslateblue', label="{:.2f}, Billing Codes".format(ehr_auc))
_ = ax.plot(risk_interp_fpr, risk_interp_tpr, lw=1, alpha=1, linestyle='--', color='dimgray',  label="{:.2f}, Risk Factors".format(risk_auc))

_ = ax.plot([0, 1], [0, 1], linestyle='-', lw=1, color='#CD5C5C', label='Chance', alpha=.8)
_ = ax.set_xlim([-0.05, 1.05])
_ = ax.set_ylim([-0.05, 1.05])
_ = ax.set_xlabel('False Positive Rate', fontsize=fsize)
_ = ax.set_ylabel('True Positive Rate', fontsize=fsize)
_ = ax.set_title('PTB Prediction at 28 weeks since conception', fontsize=fsize+2, pad=20)
_ = ax.legend(loc="lower right", fontsize=leg_fsize, frameon=True, fancybox=False, framealpha=.9, shadow=False,borderpad=0.5, title="AU-ROC")
_ = ax.tick_params(axis='both', which='major', labelsize=fsize)
_ = ax.axis('equal')

# plt.savefig(os.path.join(OUTPUT_FIG_DIR, f'{DATE}_auroc_28wks_and_riskfx.pdf'))


# %% - PR

# plot
mult=1
sns.set( style='ticks', context='paper',  font_scale=1.0, rc={'figure.figsize':(2.2*mult,2.2*mult)} )
sns.set_style( {'axes.grid': True, 'axes.edgecolor': 'k',  'grid.color': '#e1e1e1'})
sprop = fm.FontProperties(fname=fpath, size=8*mult)
fig ,ax = plt.subplots()

ehr_interp_rc, ehr_interp_pr, ehr_pr_auc, ehr_pos_prop = get_pr_coord(ehr_metrics_results, ehr_y_test)
risk_interp_rc, risk_interp_pr, risk_pr_auc, risk_pos_prop = get_pr_coord(riskfac_metrics_results, riskfac_y_test)

_ = ax.plot(ehr_interp_rc, ehr_interp_pr, lw=2, alpha=1, linestyle='-',  color='darkslateblue', label="{:.2f}, Billing Codes".format(ehr_pr_auc))
_ = ax.plot(risk_interp_rc, risk_interp_pr, lw=1, alpha=1, linestyle='--', color='dimgray',  label="{:.2f}, Risk Factors".format(risk_pr_auc))
_ = ax.plot([0, 1], [ehr_pos_prop, ehr_pos_prop], linestyle='-', lw=1, color='#CD5C5C', label='{:.2f}, Chance'.format(ehr_pos_prop), alpha=1)


# ax.set_xlim(-0.1,1.1)
# ax.set_ylim(-0.1,1.1)
ax.set_xlim(0,1.0)
ax.set_ylim(0,1.0)

ax.set_xticks(np.arange(0,1.2,0.2))
ax.set_xticklabels(['{:.1f}'.format(x) for x in np.arange(0,1.2,0.2)], fontproperties=sprop)

ax.set_yticks(np.arange(0,1.2,0.2))
ax.set_yticklabels(['{:.1f}'.format(x) for x in np.arange(0,1.2,0.2)],fontproperties=sprop)
# ax.axvline(0, color='gray', linewidth=1)
# ax.axhline(0, color='gray', linewidth=1)
_ = ax.set_xlabel('Recall', fontproperties=sprop)
_ = ax.set_ylabel('Precision', fontproperties=sprop)
# _ = ax.set_title('PTB Prediction at 28 weeks since conception', fontproperties=sprop)
legend = ax.legend(loc="best", bbox_to_anchor=(1, 1), prop=sprop, frameon=True, fancybox=False, framealpha=1, shadow=False, borderpad=0.5, title='AU-PR')
# _ = ax.tick_params(axis='both', which='major', labelsize=fsize)
# _ = ax.axis('equal')


ax.spines['left'].set_linewidth(0.5)
ax.spines['bottom'].set_linewidth(0.5)
ax.spines['top'].set_linewidth(0.5)
ax.spines['right'].set_linewidth(0.5)
ax.tick_params(width=0.5, length=2.5)
sns.despine(ax=ax, top=False, right=False)

# plt.subplots_adjust(left=0.15,right=0.9, top=0.9, bottom=0.15)
plt.savefig(os.path.join(OUTPUT_FIG_DIR, f'{DATE}_pr_roc_28wks_and_riskfx_icd_cpt_vu.pdf'),  bbox_inches = 'tight', pad_inches=0, bbox_extra_artists=[legend], transparent=True)