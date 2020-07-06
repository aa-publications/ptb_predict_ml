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
CSEC_DIR=os.path.join(BY_TYPE_SINCE_CONCEP_DIR, 'csection', 'csec_up_to_28_weeks_since_preg_start_icd9_cpt_count')
VG_DIR=os.path.join(BY_TYPE_SINCE_CONCEP_DIR, 'vaginal', 'vaginal_up_to_28_weeks_since_preg_start_icd9_cpt_count')


OUTPUT_FIG_DIR = "/dors/capra_lab/users/abraha1/projects/PTB_phenotyping/scripts/rand_forest_ptb_classification/manuscript/by_delivery_type/since_concep"


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
cs_input_file=os.path.join(CSEC_DIR,'input_data_csection_up_to_28_weeks_since_preg_start_icd9_cpt_count-2020-04-12.tsv')
vg_input_file=os.path.join(VG_DIR,'input_data_vaginal_delivery_up_to_28_weeks_since_preg_start_icd9_cpt_count-2020-04-12.tsv')

cs_model_file=os.path.join(CSEC_DIR,'best_xgb_model_csection_up_to_28_weeks_since_preg_start_icd9_cpt_count-2020-04-12.pickle')
vg_model_file=os.path.join(VG_DIR,'best_xgb_model_vaginal_delivery_up_to_28_weeks_since_preg_start_icd9_cpt_count-2020-04-12.pickle')

# load models and input files
_, _, cs_X_test, cs_y_test, cs_xgb_model, cs_input_data = unpack_input_data(cs_input_file, cs_model_file)
_, _, vg_X_test, vg_y_test, vg_xgb_model, vg_input_data = unpack_input_data(vg_input_file, vg_model_file)


cs_metrics_results, cs_metrics_df, _ = validate_best_model(cs_xgb_model, cs_X_test, cs_y_test)
vg_metrics_results, vg_metrics_df, _ = validate_best_model(vg_xgb_model, vg_X_test, vg_y_test)


# %%
###
###    plot
###



# plot - PR
mult=1
sns.set( style='ticks', context='paper',  font_scale=1.0, rc={'figure.figsize':(2.2*mult,2.2*mult)} )
sns.set_style( {'axes.grid': True, 'axes.edgecolor': 'k',  'grid.color': '#e1e1e1'})
sprop = fm.FontProperties(fname=fpath, size=8*mult)
fig ,ax = plt.subplots()

cs_interp_rc, cs_interp_pr, cs_pr_auc, cs_pos_prop = get_pr_coord(cs_metrics_results, cs_y_test)
vg_interp_rc, vg_interp_pr, vg_pr_auc, vg_pos_prop = get_pr_coord(vg_metrics_results, vg_y_test)

_ = ax.plot(cs_interp_rc, cs_interp_pr, lw=2, alpha=1, linestyle='-',  color='darkslateblue', label="{:.2f}, C-Section".format(cs_pr_auc))
_ = ax.plot(vg_interp_rc, vg_interp_pr, lw=1, alpha=1, linestyle='--', color='dimgray',  label="{:.2f}, Vaginal Delivery".format(vg_pr_auc))
_ = ax.plot([0, 1], [cs_pos_prop, cs_pos_prop], linestyle='-', lw=1, color='#CD5C5C', label='{:.2f}, Chance'.format(cs_pos_prop), alpha=1)


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

# with legend
# plt.savefig(os.path.join(OUTPUT_FIG_DIR, f'{DATE}_pr_28wks_csection_since_concep_icd_cpt_vu_w_legend.pdf'),  bbox_inches = 'tight', pad_inches=0, bbox_extra_artists=[legend], transparent=True)

# without legend
legend.remove()
plt.subplots_adjust(left=0.15,right=0.9, top=0.9, bottom=0.15)
plt.savefig(os.path.join(OUTPUT_FIG_DIR, f'{DATE}_pr_28wks_csection_since_concep_icd_cpt_vu.pdf'),  bbox_inches = None, pad_inches=0, transparent=True)





# %% - ROC

# plot
fig ,ax = plt.subplots()

cs_interp_fpr, cs_interp_tpr, cs_auc = get_auroc_coords(cs_metrics_results)
vg_interp_fpr, vg_interp_tpr, vg_auc = get_auroc_coords(riskfac_metrics_results)

_ = ax.plot(cs_interp_fpr, cs_interp_tpr, lw=2, alpha=1, linestyle='-',  color='darkslateblue', label="{:.2f}, Billing Codes".format(cs_auc))
_ = ax.plot(vg_interp_fpr, vg_interp_tpr, lw=1, alpha=1, linestyle='--', color='dimgray',  label="{:.2f}, Risk Factors".format(vg_auc))

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

