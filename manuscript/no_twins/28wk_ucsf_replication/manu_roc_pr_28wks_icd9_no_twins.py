#!/bin/python
# This script will ...
#
#
#
# Abin Abraham
# created on: 2020-03-28 22:15:58



import os, sys
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from glob import glob
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
lprop = fm.FontProperties(fname=fpath, size=6)

import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
%matplotlib inline


from datetime import datetime
DATE = datetime.now().strftime('%Y-%m-%d')

###
### paths
###

DIR="/dors/capra_lab/users/abraha1/projects/PTB_phenotyping/scripts/rand_forest_ptb_classification/manuscript/no_twins_analyses/28wk_ucsf_replication/uscf_results"
VU_DIR="/dors/capra_lab/users/abraha1/projects/PTB_phenotyping/results/ptb_predict_machine_learning/2020_05_26_time_since_conception_icd9_10_phecode_shared_ucsf_vu_codes_no_twins/icd9_28_weeks_since_concep_shared_codes_no_twins_updated_ega"
OUTPUT_DIR="/dors/capra_lab/users/abraha1/projects/PTB_phenotyping/scripts/rand_forest_ptb_classification/manuscript/no_twins_analyses/28wk_ucsf_replication/figures"

# timepoints=['28_weeks','32_weeks']
timepoints=['28_weeks']
models=['icd9']


xkcolors = sns.xkcd_palette(["faded green","faded green", "dusty purple","dusty purple", 'black'])
model_cols_dict = dict(zip(models, xkcolors))
model_linestyle_dict = dict(zip(models, ['-', ':', '-', ':', '--']))



uc_dicts = {}
for timepoint in timepoints:

    for model in models:

        this_file = os.path.join(DIR, '{}_{}_metrics_results_2020_06_08.pickle'.format(timepoint, model))
        this_dict = pickle.load( open(this_file , 'rb'))

        uc_dicts['{}_{}'.format(timepoint, model)] = this_dict




vu_dicts = {}
for timepoint in timepoints:
    for model in models:
        print("{} {}".format(timepoint, model))
        model_file = os.path.join(VU_DIR, f"best_xgb_model_{model}_{timepoint}_since_concep_shared_codes_no_twins_updated_ega-2020-06-01.pickle")
        input_file = os.path.join(VU_DIR, f"input_data_{model}_{timepoint}_since_concep_shared_codes_no_twins_updated_ega-2020-06-01.tsv")
        vu_dicts['{}_{}'.format(timepoint, model)] = {'model_file':model_file, 'input_file':input_file}


# %%
###
### functions
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


def get_pr_coord(metrics_results):

    pr = metrics_results['pr_curve']
    rc = metrics_results['rc_curve']
    pr_auc = metrics_results['avg_prec']

    pos = metrics_results['tp'] + metrics_results['fn']
    negs = metrics_results['fp'] + metrics_results['tn']
    pos_prop = pos/(pos+negs)
    # pos_prop=np.sum(y_test)/len(y_test)


    # calc
    lin_fx = interp1d(rc, pr, kind='linear')
    interp_rc = np.linspace(0, 1, 100)
    interp_pr = lin_fx(interp_rc)

    # force a 1,pos_prop end
    interp_rc = np.hstack((interp_rc, np.array([1])))
    interp_pr = np.hstack((interp_pr, np.array([pos_prop])))

    return interp_rc, interp_pr, pr_auc, pos_prop

# %%
###
### main
###

THIS_MODEL ='28_weeks_icd9'

# load vu data
X_train, y_train, X_test, y_test, xgb_model, input_data = unpack_input_data(vu_dicts[THIS_MODEL]['input_file'], vu_dicts[THIS_MODEL]['model_file'])
vu_metrics_results, _, _ = validate_best_model(xgb_model, X_test, y_test)

# uc
uc_metrics = uc_dicts[THIS_MODEL]
interp_fpr, interp_tpr, auc  = get_auroc_coords(uc_metrics)
interp_rc, interp_pr, pr_auc, pos_prop = get_pr_coord(uc_metrics)



# %%

import matplotlib.font_manager as fm
fpath='/dors/capra_lab/users/abraha1/conda/envs/py36_r_ml/lib/python3.6/site-packages/matplotlib/mpl-data/fonts/ttf/Arial.ttf'
prop = fm.FontProperties(fname=fpath, size=11)
sprop = fm.FontProperties(fname=fpath, size=9)

# %%
# -----------
# ROC
# -----------
# fig settings
sns.set( style='whitegrid',  font_scale=1, rc={'figure.figsize':(2.5,2.5)} )
sns.set_style( {'axes.grid': True, 'axes.edgecolor': 'k', 'axes.linewidth': 10,  'grid.color': '#e1e1e1'})


fig, ax = plt.subplots(nrows=1)


_ = ax.plot([0, 1], [0, 1], linestyle='-', lw=1, color='#CD5C5C', label='', alpha=.8)

# plot rocs
interp_fpr, interp_tpr, auc  = get_auroc_coords(vu_metrics_results)
_ = ax.plot(interp_fpr, interp_tpr, lw=2, alpha=1, linestyle="-", color='goldenrod',  label="VU ({:.2f})".format(auc))
interp_fpr, interp_tpr, auc  = get_auroc_coords(uc_metrics)
_ = ax.plot(interp_fpr, interp_tpr, lw=2, alpha=1, linestyle="-", color='royalblue',  label="UCSF ({:.2f})".format(auc))


_ = ax.set_xlabel('False Positive Rate',  fontproperties=sprop)
_ = ax.set_ylabel('True Positive Rate',  fontproperties=sprop)


# lg = ax.legend(loc="lower right", frameon=False, fancybox=False, framealpha=.2, shadow=False,borderpad=0.5, prop=lprop)
_ = lg = ax.legend(prop=lprop, facecolor='white', edgecolor='white', frameon=True, fancybox=False, framealpha=1, shadow=False, borderpad=0.25, loc="best")
lg.set_title('Cohort (ROC-AUC)',prop=lprop)
_ = lg._legend_box.align = "left"

_ = ax.tick_params(axis='both', which='major')
ax.set_xlim(0,1.0)
ax.set_ylim(0,1.0)
ax.set_xticks(np.arange(0,1.2,0.2))
ax.set_yticks(np.arange(0,1.2,0.2))
ax.set_xticklabels(['{:.1f}'.format(x) for x in np.arange(0,1.2,0.2)], fontproperties=sprop)
ax.set_yticklabels(['{:.1f}'.format(x) for x in np.arange(0,1.2,0.2)], fontproperties=sprop)
ax.set_aspect('equal','box')

ax.spines['left'].set_linewidth(0.5)
ax.spines['bottom'].set_linewidth(0.5)
ax.spines['top'].set_linewidth(0.5)
ax.spines['right'].set_linewidth(0.5)


plt.subplots_adjust(left=0.2,right=1, top=0.9, bottom=0.2)
plt.savefig(os.path.join(OUTPUT_DIR, f'{DATE}_auroc_28wks_icd9_vu_ucsf_no_twins.pdf'),  bbox_inches = None, pad_inches=0, transparent=True)


# %%
# -----------
# PR
# -----------
# sns.set( style='whitegrid',  font_scale=1.0, rc={'figure.figsize':(3,3)} )
mult=1
sns.set( style='ticks', context='paper',  font_scale=1.0, rc={'figure.figsize':(2.5*mult,2.5*mult)} )
sns.set_style( {'axes.grid': True, 'axes.edgecolor': 'k', 'axes.linewidth': 10,  'grid.color': '#e1e1e1'})
sprop = fm.FontProperties(fname=fpath, size=8*mult)
fig ,ax = plt.subplots()

# curves
vu_interp_rc, vu_interp_pr, vu_pr_auc, vu_pos_prop  = get_pr_coord(vu_metrics_results)
_ = ax.plot(vu_interp_rc, vu_interp_pr, lw=2, alpha=1, linestyle="-", color='goldenrod',  label="Vanderbilt ({:.2f})".format(vu_pr_auc))
uc_interp_rc, uc_interp_pr, uc_pr_auc, uc_pos_prop  = get_pr_coord(uc_metrics)
_ = ax.plot(uc_interp_rc, uc_interp_pr, lw=2, alpha=1, linestyle="-", color='royalblue',  label="UCSF ({:.2f})".format(uc_pr_auc))

_ = ax.plot([0, 1], [vu_pos_prop, vu_pos_prop], linestyle='--', lw=1, color='goldenrod', label=f'Chance ({vu_pos_prop:.2f})', alpha=0.7)
_ = ax.plot([0, 1], [uc_pos_prop, uc_pos_prop], linestyle='--', lw=1, color='royalblue', label=f'Chance ({uc_pos_prop:.2f})', alpha=0.7)
ax.set_xlim(0,1.0)
ax.set_ylim(0,1.0)

ax.set_xticks(np.arange(0,1.2,0.2))
ax.set_xticklabels(['{:.1f}'.format(x) for x in np.arange(0,1.2,0.2)], fontproperties=sprop)

ax.set_yticks(np.arange(0,1.2,0.2))
ax.set_yticklabels(['{:.1f}'.format(x) for x in np.arange(0,1.2,0.2)],fontproperties=sprop)

_ = ax.set_xlabel('Recall', fontproperties=sprop)
_ = ax.set_ylabel('Precision', fontproperties=sprop)
# lg = ax.legend(loc="best", bbox_to_anchor=(1, 1), prop=lprop, frameon=True, fancybox=False, framealpha=1, shadow=False, borderpad=0.5)

_ = lg = ax.legend(prop=lprop, facecolor='white', edgecolor='white', frameon=True, fancybox=False, framealpha=1, shadow=False, borderpad=0.25, loc="best")

_ = lg.set_title(title='Cohort (PR-AUC)', prop=lprop)
_ = lg._legend_box.align = "left"



ax.spines['left'].set_linewidth(0.5)
ax.spines['bottom'].set_linewidth(0.5)
ax.spines['top'].set_linewidth(0.5)
ax.spines['right'].set_linewidth(0.5)
ax.tick_params(width=0.5, length=2.5)
sns.despine(ax=ax, top=False, right=False)

ax.set_aspect('equal','box')
plt.subplots_adjust(left=0.2,right=1, top=0.9, bottom=0.2)

# ax.get_legend().remove()
plt.savefig(os.path.join(OUTPUT_DIR, f'{DATE}_prroc_28wks_icd9_vu_ucsf_no_twins.pdf'),  bbox_inches = None, pad_inches=0, transparent=True)

# plt.tight_layout()
# plt.savefig(os.path.join(OUTPUT_DIR, f'{DATE}_prroc_28wks_icd9_vu_ucsf_w_legend.pdf'),  bbox_inches = 'tight', pad_inches=0, bbox_extra_artists=[legend], transparent=True)