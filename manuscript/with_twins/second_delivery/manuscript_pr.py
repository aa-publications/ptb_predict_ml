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


ROOT_DATA_DIR = "/dors/capra_lab/users/abraha1/projects/PTB_phenotyping/results/ptb_predict_machine_learning/2019_09_06_2nd_ptb_icd_cpt/equal_sample_size"
OUTPUT_DIR =    "/dors/capra_lab/users/abraha1/projects/PTB_phenotyping/scripts/rand_forest_ptb_classification/manuscript/second_delivery"

# load input and model fiel
up_to_dataset_dict = OrderedDict()
for num_weeks in ['0', '90', '365']:

    input_file = os.path.join(ROOT_DATA_DIR,'eq_up_to_{0}d_before_second_delivery/input_data_eq_samp_size_raw_counts_icd_cpt_up_to_{0}_days_before_second_delivery-2019-09-08.tsv'.format(num_weeks))
    model_file = os.path.join(ROOT_DATA_DIR,'eq_up_to_{0}d_before_second_delivery/best_xgb_model_eq_samp_size_raw_counts_icd_cpt_up_to_{0}_days_before_second_delivery-2019-09-08.pickle'.format(num_weeks))
    _, _, X_test, y_test, xgb_model, input_data = unpack_input_data(input_file, model_file)
    metrics_results, _, _ = validate_best_model(xgb_model, X_test, y_test)
    interp_rc, interp_pr, pr_auc, pos_prop = get_pr_coord(metrics_results, y_test)


    up_to_dataset_dict['{}_days'.format(num_weeks)] =  {'interp_rc': interp_rc, 'interp_pr': interp_pr, 'pr_auc':pr_auc, 'pos_prop':pos_prop}


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


# colors
hex_codes = [ '#%02x%02x%02x' % tuple((np.array(x)*255).astype(int)) for x in sns.color_palette("cubehelix", 3)]

for index, timepoint in enumerate(up_to_dataset_dict.keys()):
    rc_coords = up_to_dataset_dict[timepoint]['interp_rc']
    pr_coords = up_to_dataset_dict[timepoint]['interp_pr']
    pr_auc = up_to_dataset_dict[timepoint]['pr_auc']
    pos_prop = up_to_dataset_dict[timepoint]['pos_prop']
    _ = ax.plot(rc_coords, pr_coords, lw=2, alpha=1, linestyle='-',  color=hex_codes[index], label="{:.2f}, {}".format(pr_auc, timepoint.replace('_', ' ')))

_ = ax.plot([0, 1], [pos_prop, pos_prop], linestyle='-', lw=1,   color='#CD5C5C', label='{:.2f}, Chance'.format(pos_prop), alpha=1)


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
# plt.savefig(os.path.join(OUTPUT_DIR, f'{DATE}_pr_0_90_365days_before_second_ptb_from_delivery_date_vu_w_legend.pdf'),  bbox_inches = 'tight', pad_inches=0, bbox_extra_artists=[legend], transparent=True)

# without legend
# legend.remove()
# plt.subplots_adjust(left=0.15,right=0.9, top=0.9, bottom=0.15)
# plt.savefig(os.path.join(OUTPUT_DIR, f'{DATE}_pr_0_90_365days_before_second_ptb_from_delivery_date_vu.pdf'),  bbox_inches = None, pad_inches=0, transparent=True)



