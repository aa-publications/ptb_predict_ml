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

###
###    PATHS
###
DATE = datetime.now().strftime('%Y-%m-%d')


ICDCPT_DIR = "/dors/capra_lab/users/abraha1/projects/PTB_phenotyping/results/ptb_predict_machine_learning/2020_06_14_since_conception_icd9_cpt_no_twins_timeseries_v1"

OUTPUT_FIG_DIR = "/dors/capra_lab/users/abraha1/projects/PTB_phenotyping/scripts/rand_forest_ptb_classification/manuscript/no_twins_analyses/since_concep_timeseries/figures"


exp_label="since_concep_upto_28wk_no_twins_cmp_riskfx_icd_cpt"




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


def set_up_manu_roc(fpath, figsize=(2.25,2.25)):
    
    sns.set( style='ticks',  font_scale=1.0, rc={'figure.figsize':(2.25,2.25)} )
    sns.set_style( {'axes.grid': True, 'axes.edgecolor': 'k', 'axes.linewidth': 10,  'grid.color': '#e1e1e1'})
    sprop = fm.FontProperties(fname=fpath, size=6, )


    # plot
    fig ,ax = plt.subplots()
    
    return ax,sprop,


def manu_roc_format(ax, sprop, type, legend_loc):    

    if type == "roc":
        _ = ax.set_xlabel('False Positive Rate', fontproperties=sprop,labelpad=0.59)
        _ = ax.set_ylabel('True Positive Rate', fontproperties=sprop,labelpad=0.59)
        lg_title="Weeks gestation (AUC)"
    elif type == "pr":     
        _ = ax.set_xlabel('Recall', fontproperties=sprop,labelpad=0.59)
        _ = ax.set_ylabel('Precision', fontproperties=sprop,labelpad=0.59)
        lg_title="Weeks gestation (AUC, Chance)"
        
    _ = lg = ax.legend(prop=sprop, facecolor='white', edgecolor='white', frameon=True, fancybox=False, framealpha=1, shadow=False, borderpad=0.25, loc=legend_loc)

    _ = lg.set_title(title=lg_title, prop=sprop)
    _ = lg._legend_box.align = "left"
    _ = ax.tick_params(width=0.5, length=2.5)

    ax.grid(axis='both', linewidth=0.5, color='gainsboro')
    ax.spines['left'].set_linewidth(0.5)
    ax.spines['bottom'].set_linewidth(0.5)
    ax.spines['top'].set_linewidth(0.5)
    ax.spines['right'].set_linewidth(0.5)

    ax.set_xlim(0,1.0)
    ax.set_ylim(0,1.0)
    ax.set_xticks(np.arange(0,1.2,0.2))
    ax.set_yticks(np.arange(0,1.2,0.2))
    ax.set_xticklabels(['{:.1f}'.format(x) for x in np.arange(0,1.2,0.2)], fontproperties=sprop)
    ax.set_yticklabels(['{:.1f}'.format(x) for x in np.arange(0,1.2,0.2)], fontproperties=sprop)
    ax.set_aspect('equal','box')


    plt.subplots_adjust(left=0.15,right=0.95, top=.95, bottom=0.10)
    
    return ax 


# %%
# -----------
# MAIN
# -----------


# set up paths
timeseries = ['0_weeks', '13_weeks','28_weeks', '35_weeks', '37_weeks']

roc_dict = dict()
pr_dict = dict()
f1_score=dict()
for timepoint in timeseries: 
    results_dir = os.path.join(ICDCPT_DIR, f'{timepoint}_notwins_timeseries_v1')
    input_file = glob.glob(results_dir+"/input_data*.tsv")[0]
    model_file = glob.glob(results_dir+"/best_xgb_model*.pickle")[0]
    
    
    # load models and input files
    _, _, ehr_X_test, ehr_y_test, ehr_xgb_model, ehr_input_data = unpack_input_data(input_file, model_file)
    ehr_metrics_results, ehr_metrics_df, _ = validate_best_model(ehr_xgb_model, ehr_X_test, ehr_y_test)

    ehr_interp_fpr, ehr_interp_tpr, ehr_auc = get_auroc_coords(ehr_metrics_results)
    temp_roc_dict = {'interp_fpr':ehr_interp_fpr, 'interp_tpr':ehr_interp_tpr, 'auc':ehr_auc}
    roc_dict[timepoint] = temp_roc_dict
    f1_score[timepoint] = {'f1_score': ehr_metrics_results['f1_score'], 'pr_score': ehr_metrics_results['pr_score'], 'rc_score': ehr_metrics_results['rc_score']}
    
    ehr_interp_rc, ehr_interp_pr, ehr_pr_auc, ehr_pos_prop = get_pr_coord(ehr_metrics_results, ehr_y_test)
    temp_pr_dict = {'interp_rc':ehr_interp_rc, 'interp_pr':ehr_interp_pr, 'pr_auc':ehr_pr_auc, 'pos_prop':ehr_pos_prop}
    pr_dict[timepoint] = temp_pr_dict
    break
    
roc_dict    



# %%

###
###    plot
###

colors = ["#0077BB", "#33BBEE", "#009988", "#EE7733", "#CC3311", "#EE3377"]



ax,sprop = set_up_manu_roc(fpath, figsize=(2.25,2.25) )
for ind, timepoint in enumerate(timeseries): 
    this_dict = roc_dict[timepoint]
    _ = ax.plot(this_dict['interp_fpr'], this_dict['interp_tpr'], lw=1.5, alpha=1, linestyle='-',  color=colors[ind], label="{} ({:.2f})".format(timepoint.replace('_weeks', ''), this_dict['auc']))

_ = ax.plot([0, 1], [0, 1], linestyle='-', lw=0.5, color='black', label='Chance', alpha=.8)
ax = manu_roc_format(ax, sprop,  'roc', 'lower right')

plt.savefig(os.path.join(OUTPUT_FIG_DIR, f'{DATE}_auroc_since_concept_timeseries_no_twins.pdf'), pad_inches=0,  transparent=True)


# %% - PR

# plot
# ehr_interp_rc, ehr_interp_pr, ehr_pr_auc, ehr_pos_prop = get_pr_coord(ehr_metrics_results, ehr_y_test)


ax,sprop = set_up_manu_roc(fpath, figsize=(2.25,2.25))
for ind, timepoint in enumerate(timeseries): 
    this_dict = pr_dict[timepoint]
    _ = ax.plot(this_dict['interp_rc'], this_dict['interp_pr'], lw=1.5, alpha=1, linestyle='-',   color=colors[ind], label="{} ({:.2f}, {:.2f})".format(timepoint.replace('_weeks', ''), this_dict['pr_auc'], this_dict['pos_prop']))
    # _ = ax.plot([0, 1], [this_dict['pos_prop'], this_dict['pos_prop']], linestyle='-', lw=0.5,  color=colors[ind], label='{} Chance ({:.2f})'.format(timepoint.replace('_weeks', ''), this_dict['pos_prop']), alpha=1)
    _ = ax.plot([0, 1], [this_dict['pos_prop'], this_dict['pos_prop']], linestyle='--', lw=0.5,  color=colors[ind], label='', alpha=1)

ax = manu_roc_format(ax, sprop,'pr', 'upper right')
# plt.savefig(os.path.join(OUTPUT_FIG_DIR, f'{DATE}_aupr_since_concept_timeseries_no_twins.pdf'), pad_inches=0,  transparent=True)




# %% - F1

# plot
# ehr_interp_rc, ehr_interp_pr, ehr_pr_auc, ehr_pos_prop = get_pr_coord(ehr_metrics_results, ehr_y_test)


[f1_score[time]['f1_score'] for time in timeseries]
f1_score
