#!/bin/python
# This script will ...
#
#
#
# Abin Abraham
# created on: 2020-04-09 23:49:01




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

import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42





from datetime import datetime
DATE = datetime.now().strftime('%Y-%m-%d')


###
###    functions
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

def plot_roc(metrics_results, ax, color, linestyle, label, fontprop):

    _ = ax.plot([0, 1], [0, 1], linestyle='-', lw=1, color='#CD5C5C', label='', alpha=.8)

    # plot rocs
    interp_fpr, interp_tpr, auc  = get_auroc_coords(metrics_results)
    _ = ax.plot(interp_fpr, interp_tpr, lw=2, alpha=1, linestyle=linestyle, color=color,  label="{} ({:.2f})".format(label, auc))


    ax.set_xlim(0,1.0)
    ax.set_ylim(0,1.0)
    ax.set_xticks(np.arange(0,1.2,0.2))
    ax.set_xticklabels(['{:.1f}'.format(x) for x in np.arange(0,1.2,0.2)], fontproperties=sprop)
    ax.set_yticks(np.arange(0,1.2,0.2))
    ax.set_yticklabels(['{:.1f}'.format(x) for x in np.arange(0,1.2,0.2)],fontproperties=sprop)

    _ = ax.set_xlabel('False Positive Rate', fontproperties=sprop)
    _ = ax.set_ylabel('True Positive Rate', fontproperties=sprop)


    ax.spines['left'].set_linewidth(0.5)
    ax.spines['bottom'].set_linewidth(0.5)
    ax.spines['top'].set_linewidth(0.5)
    ax.spines['right'].set_linewidth(0.5)
    ax.tick_params(width=0.5, length=2.5)
    sns.despine(ax=ax, top=False, right=False)
    ax.set_aspect('equal')

    return ax, legend

def plot_pr(metrics_results, ax, color, linestyle, label, fontprop):

    # curves
    interp_rc, interp_pr, pr_auc, pos_prop  = get_pr_coord(metrics_results)
    _ = ax.plot([0, 1], [pos_prop, pos_prop], linestyle='-', lw=1, color=color, label='', alpha=0.7)
    _ = ax.plot(interp_rc, interp_pr, lw=2, alpha=1, linestyle="-", color=color,  label="{} ({:.2f})".format(label, pr_auc))



    ax.set_xlim(0,1.0)
    ax.set_ylim(0,1.0)

    ax.set_xticks(np.arange(0,1.2,0.2))
    ax.set_xticklabels(['{:.1f}'.format(x) for x in np.arange(0,1.2,0.2)], fontproperties=sprop)

    ax.set_yticks(np.arange(0,1.2,0.2))
    ax.set_yticklabels(['{:.1f}'.format(x) for x in np.arange(0,1.2,0.2)],fontproperties=sprop)

    _ = ax.set_xlabel('Recall', fontproperties=sprop)
    _ = ax.set_ylabel('Precision', fontproperties=sprop)
    legend = ax.legend(loc="best", bbox_to_anchor=(1, 1), prop=sprop, frameon=True, fancybox=False, framealpha=1, shadow=False, borderpad=0.5, title='AU-PR')

    ax.spines['left'].set_linewidth(0.5)
    ax.spines['bottom'].set_linewidth(0.5)
    ax.spines['top'].set_linewidth(0.5)
    ax.spines['right'].set_linewidth(0.5)
    ax.tick_params(width=0.5, length=2.5)
    sns.despine(ax=ax, top=False, right=False)
    ax.set_aspect('equal')

    return ax

def save_fig(legend_bool, file_name, plt):

    if legend_bool:
        plt.tight_layout()

        plt.savefig(os.path.join(OUTPUT_DIR, f'{DATE}_roc_icd9_cpt.pdf'),  bbox_inches = None, pad_inches=0, transparent=True)


    else:
        ax.get_legend().remove()
        plt.subplots_adjust(left=0.2,right=1, top=0.9, bottom=0.2)
        plt.savefig(os.path.join(OUTPUT_DIR, f'{DATE}_roc_icd9_cpt_no_legend.eps'),  bbox_inches = 'tight', pad_inches=0, bbox_extra_artists=[legend], transparent=True)

    print("done! ")


###
### paths
###


ROOT_DATA_DIR = "/dors/capra_lab/users/abraha1/projects/PTB_phenotyping/results/ptb_predict_machine_learning/2019-01-16_xgboost_hyperopt_icd_cpt_raw_counts"
OUTPUT_DIR = "/dors/capra_lab/users/abraha1/projects/PTB_phenotyping/scripts/rand_forest_ptb_classification/manuscript/counts_icd_cpt"

icd_input = os.path.join(ROOT_DATA_DIR, 'input_data', 'input_data_all_icd9_count_subset-2019-01-25.tsv')
cpt_input = os.path.join(ROOT_DATA_DIR, 'input_data','input_data_all_cpt_count_subset-2019-01-26.tsv')
icd_cpt_input = os.path.join(ROOT_DATA_DIR, 'input_data','input_data_all_icd9_cpt_count_subset-2019-01-26.tsv')
# MODEL FILES
icd_model = os.path.join(ROOT_DATA_DIR, 'best_model','best_xgb_model_all_icd9_count_subset-2019-01-25.pickle')
cpt_model = os.path.join(ROOT_DATA_DIR, 'best_model', 'best_xgb_model_all_cpt_count_subset-2019-01-26.pickle')
icd_cpt_model = os.path.join(ROOT_DATA_DIR, 'best_model', 'best_xgb_model_all_icd9_cpt_count_subset-2019-01-26.pickle')


[icd_model, cpt_model, icd_cpt_model]


###
###    main
###

# -----------
# load data
# -----------

# %%
# for this_data, input_file, model_file in zip(['icd9','cpt','icd9_cpt'], [icd_input, cpt_input, icd_cpt_input], [icd_model, cpt_model, icd_cpt_model]):
#     print(f"data:{this_data}\ninput_file:{input_file}\nmodel_file:{model_file}")
#
#     X_train, y_train, X_test, y_test, xgb_model, input_data = unpack_input_data(input_file, model_file)
#     metrics_results, _, _ = validate_best_model(xgb_model, X_test, y_test)
#     pickle.dump(metrics_results, open(os.path.join(OUTPUT_DIR, f'{this_data}_roc_pr_data_dict_{DATE}.pickle'), 'wb'))
#     print("Done...")
#

# %%
icd9_metrics = pickle.load(open(os.path.join(OUTPUT_DIR, 'icd9_roc_pr_data_dict_2020-04-09.pickle'), 'rb'))
cpt_metrics = pickle.load(open(os.path.join(OUTPUT_DIR, 'cpt_roc_pr_data_dict_2020-04-09.pickle'), 'rb'))
icd9_cpt_metrics = pickle.load(open(os.path.join(OUTPUT_DIR, 'icd9_cpt_roc_pr_data_dict_2020-04-09.pickle'), 'rb'))


interp_rc, interp_pr, pr_auc, pos_prop = get_pr_coord(icd9_metrics)
# -----------
# section title
# -----------

# %%  plot roc
mult=1
sns.set( style='ticks', context='paper',  font_scale=1.0, rc={'figure.figsize':(2.2*mult,2.2*mult)} )
sns.set_style( {'axes.grid': True, 'axes.edgecolor': 'k',  'grid.color': '#e1e1e1'})
sprop = fm.FontProperties(fname=fpath, size=8*mult)
fig ,ax = plt.subplots()


colors=['#1b9e77', '#d95f02', '#7570b3']
linestyles = [':', ':', '-']
labels = ['icd9','cpt','icd9_cpt']



# loop through each model
ind = 0
for label, metrics_dict in zip(labels, [icd9_metrics, cpt_metrics, icd9_cpt_metrics]):
    ax = plot_roc(metrics_dict, ax, colors[ind], linestyles[ind], label, sprop)
    ind +=1

legend = ax.legend(loc="best", bbox_to_anchor=(1, 1), prop=sprop, frameon=True, fancybox=False, framealpha=1, shadow=False, borderpad=0.5, title='AU-ROC')

fig_file_name = os.path.join(OUTPUT_DIR, f'{DATE}_roc_icd9_cpt.pdf')
fig_no_legend_file_name = os.path.join(OUTPUT_DIR, f'{DATE}_roc_icd9_cpt_no_legend.pdf')

legend_bool = True

if legend_bool:
    plt.savefig(os.path.join(OUTPUT_DIR, f'{DATE}_roc_icd9_cpt.pdf'),  bbox_inches = 'tight', pad_inches=0, bbox_extra_artists=[legend], transparent=True)


else:
    ax.get_legend().remove()
    plt.subplots_adjust(left=0.2,right=1, top=0.9, bottom=0.2)
    plt.savefig(os.path.join(OUTPUT_DIR, f'{DATE}_roc_icd9_cpt_no_legend.pdf'),  bbox_inches = None, pad_inches=0, transparent=True)



# %% plot pr

mult=1
sns.set( style='ticks', context='paper',  font_scale=1.0, rc={'figure.figsize':(2.2*mult,2.2*mult)} )
sns.set_style( {'axes.grid': True, 'axes.edgecolor': 'k',  'grid.color': '#e1e1e1'})
sprop = fm.FontProperties(fname=fpath, size=8*mult)
fig ,ax = plt.subplots()

# loop through each model
ind = 0
for label, metrics_dict in zip(labels, [icd9_metrics, cpt_metrics, icd9_cpt_metrics]):
    ax = plot_pr(metrics_dict, ax, colors[ind], linestyles[ind], label, sprop)
    ind +=1


legend_bool = False

if legend_bool:
    plt.savefig(os.path.join(OUTPUT_DIR, f'{DATE}_pr_icd9_cpt.pdf'),  bbox_inches = 'tight', pad_inches=0, bbox_extra_artists=[legend], transparent=True)


else:
    ax.get_legend().remove()
    plt.subplots_adjust(left=0.2,right=1, top=0.9, bottom=0.2)
    plt.savefig(os.path.join(OUTPUT_DIR, f'{DATE}_pr_icd9_cpt_no_legend.pdf'),  bbox_inches = None, pad_inches=0, transparent=True)

# %%
###
###    Brier score
###

brier_df = pd.DataFrame()
for label, metrics_dict in zip(labels, [icd9_metrics, cpt_metrics, icd9_cpt_metrics]):


    brier_df = brier_df.append(pd.DataFrame({'label':[label], 'brier_score':[metrics_dict['brier_score']]}))


# %%
sns.set( style='ticks', context='paper',  font_scale=1.0, rc={'figure.figsize':(2.2*mult,2.2*mult)} )
# sns.set_style( {'axes.grid': True, 'axes.edgecolor': 'k',  'grid.color': '#e1e1e1'})


fig, ax = plt.subplots()
ax= sns.barplot(x='label', y='brier_score', data=brier_df, ax=ax)

ax.set_xlabel("")
ax.set_ylabel("Brier Score", fontproperties=sprop)

ax.set_ylim(0, 0.1)
ax.set_yticks(np.arange(0,0.12, 0.02))
ax.set_yticklabels(np.arange(0,0.12, 0.02),fontproperties=sprop)

ax.set_xticklabels(['ICD9','CPT','ICD9+CPT'], fontproperties=sprop)
sns.despine(ax=ax, top=True, right=True)

plt.savefig(os.path.join(OUTPUT_DIR, f'{DATE}_brier_icd9_cpt.pdf'),  bbox_inches = None, pad_inches=0,  transparent=True)
