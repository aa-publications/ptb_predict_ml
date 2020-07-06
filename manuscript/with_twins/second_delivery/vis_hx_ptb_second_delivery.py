#!/bin/python
# This script will visualize teh results from predicting preterm birth exlucding codes x days since conception.
#
#
#
# Abin Abraham
# created on: 2019-09-11 15:40:03


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


DATE = datetime.now().strftime('%Y-%m-%d')



sys.path.append('/dors/capra_lab/users/abraha1/projects/PTB_phenotyping/scripts/rand_forest_ptb_classification')
from hyperparam_tune import validate_best_model

import matplotlib.font_manager as fm
fpath='/dors/capra_lab/users/abraha1/conda/envs/py36_r_ml/lib/python3.6/site-packages/matplotlib/mpl-data/fonts/ttf/Arial.ttf'
bprop = fm.FontProperties(fname=fpath, size=10)
sprop = fm.FontProperties(fname=fpath, size=8)

import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42


sys.path.append('/dors/capra_lab/users/abraha1/projects/PTB_phenotyping/scripts/rand_forest_ptb_classification')
from hyperparam_tune import validate_best_model
%matplotlib inline



# PATHS
ROOT_DATA_DIR = "/dors/capra_lab/users/abraha1/projects/PTB_phenotyping/results/ptb_predict_machine_learning/2019_09_06_2nd_ptb_icd_cpt/equal_sample_size"
OUTPUT_DIR =    "/dors/capra_lab/users/abraha1/projects/PTB_phenotyping/scripts/rand_forest_ptb_classification/manuscript/second_delivery"

# -----------
# FUNCTIONS
# -----------

def get_test_performance(input_file, model_file):


    input_data = pd.read_csv(input_file, sep="\t")
    print("done loading {}".format(os.path.basename(input_file)))

    held_out_df = input_data.loc[input_data['partition']=='held_out'].copy()
    train_df = input_data.loc[input_data['partition']=='grid_cv'].copy()
    held_out_df.set_index('GRID',inplace=True)
    train_df.set_index('GRID',inplace=True)

    X_test = held_out_df.iloc[:,:-2]
    y_test = held_out_df.label.apply(lambda x: 1 if x == 'preterm' else 0).values


    xgb_model = pickle.load(open(model_file, "rb"))

    return X_test, y_test, xgb_model


def plot_roc(data_dict, title="ROC", plabel="", savefig_bool=False):

    plot_labels_dict = dict(zip(data_dict.keys(), ['{}_since_preg_start'.format(x) for x in data_dict.keys()]))

    fig, ax  = plt.subplots()
    interp_fpr = np.linspace(0, 1, 100)
    for index, label_dict in enumerate(data_dict.items()):
        label, inner_dict = label_dict

        # unpack data
        metrics_results = inner_dict['metrics_results']
        fpr = metrics_results['fpr']
        tpr = metrics_results['tpr']
        auc = metrics_results['roc_auc']

        # calc
        lin_fx = interp1d(fpr, tpr, kind='linear', assume_sorted=True)
        interp_tpr = lin_fx(interp_fpr)

        # force a 0,0 start
        interp_fpr = np.hstack((np.array([0]), interp_fpr))
        interp_tpr = np.hstack((np.array([0]), interp_tpr))

        _ = plt.plot(interp_fpr, interp_tpr, lw=1, alpha=1, linestyle='-', color=hex_codes[index],
                 label="{} ({:.2f})".format(label.replace("_weeks", " days"), auc))

    ax.spines['left'].set_linewidth(0.5)
    ax.spines['bottom'].set_linewidth(0.5)
    ax.spines['top'].set_linewidth(0.5)
    ax.spines['right'].set_linewidth(0.5)


    _ = plt.plot([0, 1], [0, 1], linestyle='-', lw=1, color='#CD5C5C', label='Chance', alpha=.8)

    _ = ax.set_xlabel('False Positive Rate', fontproperties=sprop)
    _ = ax.set_ylabel('True Positive Rate', fontproperties=sprop)
    _ = ax.set_title(title, fontproperties=sprop)

    _ = ax.tick_params(axis='both', which='major')
    _ = ax.axis('equal')
    _ = ax.set_xlim(0,1.0)
    _ = ax.set_ylim(0,1.0)
    _ = ax.set_xticks(np.arange(0,1.2,0.2))
    _ = ax.set_xticklabels(['{:.1f}'.format(x) for x in np.arange(0,1.2,0.2)], fontproperties=sprop)
    _ = ax.set_yticks(np.arange(0,1.2,0.2))
    _ = ax.set_yticklabels(['{:.1f}'.format(x) for x in np.arange(0,1.2,0.2)],fontproperties=sprop)



    leg = plt.legend(loc="center left",  bbox_to_anchor=(1, 0.5),  prop=sprop, frameon=True, fancybox=False, framealpha=1, shadow=False)
    leg.set_title("Days Before Delivery (AUC)", prop =sprop)


    # save
    roc_fig_file = os.path.join(OUTPUT_DIR, '{}-roc_auc_up_to_before_{}_delivery.pdf'.format(DATE,plabel))
    if savefig_bool:
        plt.savefig(roc_fig_file, bbox_inches = 'tight', pad_inches=0.1, bbox_extra_artists=[leg],  transparent=True)
    plt.clf()


def plot_pr(data_dict, title="PR", plabel="", savefig_bool=False):
    fig,ax  = plt.subplots()
    interp_rc = np.linspace(0, 1, 100)

    for index, label_dict in enumerate(data_dict.items()):
        label, inner_dict = label_dict

        # get data
        metrics_results = inner_dict['metrics_results']
        pr = metrics_results['pr_curve']
        rc = metrics_results['rc_curve']
        pr_auc = metrics_results['avg_prec']
        pos_prop=np.sum(inner_dict['y_test'])/len(inner_dict['y_test'])

        # calc
        lin_fx = interp1d(rc, pr, kind='linear')
        interp_pr = lin_fx(interp_rc)

        # force a 1,pos_prop end
        interp_rc = np.hstack((interp_rc, np.array([1])))
        interp_pr = np.hstack((interp_pr, np.array([pos_prop])))

        _ = plt.plot(interp_rc, interp_pr, lw=1, alpha=1, linestyle='-', color=hex_codes[index], label="{} ({:.2f})".format(label.replace("_weeks", " days"), pr_auc))

    _ = plt.plot([0, 1], [pos_prop, pos_prop], linestyle='-', lw=1, color='#CD5C5C', label='Chance ({:.2f})'.format(pos_prop), alpha=1)


    ax.spines['left'].set_linewidth(0.5)
    ax.spines['bottom'].set_linewidth(0.5)
    ax.spines['top'].set_linewidth(0.5)
    ax.spines['right'].set_linewidth(0.5)


    _ = ax.set_xlabel('Recall', fontproperties=sprop)
    _ = ax.set_ylabel('Precision', fontproperties=sprop)
    _ = ax.set_title(title, fontproperties=sprop)

    _ = ax.tick_params(axis='both', which='major')
    _ = ax.axis('equal')
    _ = ax.set_xlim(0,1.0)
    _ = ax.set_ylim(0,1.0)
    _ = ax.set_xticks(np.arange(0,1.2,0.2))
    _ = ax.set_xticklabels(['{:.1f}'.format(x) for x in np.arange(0,1.2,0.2)], fontproperties=sprop)
    _ = ax.set_yticks(np.arange(0,1.2,0.2))
    _ = ax.set_yticklabels(['{:.1f}'.format(x) for x in np.arange(0,1.2,0.2)],fontproperties=sprop)



    leg = plt.legend(loc="center left",  bbox_to_anchor=(1, 0.5),  prop=sprop, frameon=True, fancybox=False, framealpha=1, shadow=False)
    leg.set_title("Days Before Delivery (AUC)", prop =sprop)


    # # save
    if savefig_bool:
        pr_fig_file = os.path.join(OUTPUT_DIR, '{}-pr_auc_up_to_before_{}_delivery.pdf'.format(DATE, plabel))
        plt.savefig(pr_fig_file, bbox_inches = 'tight', pad_inches=0.1, bbox_extra_artists=[leg],  transparent=True)
    plt.clf()

# %%

# -----------
# MAIN
# -----------


###
#   LOAD DATA
###


# set paths
up_to_dataset_dict = OrderedDict()
for num_weeks in ['0','30','60', '90','273', '365']:

    input_file = os.path.join(ROOT_DATA_DIR,'eq_up_to_{0}d_before_second_delivery/input_data_eq_samp_size_raw_counts_icd_cpt_up_to_{0}_days_before_second_delivery-2019-09-08.tsv'.format(num_weeks))
    model_file = os.path.join(ROOT_DATA_DIR,'eq_up_to_{0}d_before_second_delivery/best_xgb_model_eq_samp_size_raw_counts_icd_cpt_up_to_{0}_days_before_second_delivery-2019-09-08.pickle'.format(num_weeks))

    up_to_dataset_dict['{}_weeks'.format(num_weeks)] =  {'input_file': input_file, 'model_file': model_file}

# output of pickle loaded data
STORED_DATA_FILE = os.path.join(OUTPUT_DIR,'up_to_second_delivery.pickle')



# load if the data already exists...
if os.path.isfile(STORED_DATA_FILE):
    print("loading pickled file...")
    metrics_file = open(STORED_DATA_FILE, 'rb')
    up_to_dataset_dict = pickle.load(metrics_file)

else:
    print("creating data...")
    store_results = {}
    for label, inner_dict in up_to_dataset_dict.items():

        print(label)
        X_test, y_test, xgb_model  = get_test_performance(inner_dict['input_file'], inner_dict['model_file'])
        metrics_results, metrics_df, model_params = validate_best_model(xgb_model, X_test, y_test)

        up_to_dataset_dict[label]['metrics_df'] = metrics_df
        up_to_dataset_dict[label]['metrics_results'] = metrics_results
        up_to_dataset_dict[label]['y_test'] = y_test
        up_to_dataset_dict[label]['X_test'] = X_test


    pickle.dump(up_to_dataset_dict, open(STORED_DATA_FILE, 'wb'))
    print("pickled model.")


# %%
###
#   CREATE ROCs
###


### fig paramaters
sns.set( style='whitegrid',  font_scale=1.5, rc={'figure.figsize':(8,8)} )
sns.set_style( {'axes.grid': True, 'axes.edgecolor': 'k', 'font.sans-serif': ['Arial'], 'grid.color': '#e1e1e1'})
plt.rc('axes', prop_cycle=(cycler('color', sns.color_palette("cubehelix", 5))))
fsize=20
leg_fsize=14
hex_codes = [ '#%02x%02x%02x' % tuple((np.array(x)*255).astype(int)) for x in sns.color_palette("cubehelix", 7)]




# %%

#
#   AUC ROC
#

plot_labels_dict = dict(zip(up_to_dataset_dict.keys(), ['{}_since_preg_start'.format(x) for x in up_to_dataset_dict.keys()]))


### fig paramaters
sns.set( style='whitegrid',  font_scale=1, rc={'figure.figsize':(3,3)} )
sns.set_style( {'axes.grid': True, 'axes.edgecolor': 'k', 'font.sans-serif': ['Arial'], 'grid.color': '#e1e1e1'})

fsize=20
leg_fsize=14
hex_codes = [ '#%02x%02x%02x' % tuple((np.array(x)*255).astype(int)) for x in sns.color_palette("cubehelix", 6)]



plot_roc(up_to_dataset_dict, title="ROC, Second Delivery", plabel="second_delivery", savefig_bool=True)
plot_pr(up_to_dataset_dict, title="PR, Second Delivery", plabel="second_delivery",savefig_bool=True)



fig  = plt.plot()
interp_fpr = np.linspace(0, 1, 100)
for index, label_dict in enumerate(up_to_dataset_dict.items()):
    label, inner_dict = label_dict

    # unpack data
    metrics_results = inner_dict['metrics_results']
    fpr = metrics_results['fpr']
    tpr = metrics_results['tpr']
    auc = metrics_results['roc_auc']

    # calc
    lin_fx = interp1d(fpr, tpr, kind='linear', assume_sorted=True)
    interp_tpr = lin_fx(interp_fpr)

    # force a 0,0 start
    interp_fpr = np.hstack((np.array([0]), interp_fpr))
    interp_tpr = np.hstack((np.array([0]), interp_tpr))

    _ = plt.plot(interp_fpr, interp_tpr, lw=2, alpha=1, linestyle='-',
             label="{}(AUC={:.3f})".format(label, auc))

_ = plt.plot([0, 1], [0, 1], linestyle='-', lw=1, color='#CD5C5C', label='Chance', alpha=.8)
_ = plt.xlim([-0.05, 1.05])
_ = plt.ylim([-0.05, 1.05])
_ = plt.xlabel('False Positive Rate', fontsize=fsize)
_ = plt.ylabel('True Positive Rate', fontsize=fsize)
_ = plt.title('ROC for X days before second PTB after Hx of PTB', fontsize=fsize+2)
_ = plt.legend(loc="lower right", fontsize=leg_fsize, frameon=True, fancybox=False, framealpha=.9, shadow=True,borderpad=0.5)
_ = plt.tick_params(axis='both', which='major', labelsize=fsize)

# save
roc_fig_file = os.path.join(OUTPUT_DIR, '{}-roc_auc_hx_ptb_up_to_before_second_delivery.pdf'.format(DATE))
# plt.savefig(roc_fig_file)

OUTPUT_DIR
# %%
#
#   PR AUC
#

fig  = plt.plot()
interp_rc = np.linspace(0, 1, 100)

for index, label_dict in enumerate(up_to_dataset_dict.items()):
    label, inner_dict = label_dict

    # get data
    metrics_results = inner_dict['metrics_results']
    pr = metrics_results['pr_curve']
    rc = metrics_results['rc_curve']
    pr_auc = metrics_results['avg_prec']
    pos_prop=np.sum(inner_dict['y_test'])/len(inner_dict['y_test'])

    # calc
    lin_fx = interp1d(rc, pr, kind='linear')
    interp_pr = lin_fx(interp_rc)

    # force a 1,pos_prop end
    interp_rc = np.hstack((interp_rc, np.array([1])))
    interp_pr = np.hstack((interp_pr, np.array([pos_prop])))

    _ = plt.plot(interp_rc, interp_pr, lw=2, alpha=1, linestyle='-', color=hex_codes[index],
             label="{}(AUC={:.3f})".format(label, pr_auc))


_ = plt.plot([0, 1], [pos_prop, pos_prop], linestyle='-', lw=1, color='#CD5C5C', label='Chance({:.3f})'.format(pos_prop), alpha=1)
_ = plt.xlim([-0.05, 1.05])
_ = plt.ylim([-0.05, 1.05])
_ = plt.xlabel('Recall', fontsize=fsize)
_ = plt.ylabel('Precision', fontsize=fsize)
_ = plt.title('PR for X days before second PTB after Hx of PTB', fontsize=fsize+2)
_ = plt.legend(loc="best", bbox_to_anchor=(1, 1), fontsize=leg_fsize, frameon=True, fancybox=False, framealpha=1, shadow=False,borderpad=0.5)
_ = plt.tick_params(axis='both', which='major', labelsize=fsize)

# # save
pr_fig_file = os.path.join(OUTPUT_DIR, '{}-pr_auc_hx_ptb_up_to_before_second_delivery.pdf'.format(DATE))

# plt.savefig(pr_fig_file, bbox_inches='tight')

# %%
