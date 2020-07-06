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


sys.path.append('/dors/capra_lab/users/abraha1/projects/PTB_phenotyping/scripts/rand_forest_ptb_classification')
from hyperparam_tune import validate_best_model

sys.path.append("/dors/capra_lab/users/abraha1/projects/PTB_phenotyping/scripts/rand_forest_ptb_classification/manuscript/0_helper_func")
from manip_trained_models_funcs import unpack_input_data, upickle_xgbmodel, extract_train_df, extract_test_df




import matplotlib.font_manager as fm
fpath='/dors/capra_lab/users/abraha1/conda/envs/py36_r_ml/lib/python3.6/site-packages/matplotlib/mpl-data/fonts/ttf/Arial.ttf'
bprop = fm.FontProperties(fname=fpath, size=10)
sprop = fm.FontProperties(fname=fpath, size=8)

import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

%matplotlib inline

DATE = datetime.now().strftime('%Y-%m-%d')

# PATHS
ROOT_DATA_DIR = "/dors/capra_lab/users/abraha1/projects/PTB_phenotyping/results/ptb_predict_machine_learning/2019_05_20_time_since_conception"
OUTPUT_DIR = "/dors/capra_lab/users/abraha1/projects/PTB_phenotyping/scripts/rand_forest_ptb_classification/manuscript/time_since_conception"

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

def plot_roc(store_fpr, store_tpr, aucs, plt_prefix='', roc_fig_file=None):
    '''
    plot auroc curve(s) with mean and std; save if a roc_fig_file is provided

        INPUTS:
            * store_fpr, store_tpr, aucs: a list where each element represents data for one curve
            * plt_prefix: label to add to the title of plot
            * roc_fig_file: full path to save file
            * savefig: boolean to save or not save figure

        note: first three must be a list; will not plot mean and std if only one curve
    '''
    print("Creating roc plot.....")

    interp_fpr = np.linspace(0, 1, 100)
    store_tpr_interp = []

    ax = plt.figure()
    plt.plot([0, 1], [0, 1], linestyle='--', lw=4, color='r', label='Chance', alpha=.8)

    # plot each cv iteration
    for cv_iter, fpr_tpr_auc in enumerate(zip(store_fpr, store_tpr, aucs)):
        # set_trace()
        fpr, tpr, auc = fpr_tpr_auc
        plt.plot(fpr, tpr, lw=4, alpha=0.9, label="#{}(AUC={:.3f})".format(cv_iter, auc))

        lin_fx = interp1d(fpr, tpr, kind='linear')
        interp_tpr = lin_fx(interp_fpr)

        # store_tpr_interp.append(np.interp(mean_fpr, fpr, tpr))
        # store_tpr_interp[-1][0] = 0.0
        store_tpr_interp.append(interp_tpr)

    # plot mean and std only if more than one curve present
    if len(store_fpr) != 1:
        # plot mean, sd, and shade in between
        mean_tpr = np.mean(store_tpr_interp, axis=0)
        # mean_tpr[-1] = 1.0
        mean_auc = metrics.auc(interp_fpr, mean_tpr)
        std_auc = np.std(aucs)
        plt.plot(interp_fpr, mean_tpr, color='b',
                 label="Mean(AUC={:.2f}+/-{:.2f})".format(mean_auc, std_auc),
                 lw=2, alpha=.8)

        std_tpr = np.std(store_tpr_interp, axis=0)
        tprs_upper = np.minimum(mean_tpr + 2*std_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - 2*std_tpr, 0)
        plt.fill_between(interp_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                         label="+/- 2 S.D.")

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC for {}:\nPrediting PTB vs. non-PTB'.format(plt_prefix))
    plt.legend(loc="lower right")

    if roc_fig_file:
        plt.savefig(roc_fig_file)
        print("\tDone. AUROC curve saved to:\n\t{}".format(roc_fig_file))

    return ax

def plot_pr(precisions, recalls, avg_prs, plt_prefix, pr_fig_file=None, pos_prop=None):
    ''' plot PR curve(s) with mean and std; save if pr_fig_file is provided

        INPUTS:
            * precisions, recalls, avg_prs: must be a list where each element represents data for one curve
            * plt_prefix: label to add to the title of plot
            * pr_fig_file: full path to save file
            * pos_prop: total true positives / total samples (i.e. proportion of positves)

        note: first three must be a list; will not plot mean and std if only one curve
    '''
    print("Creating PR curve plot ...")
    # mean_rc = np.linspace(0, 1, 100)
    interp_rc = np.linspace(0, 1, 100)

    store_pr_interp = []
    ax = plt.figure()

    # plot line of random chance
    if pos_prop:
        plt.plot([0, 1], [pos_prop, pos_prop], linestyle='--', lw=4,
                 color='r', label='Chance({:.3f})'.format(pos_prop), alpha=.8)

    # plot each cv_iter
    for cv_iter, pr_rc_avg in enumerate(zip(precisions, recalls, avg_prs)):

        pr_array, rc_array, pr_avg = pr_rc_avg
        # plt.plot(rc_array, pr_array, lw=1, color='k', alpha=0.4)
        plt.step(rc_array, pr_array, lw=4, alpha=0.8, where='post', label="#{}(AvgPR={:.3f})".format(cv_iter, pr_avg))

        # interpolate recall to have the same length array for taking mean
        lin_fx = interp1d(rc_array, pr_array, kind='linear')
        interp_pr = lin_fx(interp_rc)
        store_pr_interp.append(interp_pr)

    # set_trace()

    # plot mean and std only if more than one curve present
    if len(precisions) != 1:
        # mean and std
        mean_pr = np.mean(store_pr_interp, axis=0)
        mean_avg_pr = np.mean(avg_prs)
        std_avg_pr = np.std(avg_prs)

        # std of each pr-curve
        std_pr = np.std(store_pr_interp, axis=0)
        pr_upper = np.minimum(mean_pr + 2*std_pr, 1)
        pr_lower = np.maximum(mean_pr - 2*std_pr, 0)
        plt.fill_between(interp_rc, pr_lower, pr_upper, color='grey', alpha=.2,
                         label="+/- 2 S.D.")

        plt.plot(interp_rc, mean_pr, color='b',
                 label="Mean(AUC={:.2f}+/-{:.2f})".format(mean_avg_pr, std_avg_pr), lw=2, alpha=0.8)

    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('PR Curve for {}:\nPrediting PTB vs. non-PTB'.format(plt_prefix))
    plt.legend(loc="lower right")

    if pr_fig_file:
        plt.savefig(pr_fig_file)
        print("\tPR curve saved to:\n\t{}".format(pr_fig_file))

    return ax


def get_auroc_coords(up_to_dataset_dict):

    coords_dict = dict()
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


        coords_dict[label] = [interp_fpr, interp_tpr, auc]

    return coords_dict


# %%
# -----------
# MAIN
# -----------


###
#   LOAD DATA
###


# set paths
up_to_dataset_dict = OrderedDict()
for num_weeks in ['0','13','28', '32','35', '37']:

    input_file = glob.glob(ROOT_DATA_DIR+'/up_to_{0}_weeks_since_preg_start_icd9_cpt_count/input_data_up_to_{0}_weeks_since_preg_start_icd9_cpt_count-2019-06-19.tsv'.format(num_weeks))[0]
    model_file = glob.glob(ROOT_DATA_DIR+'/up_to_{0}_weeks_since_preg_start_icd9_cpt_count/best_xgb_model_up_to_{0}_weeks_since_preg_start_icd9_cpt_count-2019-06-19.pickle'.format(num_weeks))[0]

    up_to_dataset_dict['{}_weeks'.format(num_weeks)] =  {'input_file': input_file, 'model_file': model_file}

# output of pickle loaded data
UPTO_STORED_DATA_FILE = os.path.join(OUTPUT_DIR,'up_to_since_preg_start_dict.pickle')



# load if the data already exists...
if os.path.isfile(UPTO_STORED_DATA_FILE):
    print("loading pickled file...")
    metrics_file = open(UPTO_STORED_DATA_FILE, 'rb')
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


    pickle.dump(up_to_dataset_dict, open(UPTO_STORED_DATA_FILE, 'wb'))
    print("pickled model.")


# %%
###
#   CREATE ROCs
###


# %%
### fig paramaters
sns.set( style='whitegrid',  font_scale=1.0, rc={'figure.figsize':(3,3)} )
sns.set_style( {'axes.grid': True, 'axes.edgecolor': 'k',  'grid.color': '#e1e1e1'})
sprop = fm.FontProperties(fname=fpath, size=8)
fsize=14
leg_fsize=14

# plt.rc('axes', prop_cycle=(cycler('color', sns.color_palette("cubehelix", 5))))
hex_codes = [ '#%02x%02x%02x' % tuple((np.array(x)*255).astype(int)) for x in sns.color_palette("cubehelix", 6)]
# hex_codes = [ '#%02x%02x%02x' % tuple((np.array(x)*255).astype(int)) for x in sns.color_palette("Set2", 7)]


plot_labels_dict = dict(zip(up_to_dataset_dict.keys(), ['{}_since_preg_start'.format(x) for x in up_to_dataset_dict.keys()]))

### -- plot
fig ,ax = plt.subplots()

coord_dict = get_auroc_coords(up_to_dataset_dict)
count = 0
for label, vals in coord_dict.items():
    _ = plt.plot(vals[0], vals[1], lw=1, alpha=1, color=hex_codes[count], linestyle='-', label="{} ({:.2f})".format(label.replace("_", " "), vals[2]))
    count = count + 1


_ = plt.plot([0, 1], [0, 1], linestyle='-', lw=1, color='#CD5C5C', label='Chance', alpha=.8)


leg = plt.legend(loc="center left",  bbox_to_anchor=(1, 0.5), title="Weeks of Gestation", prop=sprop, frameon=True, fancybox=False, framealpha=.7, shadow=False, borderpad=0.5)
_ = plt.tick_params(axis='both', which='major', labelsize=fsize)


leg.set_title("Gestation (AUC)", prop =sprop)
ax.spines['left'].set_linewidth(0.5)
ax.spines['bottom'].set_linewidth(0.5)
ax.spines['top'].set_linewidth(0.5)
ax.spines['right'].set_linewidth(0.5)

ax.set_xlabel('False Positive Rate', fontproperties=sprop)
ax.set_ylabel('True Positive Rate', fontproperties=sprop)
ax.set_title('ROC', fontproperties=sprop)

_ = ax.tick_params(axis='both', which='major')
_ = ax.axis('equal')
_ = ax.set_xlim(0,1.0)
_ = ax.set_ylim(0,1.0)
_ = ax.set_xticks(np.arange(0,1.2,0.2))
_ = ax.set_xticklabels(['{:.1f}'.format(x) for x in np.arange(0,1.2,0.2)], fontproperties=sprop)
_ = ax.set_yticks(np.arange(0,1.2,0.2))
_ = ax.set_yticklabels(['{:.1f}'.format(x) for x in np.arange(0,1.2,0.2)],fontproperties=sprop)

# save
roc_fig_file = os.path.join(OUTPUT_DIR, 'figures', '{}-roc_auc_up_to_x_wks_since_preg_start.pdf'.format(DATE))
plt.savefig(roc_fig_file,  bbox_inches = 'tight', pad_inches=0.1, bbox_extra_artists=[leg],  transparent=True)





# %%
#
#   PR AUC
#
sns.set( style='whitegrid',  font_scale=1.0, rc={'figure.figsize':(3,3)} )
sns.set_style( {'axes.grid': True, 'axes.edgecolor': 'k',  'grid.color': '#e1e1e1'})
sprop = fm.FontProperties(fname=fpath, size=8)
fsize=14
leg_fsize=14


hex_codes = [ '#%02x%02x%02x' % tuple((np.array(x)*255).astype(int)) for x in sns.color_palette("cubehelix", 6)]

fig ,ax = plt.subplots()


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

    _ = plt.plot(interp_rc, interp_pr, lw=1, alpha=1, linestyle='-', color=hex_codes[index], label="{} ({:.2f})".format(label.replace("_", " "), pr_auc))
    _ = plt.plot([0, 1], [pos_prop, pos_prop], linestyle=':', lw=1, color=hex_codes[index], label='Chance ({:.2f})'.format(pos_prop), alpha=1)

    #


leg = plt.legend(loc="center left",  bbox_to_anchor=(1, 0.5),  prop=sprop, frameon=True, fancybox=False, framealpha=1, shadow=False)
_ = plt.tick_params(axis='both', which='major', labelsize=fsize)
leg.set_title("Gestation (AUC)", prop =sprop)


ax.spines['left'].set_linewidth(0.5)
ax.spines['bottom'].set_linewidth(0.5)
ax.spines['top'].set_linewidth(0.5)
ax.spines['right'].set_linewidth(0.5)


ax.set_xlabel('Recall', fontproperties=sprop)
ax.set_ylabel('Precision', fontproperties=sprop)
ax.set_title('PR', fontproperties=sprop)


_ = ax.tick_params(axis='both', which='major')
_ = ax.axis('equal')
_ = ax.set_xlim(0,1.0)
_ = ax.set_ylim(0,1.0)
_ = ax.set_xticks(np.arange(0,1.2,0.2))
_ = ax.set_xticklabels(['{:.1f}'.format(x) for x in np.arange(0,1.2,0.2)], fontproperties=sprop)
_ = ax.set_yticks(np.arange(0,1.2,0.2))
_ = ax.set_yticklabels(['{:.1f}'.format(x) for x in np.arange(0,1.2,0.2)],fontproperties=sprop)



# # save
pr_fig_file = os.path.join(OUTPUT_DIR, 'figures', '{}-pr_auc_up_to_x_wks_since_preg_start.pdf'.format(DATE))
plt.savefig(pr_fig_file,  bbox_inches = 'tight', pad_inches=0.1, bbox_extra_artists=[leg],  transparent=True)


# %%
