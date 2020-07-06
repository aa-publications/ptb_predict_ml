#!/bin/python
# This script will ...
#
#
#
# Abin Abraham
# created on: 'now'



import os, sys
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import glob
import pickle


from datetime import datetime
get_ipython().run_line_magic('matplotlib', 'inline')
np.set_printoptions(precision=5, suppress=True)


DATE = datetime.now().strftime('%Y-%m-%d')

sys.path.append('/dors/capra_lab/users/abraha1/projects/PTB_phenotyping/scripts/rand_forest_ptb_classification')
from hyperparam_tune import validate_best_model
from collections import OrderedDict
from cycler import cycler
import time as time


get_ipython().run_line_magic('load_ext', 'rpy2.ipython')



###
###    paths
###

ROOT_DATA_DIR = "/dors/capra_lab/users/abraha1/projects/PTB_phenotyping/results/ptb_predict_machine_learning/2019-01-16_xgboost_hyperopt_icd_cpt_raw_counts"
OUTPUT_DIR = "/dors/capra_lab/users/abraha1/projects/PTB_phenotyping/scripts/rand_forest_ptb_classification/manuscript/counts_icd_cpt"
ICD_CPT_DESCRIP_FILE ="/dors/capra_lab/users/abraha1/projects/PTB_phenotyping/data/ptb_predict_machine_learning/icd_cpt_descrip_mapping/descrip_master-col_names.txt"


###
###    functions
###


def get_train_test_arrays(input_data):
    held_out_df = input_data.loc[input_data['partition']=='held_out'].copy()
    train_df = input_data.loc[input_data['partition']=='grid_cv'].copy()
    held_out_df.set_index('GRID',inplace=True)
    train_df.set_index('GRID',inplace=True)

    binarize = lambda x: 1 if x == 'preterm' else 0

    X_train = train_df.iloc[:,:-2]
    y_train = train_df.label.apply(binarize).values
    X_test = held_out_df.iloc[:,:-2]
    y_test = held_out_df.label.apply(binarize).values


    return X_train, y_train, X_test, y_test

def unpack_input_data(input_file, model_file):
    '''load input file and split into training, and testing data '''

    input_data = pd.read_csv(input_file, sep="\t")
    print("done loading {}".format(os.path.basename(input_file)))

    X_train, y_train, X_test, y_test = get_train_test_arrays(input_data)
    xgb_model = pickle.load(open(model_file, "rb"))

    return  X_train, y_train, X_test, y_test, xgb_model, input_data


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


def create_descrip_dictionary(desrip_file):

    dsc_df = pd.read_csv(ICD_CPT_DESCRIP_FILE,sep="\t")

    dsc_df['good_map'] = ~(dsc_df['feature'] == dsc_df['short_desc'])
    dsc_df['mod_desc'] = dsc_df['feature'] + "-"+  dsc_df['short_desc']

    # remove [, ] or <
    dsc_df.mod_desc = dsc_df.mod_desc.str.replace("[", "-")
    dsc_df.mod_desc = dsc_df.mod_desc.str.replace("]","-")
    dsc_df.mod_desc = dsc_df.mod_desc.str.replace("<","-")


    desc_dict = dict(zip(dsc_df.feature, dsc_df.mod_desc))

    return desc_dict


#TODO: create fucntions for standardizing figures


###
###    load data
###


# INPUT  FILES
icd_input = os.path.join(ROOT_DATA_DIR, 'input_data', 'input_data_all_icd9_count_subset-2019-01-25.tsv')
cpt_input = os.path.join(ROOT_DATA_DIR, 'input_data','input_data_all_cpt_count_subset-2019-01-26.tsv')
icd_cpt_input = os.path.join(ROOT_DATA_DIR, 'input_data','input_data_all_icd9_cpt_count_subset-2019-01-26.tsv')
# MODEL FILES
icd_model = os.path.join(ROOT_DATA_DIR, 'best_model','best_xgb_model_all_icd9_count_subset-2019-01-25.pickle')
cpt_model = os.path.join(ROOT_DATA_DIR, 'best_model', 'best_xgb_model_all_cpt_count_subset-2019-01-26.pickle')
icd_cpt_model = os.path.join(ROOT_DATA_DIR, 'best_model', 'best_xgb_model_all_icd9_cpt_count_subset-2019-01-26.pickle')
# FEATURE IMPORTANCE FILES
icd_feats = os.path.join(ROOT_DATA_DIR, 'feature_importance', 'w_descrip','descrip__-feature_importance_all_icd9_count_subset-2019-01-25.tsv')
cpt_feats = os.path.join(ROOT_DATA_DIR, 'feature_importance', 'w_descrip','descrip__-feature_importance_all_cpt_count_subset-2019-01-26.tsv')
icd_cpt_feats = os.path.join(ROOT_DATA_DIR, 'feature_importance', 'w_descrip','descrip__-feature_importance_all_icd9_cpt_count_subset-2019-01-26.tsv')


dataset_dict = OrderedDict()
dataset_dict['icd'] =  {'input_file': icd_input, 'model_file': icd_model, 'feat_file': icd_feats}
dataset_dict['cpt'] =   {'input_file': cpt_input, 'model_file': cpt_model, 'feat_file': cpt_feats}
dataset_dict['icd_cpt'] = {'input_file': icd_cpt_input, 'model_file': icd_cpt_model, 'feat_file': icd_cpt_feats}


# output dictionary path with required data....
STORED_DATA_FILE = os.path.join(OUTPUT_DIR,'{}_icd_cpt_datasets_dict.pickle'.format(DATE))



if os.path.isfile(STORED_DATA_FILE):
    print("loading pickled file...")
    metrics_file = open(STORED_DATA_FILE, 'rb')
    dataset_dict = pickle.load(metrics_file)

else:
    print("compiling data...")
    store_results = {}
    for label, inner_dict in dataset_dict.items():
        sstart = time.time()
        print(label)

        X_train, y_train, X_test, y_test, xgb_model, input_data = unpack_input_data(inner_dict['input_file'], inner_dict['model_file'])
        metrics_results, metrics_df, model_params = validate_best_model(xgb_model, X_test, y_test)

        feat_df = pd.read_csv(inner_dict['feat_file'], sep="\t")

        dataset_dict[label]['metrics_df'] = metrics_df
        dataset_dict[label]['metrics_results'] = metrics_results
        dataset_dict[label]['input_df'] = input_data
        dataset_dict[label]['xgb_model'] = xgb_model
        dataset_dict[label]['feat_df'] = feat_df
        print("Finished {} in {:.2} mins".format(label, (time.time()-sstart)/60 ))

    pickle.dump(dataset_dict, open(STORED_DATA_FILE, 'wb'))
    print("saved to...{}".format(OUTPUT_DIR))

# %%
###
###    main
###


# load metrics summary
# append all the evaluation metrics into on df
all_metrics_df = pd.DataFrame()
for label, inner_dict in dataset_dict.items():

    print(label)
    metrics_df = inner_dict['metrics_df']

    metrics_df['dataset'] = label
    all_metrics_df = all_metrics_df.append(metrics_df)


all_metrics_df.dataset = all_metrics_df.dataset.apply(lambda x: x.upper().replace("_", "+"))

# -----------
# plot
# -----------

#### ROC AUC
# todo: resize...
sns.set( style='whitegrid', context='paper',  font_scale=1.0, rc={'figure.figsize':(8,8)} )
sns.set_style( {'axes.grid': True, 'axes.edgecolor': 'k', 'font.sans-serif': ['Arial'], 'grid.color': '#e1e1e1'})
plt.rc('axes', prop_cycle=(cycler('color', ['#1b9e77', '#d95f02', '#7570b3']) + cycler('linestyle', [':', ':', '-'])))
fsize=20
leg_fsize=14




interp_fpr = np.linspace(0, 1, 500)
fig = plt.figure()
for label, inner_dict in dataset_dict.items():
    metrics_results = inner_dict['metrics_results']
    fpr = metrics_results['fpr']
    tpr = metrics_results['tpr']
    auc = metrics_results['roc_auc']

    lin_fx = interp1d(fpr, tpr, kind='linear')
    interp_tpr = lin_fx(interp_fpr)
    _ = plt.plot(interp_fpr, interp_tpr, lw=3.5, alpha=1,
                 label="{} (AUC={:.3f})".format(label.upper().replace("_","+"), auc))


_ = plt.plot([0, 1], [0, 1], linestyle='-', lw=1, color='#CD5C5C', label='Chance', alpha=.8)
_ = plt.xlim([-0.05, 1.05])
_ = plt.ylim([-0.05, 1.05])
_ = plt.xlabel('False Positive Rate', fontsize=fsize)
_ = plt.ylabel('True Positive Rate', fontsize=fsize)
_ = plt.legend(loc="lower right", fontsize=leg_fsize, frameon=True, fancybox=False, framealpha=.9, shadow=True,borderpad=0.5)
_ = plt.tick_params(axis='both', which='major', labelsize=fsize)


roc_fig_file = os.path.join(OUTPUT_DIR, '{}-roc_auc-icd_cpt.pdf'.format(DATE))
# plt.savefig(roc_fig_file)

#### PR AUC
interp_rc = np.linspace(0, 1, 500)
fig = plt.figure()
for label, inner_dict in dataset_dict.items():
    metrics_results = inner_dict['metrics_results']
    pr = metrics_results['pr_curve']
    rc = metrics_results['rc_curve']
    pr_auc = metrics_results['avg_prec']
    pos_prop=0.2

    lin_fx = interp1d(rc, pr, kind='linear')
    interp_pr = lin_fx(interp_rc)
    _ = plt.plot(interp_rc, interp_pr, lw=3.5, alpha=1,
                 label="{} (AUC={:.3f})".format(label.upper().replace("_","+"), pr_auc))

_ = plt.plot([0, 1], [pos_prop, pos_prop], linestyle='-', lw=1,
                 color='#CD5C5C', label='Chance({:.3f})'.format(pos_prop), alpha=.8)

_ = plt.xlim([-0.05, 1.05])
_ = plt.ylim([-0.05, 1.05])
_ = plt.xlabel('Recall', fontsize=fsize)
_ = plt.ylabel('Precision', fontsize=fsize)
_ = plt.legend(loc="lower left", fontsize=leg_fsize, frameon=True, fancybox=False, framealpha=.9, shadow=True,borderpad=0.5)
_ = plt.tick_params(axis='both', which='major', labelsize=fsize)


pr_fig_file = os.path.join(OUTPUT_DIR, '{}-pr_auc-icd_cpt.pdf'.format(DATE))
# plt.savefig(pr_fig_file)

# BRIER SCORE