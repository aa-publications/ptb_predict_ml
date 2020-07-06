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


import xgboost as xgb
from collections import OrderedDict
from datetime import datetime
from scipy import stats




import matplotlib.font_manager as fm
fpath='/dors/capra_lab/users/abraha1/conda/envs/py36_r_ml/lib/python3.6/site-packages/matplotlib/mpl-data/fonts/ttf/Arial.ttf'
bprop = fm.FontProperties(fname=fpath, size=10)
sprop = fm.FontProperties(fname=fpath, size=8)

import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

%matplotlib inline

DATE = datetime.now().strftime('%Y-%m-%d')

###
###    paths
###

sys.path.append('/dors/capra_lab/users/abraha1/projects/PTB_phenotyping/scripts/rand_forest_ptb_classification')
from hyperparam_tune import validate_best_model

sys.path.append("/dors/capra_lab/users/abraha1/projects/PTB_phenotyping/scripts/rand_forest_ptb_classification/manuscript/0_helper_func")
from manip_trained_models_funcs import unpack_input_data, upickle_xgbmodel, extract_train_df, extract_test_df


ROOT_DATA_DIR = "/dors/capra_lab/users/abraha1/projects/PTB_phenotyping/results/ptb_predict_machine_learning/2019-02-02_manuscript_time_to_delivery_icd_cpt/without_age_race_count"
OUTPUT_DIR = "/dors/capra_lab/users/abraha1/projects/PTB_phenotyping/scripts/rand_forest_ptb_classification/manuscript/time_to_delivery/figure"


###
###    functions
###
# In[7]:


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

def plot_roc(store_fpr, store_tpr, aucs, ax, plt_prefix='', roc_fig_file=None):
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


    ax.plot([0, 1], [0, 1], linestyle='--', lw=4, color='r', label='Chance', alpha=.8)

    # plot each cv iteration
    for cv_iter, fpr_tpr_auc in enumerate(zip(store_fpr, store_tpr, aucs)):
        # set_trace()
        fpr, tpr, auc = fpr_tpr_auc
        ax.plot(fpr, tpr, lw=4, alpha=0.9, label="#{}(AUC={:.3f})".format(cv_iter, auc))

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
        ax.plot(interp_fpr, mean_tpr, color='b',
                 label="Mean(AUC={:.2f}+/-{:.2f})".format(mean_auc, std_auc),
                 lw=2, alpha=.8)

        std_tpr = np.std(store_tpr_interp, axis=0)
        tprs_upper = np.minimum(mean_tpr + 2*std_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - 2*std_tpr, 0)
        ax.fill_between(interp_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                         label="+/- 2 S.D.")


    ax.legend(loc="lower right")

    return ax

def plot_pr(precisions, recalls, avg_prs, ax, plt_prefix, pr_fig_file=None, pos_prop=None):
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


    # plot line of random chance
    if pos_prop:
        ax.plot([0, 1], [pos_prop, pos_prop], linestyle='--', lw=4,
                 color='r', label='Chance({:.3f})'.format(pos_prop), alpha=.8)

    # plot each cv_iter
    for cv_iter, pr_rc_avg in enumerate(zip(precisions, recalls, avg_prs)):

        pr_array, rc_array, pr_avg = pr_rc_avg
        # plt.plot(rc_array, pr_array, lw=1, color='k', alpha=0.4)
        ax.step(rc_array, pr_array, lw=4, alpha=0.8, where='post', label="#{}(AvgPR={:.3f})".format(cv_iter, pr_avg))

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

def classifier(feat_df_og, y_test):
    feat_df = feat_df_og.copy()

    # add y label
    feat_df['label_temp'] = y_test
    feat_df['label'] = feat_df.label_temp.apply(lambda x: 'preterm' if x ==1 else 'not-preterm')
    feat_df.drop('label_temp', axis=1, inplace=True)


    feat_df['total_code_count'] = feat_df.sum(1)

    ### set up thresholds
    classify_values = feat_df.total_code_count.values
    num_thresholds = 1000
    tot_thres =  (np.max(classify_values)+1) - np.min(classify_values)
    thresholds = np.arange(np.min(classify_values), (np.max(classify_values)+1), tot_thres/num_thresholds)

    ### calc confusion matrix
    tprs, fprs, prs, rcs = [],[],[],[]
    actual_neg = feat_df.loc[ feat_df['label'] != 'preterm'].shape[0]
    actual_pos = feat_df.loc[ feat_df['label'] == 'preterm'].shape[0]
    for ind, thresh in enumerate(thresholds):
        print(ind) if ((ind %100) == 0)  else None
        # calc TP, TN, FP, FN

        pos_label = 'preterm'

        tp = feat_df.loc[ (feat_df['total_code_count'] >= thresh) & (feat_df['label'] == 'preterm')].shape[0]
        fp = feat_df.loc[ (feat_df['total_code_count'] >= thresh) & (feat_df['label'] != 'preterm')].shape[0]


        fn = feat_df.loc[ (feat_df['total_code_count'] < thresh) & (feat_df['label'] == 'preterm')].shape[0]
        tn = feat_df.loc[ (feat_df['total_code_count'] < thresh) & (feat_df['label'] != 'preterm')].shape[0]


        fpr = fp/(tn+fp+0.001)
        tpr = tp/(tp+fn+0.001)
        pr = tp/(tp+fp+0.001)
        rc = tp/(tp+fn+0.001)


        tprs.append(fpr)
        fprs.append(tpr)
        prs.append(pr)
        rcs.append(rc)

    return  tprs, fprs, prs, rcs, thresholds

def prep_for_auc(fprs, tprs, prs, rcs, thresholds):

    # sort fpr and recall
    sorted_ind = np.argsort(fprs)
    sorted_ind_recall = np.argsort(rcs)

    sorted_fprs = [fprs[x] for x in sorted_ind]
    sorted_tprs = [tprs[x] for x in sorted_ind]
    sorted_recall = [rcs[x] for x in sorted_ind_recall]
    sorted_precision =[prs[x] for x in sorted_ind_recall]
    sorted_thresholds =[thresholds[x] for x in sorted_ind_recall]
    # calc auc
    auc = np.trapz(sorted_tprs,sorted_fprs)
    pr_auc = np.trapz(sorted_precision, sorted_recall)

    return sorted_fprs, sorted_tprs, sorted_recall, sorted_precision, sorted_thresholds

def calc_auc(sorted_tprs, sorted_fprs,sorted_precision, sorted_recall):
    auc = np.trapz(sorted_tprs,sorted_fprs)
    pr_auc = np.trapz(sorted_precision, sorted_recall)

    return auc, pr_auc

def plot_roc(sorted_fprs, sorted_tprs, auc, plt_title, sorted_threshold=None):
    fig, ax = plt.subplots()
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Chance', alpha=.8)
    plt.plot(sorted_fprs, sorted_tprs, lw=3,label='AUC={:.3f}'.format(auc))

    if sorted_threshold:
        norm_thres = sorted_threshold/np.max(sorted_threshold)
        plt.plot(sorted_fprs, norm_thres,  '--k',lw=1, label='threshold')
    _ = ax.set_ylabel('TPR')
    _ = ax.set_xlabel('FPR')
    _ = ax.set_title('ROC: {}'.format(plt_title))
    _ = ax.set(xlim=(0,1), ylim=(0,1))
    plt.legend(loc='best')

    return ax

def plot_pr(sorted_recall, sorted_precision, pr_auc, plt_title,sorted_threshold=None, chance=None):

    fig, ax = plt.subplots()

    plt.plot(sorted_recall, sorted_precision,lw=3, label='AVG-PR={:.3f}'.format(pr_auc))

    if chance:
        plt.plot([0, 1], [chance, chance], linestyle='--', lw=2, color='r', label='Chance(AUC={})'.format(chance), alpha=.8)

    if sorted_threshold:
        norm_thres = sorted_threshold/np.max(sorted_threshold)
        plt.plot(sorted_fprs, norm_thres,  '--k',lw=1, label='threshold')
    _ = ax.set_ylabel('Preicsion')
    _ = ax.set_xlabel('Recall')
    _ = ax.set_title('PR: {}'.format(plt_title))
    _ = ax.set(xlim=(0,1), ylim=(0,1))
    plt.legend(loc='best')

    return ax

def creat_count_df(xgb_model, X_test, y_test):
    bin_X_test = X_test.applymap(lambda x: 1 if x > 1 else 0)

    y_test_proba = xgb_model.predict_proba(X_test)
    y_test_pred = xgb_model.predict(X_test)
    x_test_scount= X_test.sum(1).values
    x_test_uniq_count= bin_X_test.sum(1).values
    log_x_test_scount= bin_X_test.sum(1).values

    cor_df = pd.DataFrame({'y_test':y_test, 'y_test_pred':y_test_pred, 'y_test_proba_1':y_test_proba[:,1],
                          'x_test_scount':x_test_scount, 'x_test_uniq_count':x_test_uniq_count})

    cor_df['log_x_test_scount'] = cor_df.x_test_scount.apply(lambda x: np.log10(x))

    return cor_df

def plot_code_violins(cor_df, y1,y2):
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(8,4))

    sns.violinplot(x="y_test", y=y1, data=cor_df, scale='width', ax=ax[0])
    sns.violinplot(x="y_test", y=y2, data=cor_df, scale='width',ax=ax[1])

    _ , pval = stats.ks_2samp(cor_df.loc[cor_df['y_test']==1,y1],
                   cor_df.loc[cor_df['y_test']==0,y1])


    _ , pval_log = stats.ks_2samp(cor_df.loc[cor_df['y_test']==1,y2],
                   cor_df.loc[cor_df['y_test']==0,y2])

    _ = ax[0].set_xlabel('True Label\n(1=preterm,0=not-preterm)')
    _ = ax[1].set_xlabel('True Label\n(1=preterm,0=not-preterm)')

    ypos_= ax[0].get_ylim()[1] - ax[0].get_ylim()[1]*.05
    ypos_log= ax[1].get_ylim()[1] - ax[1].get_ylim()[1]*.05
    _ = ax[0].text(1.1,ypos_,'K-S\n(p={:.2E})'.format(pval), color='r', size=14)
    _ = ax[1].text(1.1,ypos_log,'K-S\n(p={:.2E})'.format(pval_log), color='r', size=14)
    sns.despine(left=True)
    plt.tight_layout()

    return ax

def get_counts_and_clasif(up_to_dataset_dict, timepoint):


    X_train, y_train, X_test, y_test, xgb_model, input_data = unpack_input_data(up_to_dataset_dict[timepoint]['input_file'], up_to_dataset_dict[timepoint]['model_file'])
    bin_X_test = X_test.applymap(lambda x: 1 if x > 1 else 0)

    cor_df = creat_count_df(xgb_model, X_test, y_test)


    # train classifier
    tprs, fprs, prs, rcs, thresholds = classifier(X_test, y_test)
    sorted_fprs, sorted_tprs, sorted_recall, sorted_precision, sorted_thresholds = prep_for_auc(fprs, tprs, prs, rcs, thresholds)
    auc, pr_auc = calc_auc(sorted_tprs, sorted_fprs,sorted_precision, sorted_recall)
    pr_chance= np.round(np.sum(y_test)/len(y_test),2)

    return cor_df, sorted_fprs, sorted_tprs, sorted_recall, sorted_precision, auc, pr_auc, pr_chance


###
###    main
###

# load all up_to_ file names
up_to_dataset_dict = OrderedDict()
for num_days in ['0','10','90', '273','365']:

    input_file = glob.glob(ROOT_DATA_DIR+'/up_to_*_days/input_data_up_to_{}_days*'.format(num_days))[0]
    model_file = glob.glob(ROOT_DATA_DIR+'/up_to_*_days/best_xgb_model_up_to_{}_days*.pickle'.format(num_days))[0]

    up_to_dataset_dict['{}_days'.format(num_days)] =  {'input_file': input_file, 'model_file': model_file}



# -----------
# timepoint 0 days
# -----------
timepoint='0_days'
t0_cor_df, t0_sorted_fprs, t0_sorted_tprs, t0_sorted_recall, t0_sorted_precision, t0_auc, t0_pr_auc, t0_pr_chance = get_counts_and_clasif(up_to_dataset_dict, timepoint)


# %%
# -----------
# timepoint 90 days
# -----------
timepoint='90_days'
t90_cor_df, t90_sorted_fprs, t90_sorted_tprs, t90_sorted_recall, t90_sorted_precision, t90_auc, t90_pr_auc, t90_pr_chance = get_counts_and_clasif(up_to_dataset_dict, timepoint)


# %%
# -----------
# PLOTS
# -----------

# %%
# plot total code count distribution
sns.set(style="ticks",  font_scale=1.0, rc={"figure.figsize": (8, 4)})
fig, axs = plt.subplots(nrows=1,ncols=2, sharey=True)
hex_codes = [ '#%02x%02x%02x' % tuple((np.array(x)*255).astype(int)) for x in sns.color_palette("Dark2",2)]



sns.violinplot(x="y_test", y="log_x_test_scount", data=t0_cor_df, scale='width',ax=axs[0], color=hex_codes[0]) # log
_ , pval_log = stats.ks_2samp(cor_df.loc[t0_cor_df['y_test']==1, "log_x_test_scount"], t0_cor_df.loc[t0_cor_df['y_test']==0, "log_x_test_scount"])
_ = axs[0].text(1.1,ypos_log,'K-S\n(p={:.2E})'.format(pval_log), color='r', size=14)

sns.violinplot(x="y_test", y="log_x_test_scount", data=t90_cor_df, scale='width',ax=axs[1], color=hex_codes[1]) # log
_ , pval_log = stats.ks_2samp(cor_df.loc[t90_cor_df['y_test']==1, "log_x_test_scount"], t90_cor_df.loc[t90_cor_df['y_test']==0, "log_x_test_scount"])
_ = axs[1].text(1.1,ypos_log,'K-S\n(p={:.2E})'.format(pval_log), color='r', size=14)



_ = axs[0].set_xlabel('', fontproperties=sprop)
_ = axs[1].set_xlabel('', fontproperties=sprop)
_ = axs[0].set_ylabel('Log10 (Total Number of Codes)', fontproperties=sprop)
_ = axs[1].set_ylabel('', fontproperties=sprop)
_ = axs[0].set_ylim(0,3.5)
_ = axs[1].set_ylim(0,3.5)

axs[0].set_xticklabels(['not-preterm','preterm'],fontproperties=sprop)
axs[1].set_xticklabels(['not-preterm','preterm'],fontproperties=sprop)
axs[0].set_yticklabels(np.arange(0,4,0.5),fontproperties=sprop)

sns.despine(ax=axs[0], right=True, top=True, bottom=True, trim=True, offset=1)
sns.despine(ax=axs[1], right=True, top=True, bottom=True, trim=True, offset=1)
# to do: format

# plt.savefig(os.path.join(OUTPUT_DIR, '{}_{}_violin_total_code_dist.png'.format(DATE, timepoint)))

# %%

#
#   ROC
#
sns.set( style='ticks',  font_scale=1.0, rc={'figure.figsize':(3,3)} )
sns.set_style( {'axes.grid': False, 'axes.edgecolor': 'k',  'grid.color': '#e1e1e1'})
hex_codes = [ '#%02x%02x%02x' % tuple((np.array(x)*255).astype(int)) for x in sns.color_palette("Dark2",2)]

fsize=14
leg_fsize=14

# plot
fig ,ax = plt.subplots()


_ = ax.plot(t0_sorted_fprs, t0_sorted_tprs, lw=1, alpha=1, linestyle='-',  color=hex_codes[0], label="0 days ({:.2f})".format(t0_auc))
_ = ax.plot(t90_sorted_fprs, t90_sorted_tprs, lw=1, alpha=1, linestyle='-', color=hex_codes[1],  label="90 days ({:.2f})".format(t90_auc))
_ = ax.plot([0, 1], [0, 1], linestyle='--', lw=1, color='black', label='Chance', alpha=.8)


leg = plt.legend(loc="lower right",   prop=sprop, frameon=True, fancybox=False, framealpha=0.5, shadow=False)
leg.set_title("Days Before Delivery (AUC)", prop =sprop)


ax.spines['left'].set_linewidth(0.5)
ax.spines['bottom'].set_linewidth(0.5)
ax.spines['top'].set_linewidth(0.5)
ax.spines['right'].set_linewidth(0.5)


_ = ax.set_xlabel('False Positive Rate', fontproperties=sprop)
_ = ax.set_ylabel('True Positive Rate', fontproperties=sprop)
_ = ax.set_title('ROC', fontproperties=sprop)


_ = ax.tick_params(axis='both', which='major', width=0.5)
# _ = ax.axis('equal')
_ = ax.set_xlim(0,1.0)
_ = ax.set_ylim(0,1.0)
_ = ax.set_xticks(np.arange(0,1.2,0.2))
_ = ax.set_xticklabels(['{:.1f}'.format(x) for x in np.arange(0,1.2,0.2)], fontproperties=sprop)
_ = ax.set_yticks(np.arange(0,1.2,0.2))
_ = ax.set_yticklabels(['{:.1f}'.format(x) for x in np.arange(0,1.2,0.2)],fontproperties=sprop)

roc_fig_file = os.path.join(OUTPUT_DIR, '{}_roc_w_total_counts.pdf'.format(DATE))
plt.savefig(roc_fig_file,  bbox_inches = 'tight', pad_inches=0.1, transparent=True)



# %%

#
#   PR
#

sns.set( style='ticks',  font_scale=1, rc={'figure.figsize':(3,3)} )
sns.set_style( {'axes.grid': False, 'axes.edgecolor': 'k',  'grid.color': '#e1e1e1'})


fig ,ax = plt.subplots()
_ = ax.plot(t0_sorted_recall, t0_sorted_precision, lw=1, alpha=1, linestyle='-',  color=hex_codes[0], label="0 days ({:.2f})".format(t0_pr_auc))
_ = ax.plot(t90_sorted_recall, t90_sorted_precision, lw=1, alpha=1, linestyle='-', color=hex_codes[1],  label="90 days ({:.2f})".format(t90_pr_auc))
_ = ax.plot([0, 1], [t0_pr_chance, t0_pr_chance], linestyle='--', lw=1, color='black', label='Chance ({:.2f})'.format(t0_pr_chance), alpha=1)

leg = plt.legend(loc="upper right",   prop=sprop, frameon=True, fancybox=False, framealpha=0.5, shadow=False)
leg.set_title("Days Before Delivery (AUC)", prop =sprop)


_ = ax.spines['left'].set_linewidth(0.5)
_ = ax.spines['bottom'].set_linewidth(0.5)
_ = ax.spines['top'].set_linewidth(0.5)
_ = ax.spines['right'].set_linewidth(0.5)


_ = ax.set_xlabel('Recall', fontproperties=sprop)
_ = ax.set_ylabel('Precision', fontproperties=sprop)
_ = ax.set_title('PR', fontproperties=sprop)


_ = ax.tick_params(axis='both', which='major', width=0.5)
# _ = ax.axis('equal')
_ = ax.set_xlim(0,1.0)
_ = ax.set_ylim(0,1.0)
#
_ = ax.set_xticks(np.arange(0,1.2,0.2))
_ = ax.set_xticklabels(['{:.1f}'.format(x) for x in np.arange(0,1.2,0.2)], fontproperties=sprop)
_ = ax.set_yticks(np.arange(0,1.2,0.2))
_ = ax.set_yticklabels(['{:.1f}'.format(x) for x in np.arange(0,1.2,0.2)],fontproperties=sprop)


pr_fig_file = os.path.join(OUTPUT_DIR, '{}_pr_w_total_counts.pdf'.format(DATE))
plt.savefig(pr_fig_file,  bbox_inches = 'tight', pad_inches=0.1, transparent=True)

