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

from collections import OrderedDict
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

# PATHS
# ROOT_DATA_DIR = "/dors/capra_lab/users/abraha1/projects/PTB_phenotyping/results/ptb_predict_machine_learning/2019-01-21_manuscript_add_ehr_data"
# BIN_CLIN_DATA_DIR = "/dors/capra_lab/users/abraha1/projects/PTB_phenotyping/results/ptb_predict_machine_learning/2019_09_05_add_clin_labs/bin_labs"
# CLIN_DATA_DIR = "/dors/capra_lab/users/abraha1/projects/PTB_phenotyping/results/ptb_predict_machine_learning/2019_09_05_add_clin_labs/lab_values"
OUTPUT_DIR = "/dors/capra_lab/users/abraha1/projects/PTB_phenotyping/scripts/rand_forest_ptb_classification/manuscript/no_twins_analyses/add_ehr_at_28wks"

ADD_EHR_DIR="/dors/capra_lab/users/abraha1/projects/PTB_phenotyping/results/ptb_predict_machine_learning/2020_06_05_since_concep_add_ehr_no_twins_v1"

DATE = datetime.now().strftime('%Y-%m-%d')

# -----------
# Function
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

def autolabel(rects, ax, fprop=None):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{:.2}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontproperties=fprop)


###
###    main
###
# %%
# set up path
dataset_dict = OrderedDict()
for dlabel in ['age_race', 'prs','unstruc', 'clin_lab']:


    for flabel in ['all','icd_cpt', dlabel]:


        input_file = glob.glob(ADD_EHR_DIR+'/vs_{}__{}_no_twins_v1/input_data_vs_{}__{}*'.format(dlabel, flabel, dlabel, flabel) )[0]
        model_file = glob.glob(ADD_EHR_DIR+'/vs_{}__{}_no_twins_v1/best_xgb_model_vs_{}__{}*'.format(dlabel, flabel, dlabel, flabel))[0]

        dataset_dict['{}_{}'.format(dlabel,flabel)] =  {'input_file': input_file, 'model_file': model_file}


# %%
# -----------
# Main
# -----------

store_results = {}
for label, inner_dict in dataset_dict.items():

    print(label)
    X_test, y_test, xgb_model  = get_test_performance(inner_dict['input_file'], inner_dict['model_file'])
    metrics_results, metrics_df, model_params = validate_best_model(xgb_model, X_test, y_test)

    dataset_dict[label]['metrics_df'] = metrics_df
    dataset_dict[label]['metrics_results'] = metrics_results
    dataset_dict[label]['y_test'] = y_test



#  save stored dictionary

store_perf_metrics_file =os.path.join(OUTPUT_DIR, '28_wk_add_ehr_no_twins.pickle')
if not os.path.exists(store_perf_metrics_file):
    pickle.dump(dataset_dict, open(store_perf_metrics_file, 'wb'))
    print("pickled model.")


# concat metrics_df
all_metrics_df = pd.DataFrame()
for label, inner_dict in dataset_dict.items():

    print(label)
    metrics_df = inner_dict['metrics_df']
    metrics_df['dataset'] = label
    all_metrics_df = all_metrics_df.append(metrics_df)





all_metrics_df['model']='feat_only'
all_metrics_df.loc[all_metrics_df['dataset'].apply(lambda x: True if (x.find('all') > -1) else False), 'model'] = 'all'
all_metrics_df.loc[all_metrics_df['dataset'].apply(lambda x: True if (x.find('icd_cpt') > -1) else False), 'model'] = 'baseline'



all_metrics_df.loc[all_metrics_df.dataset.apply(lambda x: True if x.startswith('age_race') else False ), 'datatype'] = 'age_race'
all_metrics_df.loc[all_metrics_df.dataset.apply(lambda x: True if x.startswith('prs') else False ), 'datatype'] = 'prs'
all_metrics_df.loc[all_metrics_df.dataset.apply(lambda x: True if x.startswith('unstruc') else False ), 'datatype'] = 'unstruc'
all_metrics_df.loc[all_metrics_df.dataset.apply(lambda x: True if x.startswith('clin_lab') else False ), 'datatype'] = 'clin_lab'


all_metrics_df['random_pr'] = (all_metrics_df.tp_count + all_metrics_df.fn_count)/all_metrics_df.total_count


# # new lables
# mod_labels = {'age_race_all':'ICD+CPT+age_race', 'age_race_icd_cpt':'ICD+CPT', 'age_race_only':'age_race',
#               'obnotes_all':'ICD+CPT+obnotes', 'obnotes_icd_cpt':'ICD+CPT', 'obnotes_only':'obnotes',
#               'prs_all':'ICD+CPT+prs', 'prs_icd_cpt':'ICD+CPT', 'prs_only':'prs',
#               'unstruc_all':'ICD+CPT+unstruc', 'unstruc_icd_cpt':'ICD+CPT', 'unstruc_only':'unstruc',
#               'bin_labs_all':'ICD+CPT+binary_labs', 'bin_labs_icd_cpt':'ICD+CPT', 'bin_labs_only':'binary_labs',
#               'clin_labs_all':'ICD+CPT+labs', 'clin_labs_icd_cpt':'ICD+CPT', 'clin_labs_only':'labs'
#              }


###
### :subsection title
###

rand_pr_dict = dict(zip(all_metrics_df.dataset, all_metrics_df.random_pr))

all_metrics_df.head()

raw_melted_metrics_df = pd.melt(all_metrics_df, id_vars=['model','dataset','datatype'], value_vars=['roc_auc','avg_pr'] )
raw_melted_metrics_df.variable = raw_melted_metrics_df.variable.str.upper()


# udpate random expected value
random_df = raw_melted_metrics_df.copy()
random_df.loc[random_df['variable']=='ROC_AUC', 'value'] = 0.50
random_df.loc[random_df['variable']=='AVG_PR', 'value'] = random_df.loc[random_df['variable']=='AVG_PR', 'dataset'].map(rand_pr_dict)



# %%
###
### PLOTS
###

# set up
fpath='/dors/capra_lab/users/abraha1/conda/envs/py36_r_ml/lib/python3.6/site-packages/matplotlib/mpl-data/fonts/ttf/Arial.ttf'
sprop = fm.FontProperties(fname=fpath, size=8)
ssprop = fm.FontProperties(fname=fpath, size=6)



# %%
# set up fonts


# specify plt positions
bottom = 0.05
height = 0.9
width = 0.15  # * 4 = 0.6 - minus the 0.1 padding 0.3 left for space
left1, left2, left3, left4, left5 = 0.05, 0.25, 1 - 0.25 - width, 1 - 0.05 - width, 1 - 0.05 - width


rectangle1 = [left1, bottom, width, height]
rectangle2 = [left2, bottom, width, height]
rectangle3 = [left3, bottom, width, height]
rectangle4 = [left4, bottom, width, height]
rectangle5 = [left5, bottom, width, height]


# set_style()
melted_metrics_df = raw_melted_metrics_df.copy()
metric_type = melted_metrics_df.variable.unique().tolist()
ehr_feature_labels = ['age_race', 'unstruc', 'clin_lab', 'prs',]
clean_ehr_feature_labels = ['Age & Race', 'Clinical Keywords', 'Clinical Labs', 'Genetic Risk',]


# order and position of bars
x_model_labels =  ['feat_only','baseline','all']
x_coords = [0, 0.75, 1.5]

# the width of the bars
width = 0.55




fig, axs = plt.subplots(nrows=2, ncols=4, figsize=(4.5, 2.5), sharex=False, sharey=True)
for row in range(len(metric_type)):
    for col in range(len(ehr_feature_labels)):

        col_name = ehr_feature_labels[col]
        models = melted_metrics_df.loc[  (melted_metrics_df['variable'] == metric_type[row]) & (melted_metrics_df['datatype'] == ehr_feature_labels[col])].copy()

        melted_metrics_df

        # get y_values
        y_vals = []
        for this_model in x_model_labels:
            y_vals.append(models.loc[models['model']==this_model, 'value'].values.tolist()[0])

        this_ax = axs[row, col]
        this_ax.tick_params(width=0.5, length=2.5)
#         plt.axes(rectangle1)

        # format
        this_ax.grid(False)
        this_ax.spines['right'].set_visible(False)
        this_ax.spines['top'].set_visible(False)

        # draw bars
        rects1 = this_ax.bar(x_coords, y_vals, width)
        rects2 = this_ax.bar(x_coords[0], y_vals[0], width, color='#DC1A8E')
        rects3 = this_ax.bar(x_coords[1], y_vals[1], width, color='#7570b3')
        rects4 = this_ax.bar(x_coords[2], y_vals[2], width, color='#58595B')
        _ = this_ax.set_ylim(0,1)

        # add y-labels
        autolabel(rects1, this_ax, ssprop)

        if row == 1:
            _ = this_ax.set_xticks(x_coords)
            _ = this_ax.set_xticklabels([clean_ehr_feature_labels[col], 'ICD&CPT', 'Both'], fontproperties=sprop, rotation = 330, ha="left", va='top')
            _ = this_ax.axhline(y=rand_pr_dict[col_name+"_all"], color='black', ls='--', lw=0.8)
            _ = this_ax.tick_params(axis='x', which='major', length=0, pad=4)
        else:
            _ = this_ax.axhline(y=0.5, color='black', ls='--', lw=0.8)
            _ = this_ax.set_xticklabels([''])
            _ = this_ax.tick_params(axis='x', which='major', length=0)




        if col == 0:
            _ = this_ax.set_yticks(np.arange(0,1.25,0.25))
            ylabels_ = ['{:.1f}'.format(x) if not ((x==0.25) or (x==0.75)) else '' for x in np.arange(0,1.25,0.25) ]
            _ = this_ax.set_yticklabels(ylabels_, fontproperties=sprop)
            _ = this_ax.tick_params(axis='y', direction='out', left=True)

        if ((row == 0) and (col == 0)):

            this_ax.set_ylabel('ROC AUC', fontproperties=sprop, labelpad=0)
        if ((row == 1) and (col ==0)):
            this_ax.set_ylabel('PR AUC', fontproperties=sprop, labelpad=0)


        this_ax.spines['left'].set_linewidth(0.5)
        this_ax.spines['bottom'].set_linewidth(0.5)

        # this_ax.spines['top'].set_linewidth(0.5)
        # this_ax.spines['right'].set_linewidth(0.5)
        #
# fig.tight_layout(h_pad=3, w_pad=2)

# plt.savefig(os.path.join(OUTPUT_DIR, '{}_add_ehr_barplot.pdf'.format(DATE)))
plt.savefig(os.path.join(OUTPUT_DIR, f'{DATE}_add_ehr_barplot.pdf'),   pad_inches=0,  transparent=True)

# Make nicer plot for manuscript





# # %%
# # 4.5 x 2.5
# raw_melted_metrics_df.datatype.nunique()
#
# sns.set(style="ticks",  font_scale=1.0, rc={"figure.figsize": (4.5, 2.5)})
# fig, axs = plt.subplots(nrows=2, ncols=4,sharey=True)
#
# axs[0,3]
# axs
#
# for row_ind, variable in enumerate(['ROC_AUC', "AVG_PR"]):
#     for col_ind, datatype in enumerate(['age_race', 'prs', 'unstruc', 'clin_lab']):
#
#         temp_df = raw_melted_metrics_df.loc[(raw_melted_metrics_df['variable'] == variable) & (raw_melted_metrics_df['datatype']==datatype)].copy()
#         ordered_y = [ temp_df.loc[temp_df['model']==x, 'value'].values[0] for x in ['feat_only', 'baseline', 'all']]
#         ax = axs[row_ind, col_ind]
#         ax.bar(x=np.arange(0,3), height=ordered_y)
#         if (col_ind == 0):
#             ax.set_ylim(0,1)
#             ax.set_yticks(np.arange(0,1.25,0.25))
#             ylabels_ = ['{:.1f}'.format(x) if not ((x==0.25) or (x==0.75)) else '' for x in np.arange(0,1.25,0.25) ]
#             ax.set_yticklabels(ylabels_, fontproperties=sprop)
#
#
#
#         if (row_ind ==1):
#             ax.set_xticks(np.arange(0,3))
#             ax.set_xticklabels(['{:.1f}'.format(x) for x in np.arange(0,3)], fontproperties=sprop)
#         else:
#             ax.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
#
#         sns.despine(ax=ax, right=True, top=True)
        ##


