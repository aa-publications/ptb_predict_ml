
#!/bin/python
# This script will ...
#
#
#
# Abin Abraham
# created on: 2020-01-22 13:57:57


import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime
from collections import OrderedDict

import pickle
from glob import glob
import matplotlib.pyplot as plt
import seaborn as sns


DATE = datetime.now().strftime('%Y-%m-%d')

sys.path.append('/dors/capra_lab/users/abraha1/projects/PTB_phenotyping/scripts/rand_forest_ptb_classification')
from hyperparam_tune import validate_best_model
from sklearn.metrics import recall_score, precision_score, confusion_matrix, average_precision_score



import matplotlib.font_manager as fm
fpath='/dors/capra_lab/users/abraha1/conda/envs/py36_r_ml/lib/python3.6/site-packages/matplotlib/mpl-data/fonts/ttf/Arial.ttf'
sprop = fm.FontProperties(fname=fpath, size=6)

import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

%matplotlib inline


# -----------
# FUNCTIONS
# -----------

def get_input_and_model(input_file, model_file):



    input_data = pd.read_csv(input_file, sep="\t")
    print("done loading {}".format(os.path.basename(input_file)))

    held_out_df = input_data.loc[input_data['partition']=='held_out'].copy()
    train_df = input_data.loc[input_data['partition']=='grid_cv'].copy()
    held_out_df.set_index('GRID',inplace=True)
    train_df.set_index('GRID',inplace=True)

    X_test = held_out_df.iloc[:,:-2]
    y_test = held_out_df.label.apply(lambda x: 1 if x == 'preterm' else 0).values

    X_train = train_df.iloc[:,:-2]
    y_train = train_df.label.apply(lambda x: 1 if x == 'preterm' else 0).values


    xgb_model = pickle.load(open(model_file, "rb"))

    return X_train, X_test, y_train,  y_test, xgb_model

def get_preds(xgb_model, X_input):

    # use best model to predict on test set
    model_params = xgb_model.get_params()
    y_pred = xgb_model.predict(X_input)
    y_proba = xgb_model.predict_proba(X_input)


    return y_pred, y_proba


def intersect_sptb_grids(X_input, y_input, sptb_grids):

    X_input_cp = X_input.copy()
    y_input_cp = y_input.copy()


    X_input_cp.reset_index(inplace=True)

    keep_sptbs = (X_input_cp.GRID.isin(sptb_grids))
    X_input_cp = X_input_cp[keep_sptbs]
    y_input_cp = y_input_cp[keep_sptbs]

    X_input_cp.set_index('GRID', inplace=True)

    return X_input_cp, y_input_cp


def remove_non_sptb_cases(X_train, y_train, sptb_grids):
    X_input_cp = X_train.copy()
    y_input_cp = y_train.copy()
    X_input_cp.reset_index(inplace=True)


    # out of all positives, keep only
    cases = X_input_cp[y_input_cp==1].copy()
    sptb_cases = cases[cases.GRID.isin(sptb_grids)].copy()
    controls = X_input_cp[y_input_cp==0].copy()

    X_new = pd.concat([sptb_cases, controls]).reset_index(drop=True)
    X_new.set_index('GRID', inplace=True)
    y_new = np.concatenate([np.ones(sptb_cases.shape[0]), np.zeros(controls.shape[0])])

    return X_new, y_new


def get_pr_auc(input_file, model_file, sptb_grids):
    X_train, X_test, y_train,  y_test, xgb_model  = get_input_and_model(input_file, model_file)

    X_train_new, y_train_new = remove_non_sptb_cases(X_train, y_train, sptb_grids)
    X_test_new, y_test_new = remove_non_sptb_cases(X_test, y_test, sptb_grids)

    sptb_train_pred, sptb_train_proba = get_preds(xgb_model, X_train_new)
    sptb_test_pred, sptb_test_proba = get_preds(xgb_model, X_test_new)

    train_prev = np.sum(y_train_new==1)/len(y_train)
    test_prev = np.sum(y_test_new==1)/len(y_train)

    train_prauc = average_precision_score(y_train_new, sptb_train_pred, average=None, pos_label=1)
    test_prauc = average_precision_score(y_test_new, sptb_test_pred, average=None, pos_label=1)

    tn,fp, fn, tp  = confusion_matrix(y_test_new, sptb_test_pred).ravel()

    return train_prauc, test_prauc,train_prev, test_prev, tn,fp, fn, tp

def get_recall(input_file, model_file, sptb_grids):
    print(os.path.basename(input_file))
    X_train, X_test, y_train,  y_test, xgb_model  = get_input_and_model(input_file, model_file)
    X_train_sptb, y_train_sptb = intersect_sptb_grids(X_train, y_train, sptb_grids)
    X_test_sptb, y_test_sptb = intersect_sptb_grids(X_test, y_test, sptb_grids)
    sptb_train_pred, sptb_train_proba = get_preds(xgb_model, X_train_sptb)
    sptb_test_pred, sptb_test_proba = get_preds(xgb_model, X_test_sptb)

    train_rc = recall_score(y_train_sptb, sptb_train_pred)
    test_rc = recall_score(y_test_sptb, sptb_test_pred)

    # ppv will be 100%
    # train_pr = precision_score(y_train_sptb, sptb_train_pred)
    # test_pr = precision_score(y_test_sptb, sptb_test_pred)

    tn,fp, fn, tp  = confusion_matrix(y_test_sptb, sptb_test_pred).ravel()

    return train_rc, test_rc, tn, fp, fn, tp

# -----------
# PATHS
# -----------

sptb_grids_file = "/dors/capra_lab/users/abraha1/projects/PTB_phenotyping/data/spontaneous_ptb/sptb_GRIDS.txt"

output_dir="/dors/capra_lab/users/abraha1/projects/PTB_phenotyping/scripts/rand_forest_ptb_classification/manuscript/no_twins_analyses/28wk_since_concep_w_risk_fx/figures"
# exp_label="since_concep_upto_28wk_no_twins_cmp_riskfx"



ICDCPT_DIR = "/dors/capra_lab/users/abraha1/projects/PTB_phenotyping/results/ptb_predict_machine_learning/2020_06_03_since_conception_icd9_cpt_no_twins_compare_risk_fx/concep_icdcpt_notwins_riskcomp"
CLIN_RISK_DIR = "/dors/capra_lab/users/abraha1/projects/PTB_phenotyping/results/ptb_predict_machine_learning/2020_06_03_since_conception_icd9_cpt_no_twins_compare_risk_fx/concep_icdcpt_notwins_riskcomp_redo"
OUTPUT_FIG_DIR = "/dors/capra_lab/users/abraha1/projects/PTB_phenotyping/scripts/rand_forest_ptb_classification/manuscript/no_twins_analyses/28wk_since_concep_w_risk_fx/figures"


input_file=os.path.join(ICDCPT_DIR,'input_data_up_to_28_weeks_since_preg_start_icd9_cpt_no_twins_count_compare_riskfx-2020-06-03.tsv')
clinrisk_input_file=os.path.join(CLIN_RISK_DIR,'input_data_up_to_28_weeks_since_preg_start_icd9_cpt_no_twins_count_compare_riskfx-2020-06-04.tsv')

model_file=os.path.join(ICDCPT_DIR,'best_xgb_model_up_to_28_weeks_since_preg_start_icd9_cpt_no_twins_count_compare_riskfx-2020-06-03.pickle')
clinrisk_model_file=os.path.join(CLIN_RISK_DIR,'best_xgb_model_up_to_28_weeks_since_preg_start_icd9_cpt_no_twins_count_compare_riskfx-2020-06-04.pickle')



# %%
# -----------
# MAIN
# -----------

# load sPTB grids
with open(sptb_grids_file, 'r') as fo:
    a = fo.readlines()

sptb_grids = [x.splitlines()[0] for x in a][1:]
len(sptb_grids)


# %%

###
### get predcitions
###



# load 28 week model predictions


billing_train_rc, billing_test_rc, billing_tn, billing_fp, billing_fn, billing_tp = get_recall(input_file, model_file, sptb_grids)
clin_train_rc, clin_test_rc, clin_tn, clin_fp, clin_fn, clin_tp = get_recall(clinrisk_input_file, clinrisk_model_file, sptb_grids)


train_prauc, test_prauc,train_prev, test_prev, tn,fp, fn, tp = get_pr_auc(input_file, model_file,sptb_grids)

print("train pr auc:{:.4f} w/ prev: {:.4f}".format(train_prauc, train_prev))
print("test pr auc: {:.4f} w/ prev: {:.4f}".format(test_prauc, test_prev))


# %%
sns.set(style="whitegrid",  font_scale=1.0, rc={"figure.figsize": (1,1)})
fig, ax =plt.subplots()
sns.heatmap([ [tp,fn], [fp, tn]], annot=False,  linewidths=2, square=False, ax=ax, cbar=False)

ax.annotate(tp, (0.5,0.75), color='white', size=10, va='center', ha='center')
ax.annotate(fn, (1.5,0.75), color='white', size=10, va='center', ha='center')
ax.annotate(fp, (0.5,1.25), color='white', size=10, va='center', ha='center')
ax.annotate(tn, (1.5,1.25), color='black', size=10, va='center', ha='center')


# ax.set_yticks([0.25, 1.75])
ax.set_yticklabels(['true\npos','true\nneg'],rotation='horizontal')
ax.set_xticklabels(['pred\npos','pred\nneg'])

# %%
# figure option 1: comapre Recall/Sensitivity

sns.set(style="ticks", context='paper', font_scale=1.0, rc={"figure.figsize": (2.3, 2.3), 'figure.dpi':300})
fig ,ax = plt.subplots()
sns.barplot(x=['Billing Codes','Risk Factors'], y = [billing_test_rc, clin_test_rc], ax=ax, palette=['darkslateblue','dimgray'])
ax.annotate("{}".format(np.round(billing_test_rc,2)), xy=(0, billing_test_rc+0.03), xycoords='data', ha='center', va='center', color='darkslateblue', fontproperties=sprop)
ax.annotate("{}".format(np.round(clin_test_rc,2)), xy=(1, clin_test_rc+0.03), xycoords='data', ha='center', va='center', color='dimgray', fontproperties=sprop)

ax.set_ylabel("Recall", fontproperties=sprop,labelpad=0.59)
ax.set_yticks(np.arange(0,1.25,0.25))
ax.set_yticklabels(np.arange(0,1.25,0.25), fontproperties=sprop)
ax.set_xticklabels(['Billing Codes','Risk Factors'], fontproperties=sprop)

sns.despine(ax=ax, top=True, right=True)
ax.spines['left'].set_linewidth(0.5)
ax.spines['bottom'].set_linewidth(0.5)
ax.tick_params(width=0.5, length=2.5)
sns.despine(ax=ax, top=True, right=True)

plt.subplots_adjust(left=0.15,right=0.95, top=.88, bottom=0.10)
plt.savefig(os.path.join(output_dir, f'{DATE}_sptb_post_hoc_recall.pdf'),  pad_inches=0,  transparent=True)



# %%
# figure option 2: ppv across time...

# load all time points
timepoints = ['13_weeks','28_weeks','32_weeks','35_weeks']

all_recall_df = pd.DataFrame()
for timepoint in timepoints:


    billing_input_file = glob(f"/dors/capra_lab/users/abraha1/projects/PTB_phenotyping/results/ptb_predict_machine_learning/2019_05_20_time_since_conception/up_to_{timepoint}_since_preg_start_icd9_cpt_count/input_data_*.tsv")[0]
    billing_model_file = glob(f"/dors/capra_lab/users/abraha1/projects/PTB_phenotyping/results/ptb_predict_machine_learning/2019_05_20_time_since_conception/up_to_{timepoint}_since_preg_start_icd9_cpt_count/best_xgb_*.pickle")[0]


    clin_train_rc, clin_test_rc, clin_tn, clin_fp, clin_fn, clin_tp = get_recall(billing_input_file, billing_model_file, sptb_grids)

    all_recall_df = all_recall_df.append(pd.DataFrame({'timepoint':[timepoint], 'billing_train_rc':[clin_train_rc], 'billing_test_rc':[clin_test_rc], 'test_tn':[clin_tn],'test_fp':[clin_fp],'test_fn':[clin_fn],'test_tp':[clin_tp]}))

all_recall_df.reset_index(drop=True,inplace=True)

# %% plot

sns.set(style="ticks",  font_scale=1.0, rc={"figure.figsize": (1.5,1.5)})
fig ,ax = plt.subplots()
# sns.barplot(x=['Billing Codes','Risk Factors'], y = [billing_test_rc, clin_test_rc], ax=ax, palette=['darkslateblue','gray'])
sns.barplot(x="timepoint", y="billing_test_rc", data=all_recall_df, ax=ax, color='darkslateblue')

xticklabels_w_n = []
for ind, row in all_recall_df.iterrows():

    ax.annotate("{}".format(np.round(row.billing_test_rc,2)), xy=(ind, row.billing_test_rc+0.03), xycoords='data', ha='center', va='center', fontproperties=sprop)
    templabel = row.timepoint.split("_")[0] + "\nn={}".format(row.test_fn+row.test_tp)
    xticklabels_w_n.append(templabel)

ax.set_ylabel("Recall", fontproperties=bprop)
ax.set_yticks(np.arange(0,1.5,0.5))
ax.set_yticklabels(np.arange(0,1.5,0.5), fontproperties=sprop)
ax.set_xticklabels(xticklabels_w_n, fontproperties=sprop)
ax.set_xlabel("Weeks since conception", fontproperties=bprop)

sns.despine(ax=ax, top=True, right=True)
# plt.savefig(os.path.join(output_dir, f'{DATE}_sptb_post_hoc_recall_many_timpepoint.pdf'), bbox_inches = "tight")


