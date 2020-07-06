#!/bin/python
# This script will ...
#
#
#
# Abin Abraham
# created on: 2020-03-20 14:16:03



import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime

import matplotlib.pyplot as plt
import seaborn as sns

import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

%matplotlib inline

import statsmodels.api as sm
# from scipy.stats import chi2_contingency
from sklearn import metrics

DATE = datetime.now().strftime('%Y-%m-%d')

# sys.path.append('/dors/capra_lab/users/abraha1/projects/PTB_phenotyping/scripts/rand_forest_ptb_classification')
# from hyperparam_tune import validate_best_model

sys.path.append("/dors/capra_lab/users/abraha1/projects/PTB_phenotyping/scripts/rand_forest_ptb_classification/manuscript/0_helper_func")
from manip_trained_models_funcs import unpack_input_data, upickle_xgbmodel, extract_train_df, extract_test_df, validate_best_model, get_preds

import matplotlib.font_manager as fm
fpath='/dors/capra_lab/users/abraha1/conda/envs/py36_r_ml/lib/python3.6/site-packages/matplotlib/mpl-data/fonts/ttf/Arial.ttf'
bprop = fm.FontProperties(fname=fpath, size=10)
sprop = fm.FontProperties(fname=fpath, size=8)



###
###    PATHS
###



# RESULTS_DIR="/dors/capra_lab/users/abraha1/projects/PTB_phenotyping/results/ptb_predict_machine_learning/2019_05_20_time_since_conception/up_to_28_weeks_since_preg_start_icd9_cpt_count"
RESULTS_DIR="/dors/capra_lab/users/abraha1/projects/PTB_phenotyping/results/ptb_predict_machine_learning/2020_06_03_since_conception_icd9_cpt_no_twins_compare_risk_fx/concep_icdcpt_notwins_riskcomp"
RISK_DIR="/dors/capra_lab/users/abraha1/projects/PTB_phenotyping/data/ptb_predict_machine_learning/quantify_high_risk_ptb"
DEMO_FILE="/dors/capra_lab/users/abraha1/projects/PTB_phenotyping/data/ptb_predict_machine_learning/demographics/complete_demographics.tsv"
OUTPUT_DIR="/dors/capra_lab/users/abraha1/projects/PTB_phenotyping/scripts/rand_forest_ptb_classification/manuscript/no_twins_analyses/28wk_since_concep_w_risk_fx/figures"
exp_label="since_concep_upto_28wk_no_twins_cmp_riskfx"



risk_file_dict={'prepreg_bmi': os.path.join(RISK_DIR, 'nine_mo_before_delivery_earliest_bmi_df.tsv'),
               'prepreg_bp':os.path.join(RISK_DIR, 'nine_mo_before_delivery_earliest_bp_df.tsv'),
               'diabetes':os.path.join(RISK_DIR, 'diabetes_icd9_before_delivery.tsv'),
               'fetal_abnl':os.path.join(RISK_DIR, 'fetal_abnl_icd9_before_delivery.tsv'),
               'sickle_cell':os.path.join(RISK_DIR, 'sickle_cell_icd9_before_delivery.tsv'),
               'age_at_delivery':os.path.join(RISK_DIR, 'years_at_delivery_matrix.tsv'),
               'race':DEMO_FILE}

risk_cols_to_keep_dict={'prepreg_bmi':  ['GRID','BMI_CLEAN','STATUS'],
                       'prepreg_bp': ['GRID','BLOOD_PRESSURE','STATUS'],
                       'diabetes': ['GRID','ICD_count'],
                       'fetal_abnl': ['GRID','ICD_count'],
                       'sickle_cell': ['GRID','ICD_count'],
                       'age_at_delivery': ['GRID','years_at_delivery'],
                       'race': ['GRID','RACE_LIST']}

input_file=os.path.join(RESULTS_DIR,'input_data_up_to_28_weeks_since_preg_start_icd9_cpt_no_twins_count_compare_riskfx-2020-06-03.tsv')
model_file=os.path.join(RESULTS_DIR,'best_xgb_model_up_to_28_weeks_since_preg_start_icd9_cpt_no_twins_count_compare_riskfx-2020-06-03.pickle')

###
###    FUNCTION
###


def safe_divide(x, y):

    try:
        return (x/y)
    except ZeroDivisionError:
        return np.nan

def harmonize_risk_fx(risk_file_dict, risk_cols_to_keep_dict):

    long_risk_df = pd.DataFrame()

    label='prepreg_bmi'
    bmi_df = pd.read_csv(risk_file_dict[label], sep="\t", usecols=risk_cols_to_keep_dict[label])
    bmi_df.rename(columns={'BMI_CLEAN':'RISK_VALUE','STATUS':'SPECIFIC_RISK'}, inplace=True)


    risk_cat_map = {'not-obese':'Low-risk', 'obese':'High-risk','severe-obese':'High-risk', 'more-obese':'High-risk'}
    bmi_df['RISK_CATEGORY'] = bmi_df.SPECIFIC_RISK.map(risk_cat_map)
    bmi_df['RISK_LABEL'] = label
    long_risk_df = long_risk_df.append(bmi_df, sort=True)


    label='prepreg_bp'
    bmi_df = pd.read_csv(risk_file_dict[label], sep="\t", usecols=risk_cols_to_keep_dict[label])
    bmi_df.rename(columns={'BLOOD_PRESSURE':'RISK_VALUE','STATUS':'SPECIFIC_RISK'}, inplace=True)
    risk_cat_map={'hypertensive':'High-risk', 'pre-hypertensive':'High-risk', 'normal':'Low-risk', 'hypotensive':'Low-risk'}
    bmi_df['RISK_CATEGORY'] = bmi_df.SPECIFIC_RISK.map(risk_cat_map)
    bmi_df['RISK_LABEL'] = label
    long_risk_df = long_risk_df.append(bmi_df, sort=True)



    for label in ['diabetes','fetal_abnl','sickle_cell']:
        bmi_df = pd.read_csv(risk_file_dict[label], sep="\t", usecols=risk_cols_to_keep_dict[label])
        bmi_df.rename(columns={'ICD_count':'RISK_VALUE'}, inplace=True)
        bmi_df['SPECIFIC_RISK'] = 'NONE'
        bmi_df['RISK_CATEGORY'] = 'High-risk'
        bmi_df['RISK_LABEL'] = label
        long_risk_df = long_risk_df.append(bmi_df, sort=True)

    label='age_at_delivery'
    bmi_df = pd.read_csv(risk_file_dict[label], sep="\t", usecols=risk_cols_to_keep_dict[label])
    bmi_df.rename(columns={'years_at_delivery':'RISK_VALUE'}, inplace=True)
    bmi_df['SPECIFIC_RISK'] = 'NONE'
    bmi_df['RISK_CATEGORY'] = 'Low-risk'
    bmi_df['RISK_LABEL'] = label
    bmi_df.loc[ (bmi_df['RISK_VALUE'] > 34) | (bmi_df['RISK_VALUE'] < 18), 'RISK_CATEGORY' ] = 'High-risk'
    long_risk_df = long_risk_df.append(bmi_df, sort=True)


    label='race'
    bmi_df = pd.read_csv(risk_file_dict[label], sep="\t", usecols=risk_cols_to_keep_dict[label])
    bmi_df.rename(columns={'RACE_LIST':'RISK_VALUE'}, inplace=True)
    bmi_df['SPECIFIC_RISK'] = 'NONE'

    highrisk_races = ['AFRICAN_AMERICAN', 'HISPANIC', 'ASIAN']

    # seperate by race
    for race in highrisk_races:

        bmi_df['RISK_CATEGORY'] = 'Low-risk'
        bmi_df.loc[bmi_df['RISK_VALUE'].isin([race]), 'RISK_CATEGORY' ] = 'High-risk'
        bmi_df['RISK_LABEL'] = race
        long_risk_df = long_risk_df.append(bmi_df, sort=True)


    return long_risk_df

def count_merged_df(wide_risk_df, input_data, risk_cols):
    temp_wide_df = wide_risk_df.reset_index().copy()
    merged_df = temp_wide_df[temp_wide_df.GRID.isin(input_data.GRID)].reset_index(drop=True)



    for ind, x in enumerate(risk_cols):
        temp_df = pd.value_counts(merged_df[x], dropna=False).reset_index()

        if ind ==0:
            all_counts_df=temp_df
        else:
            all_counts_df = pd.merge(all_counts_df, temp_df, on='index', how='outer')


    return all_counts_df, merged_df

def calc_or_by_risk(input_data, risk_cols, low_risk_default_df):

    or_df = pd.DataFrame()
    for riskfx in risk_cols:

        ptb_df = input_data.loc[:,['GRID','label']].copy()
        ptb_df['delivery_type'] = ptb_df['label'].apply(lambda x: 'preterm' if x == 'preterm' else 'not-preterm')

        label_risk_df = pd.merge(ptb_df[['GRID','delivery_type']], low_risk_default_df.loc[:,['GRID',riskfx]], on='GRID', how='outer')

        table = sm.stats.Table.from_data(label_risk_df[['delivery_type',riskfx]])
        chi_indep_rslt = table.test_nominal_association()

        colnames =['High-risk_not-preterm', 'Low-risk_not-preterm', 'High-risk_preterm', 'Low-risk_preterm']
        counts_df = pd.DataFrame(dict(zip(colnames, table.table_orig.values.reshape(1,4).flatten())), index=[0])


        # calc
        pvalue = chi_indep_rslt.pvalue
        or_ptb = 1/table.local_oddsratios.loc['not-preterm', 'High-risk']

        temp_df = pd.DataFrame({'riskfx':[riskfx], 'oddsratio':[or_ptb], 'pvalue':[pvalue]})
        or_df = or_df.append(pd.concat([temp_df,counts_df], axis=1))


    return or_df

def calc_npv(y_true, y_pred):
    tn, fp, fn, tp = metrics.confusion_matrix(y_true, y_pred).ravel()
    return tn/(tn+fn)



# -----------
# MAIN
# -----------

# %%
###
###   get model predictions
###
X_train, y_train, X_test, y_test, xgb_model, input_data = unpack_input_data(input_file, model_file)
metrics_results, metrics_df, model_params = validate_best_model(xgb_model, X_test, y_test)
y_pred, y_proba = get_preds(xgb_model, X_test)

pred_df = pd.DataFrame({'GRIDS':X_test.index.tolist(), 'y_true':y_test, 'y_pred':y_pred, 'y_proba':y_proba[:,1]})

# %%
###
###    load risk fx
###

long_risk_df = harmonize_risk_fx(risk_file_dict, risk_cols_to_keep_dict)
long_risk_df['RISK_CAT_LABEL'] = long_risk_df.RISK_CATEGORY +", "+ long_risk_df.RISK_LABEL
wide_risk_df = long_risk_df.pivot(index='GRID', columns='RISK_LABEL', values='RISK_CATEGORY')
risk_cols = long_risk_df.RISK_LABEL.unique()

# %%
###
###    merge prediction data with risk fx data
###

all_counts_df, all_merged_df = count_merged_df(wide_risk_df, input_data, risk_cols)
train_counts_df, train_merged_df = count_merged_df(wide_risk_df, X_train.reset_index(), risk_cols)
test_counts_df, test_merged_df = count_merged_df(wide_risk_df, X_test.reset_index(), risk_cols)


# %%
###
###    convert NaN to 'Low-Risk'
###
all_merged_df.columns.name=''
low_risk_default_df = all_merged_df.fillna('Low-risk')

# calc by number of risk factors
low_risk_default_df['num_risk_fx'] = (low_risk_default_df.loc[:, low_risk_default_df.columns.difference(['GRID'])] == 'High-risk').sum(1)



# %%
###
###    model predictino stratified by risk status
###


# -----------
# ppv by by number of risk fx
# -----------
held_out_ptb_prev =pred_df.y_true.sum()/pred_df.shape[0]
num_risk_pred_df = pred_df.copy()
map_num_risk_fx = dict(zip(low_risk_default_df.GRID, low_risk_default_df['num_risk_fx']))


# map risk column to prediction
num_risk_pred_df = pred_df.copy()
num_risk_pred_df['og_num_risk'] = num_risk_pred_df.GRIDS.map(map_num_risk_fx)

# since 5 risk fx only has 3, combine with 4
pd.value_counts(num_risk_pred_df.og_num_risk)
num_risk_pred_df['risk'] = num_risk_pred_df.og_num_risk.apply(lambda x: str(x) if x <= 3 else '4+' )

# for # of risk factors, calc metrics
all_metrics_risk_num_df=pd.DataFrame()
for risk_status in np.sort(num_risk_pred_df.risk.unique()):

    strat_risk_df = num_risk_pred_df.loc[num_risk_pred_df['risk']==risk_status]
    ppv = metrics.precision_score(strat_risk_df['y_true'], strat_risk_df['y_pred'])
    npv = calc_npv(strat_risk_df['y_true'], strat_risk_df['y_pred'])

    n_total = strat_risk_df['y_true'].shape[0]
    ptb_prev = np.sum(strat_risk_df['y_true'])/n_total

    all_metrics_risk_num_df = all_metrics_risk_num_df.append(pd.DataFrame({'risk_status':[risk_status], 'ppv':[ppv], 'npv':[npv], 'ptb_prev':[ptb_prev], 'n_total':[n_total]  }))

#
#
#

# %% -- plot groupbed barplot
sns.set( style='ticks',  font_scale=1.0, rc={'figure.figsize':(2.25,2.25)} )
sns.set_style( {'axes.grid': False, 'axes.edgecolor': 'k', 'axes.linewidth': 10,  'grid.color': '#e1e1e1'})
sprop = fm.FontProperties(fname=fpath, size=6, )

fig, ax  = plt.subplots()

sns.barplot(x='risk_status', y="ppv", data=all_metrics_risk_num_df, color="darkslateblue", ax=ax, linewidth=0.5, edgecolor="darkslateblue")
sns.barplot(x='risk_status', y="ptb_prev", data=all_metrics_risk_num_df, color='mediumslateblue', alpha=0.5, ax=ax, linewidth=0.5, edgecolor='slateblue')


sns.despine(ax=ax, top=True, right=True)

sprop = fm.FontProperties(fname=fpath, size=6)
ax.set_ylabel("PPV", fontproperties=sprop, labelpad=0.59)
ax.set_yticks(np.arange(0,1.25,0.25))
ax.set_yticklabels(np.arange(0,1.25,0.25), fontproperties=sprop)
a = ax.set_xticklabels(['0','1','2','3','4+'], fontproperties=sprop)
ax.set_xlabel("Number of Risk Factors", fontproperties=sprop,labelpad=0.59)


ax.spines['left'].set_linewidth(0.5)
ax.spines['bottom'].set_linewidth(0.5)
ax.tick_params(width=0.5, length=2.5)

counter = 0
for ind, row in all_metrics_risk_num_df.iterrows():
    ax.annotate("{:.2}".format(row.ppv), xy=(counter, row.ppv), ha='center', va='bottom',color='darkslateblue', fontproperties=sprop)
    counter += 1

ax.set_aspect('auto',None)
plt.subplots_adjust(left=0.15,right=0.95, top=.9, bottom=0.10)
plt.savefig(os.path.join(OUTPUT_DIR, f'{DATE}_post_hoc_ppv_by_num_riskfx_{exp_label}_.pdf'),  pad_inches=0, transparent=True)



# # %%
# ###
# ###    calc risk (OR) in dataset ...
# ###
#
# calc_or_by_risk(input_data, risk_cols, low_risk_default_df)
# or_held_out_df = calc_or_by_risk(input_data.query('partition=="held_out"'), risk_cols, low_risk_default_df)
# or_grid_cv_df = calc_or_by_risk(input_data.query('partition=="grid_cv"'), risk_cols, low_risk_default_df)
#
# # TODO
# # or_held_out_df.to_csv(os.path.join(OUTPUT_DIR, 'or_held_out_for_risk_fx_28_wk_since_concep.tsv'), sep="\t", index=False, float_format="%.2f")
# # or_grid_cv_df.to_csv(os.path.join(OUTPUT_DIR, 'or_grid_cv_for_risk_fx_28_wk_since_concep.tsv'), sep="\t", index=False,float_format="%.2f")
#
#
# # calc by number of risk factors
# low_risk_default_df['num_risk_fx'] = (low_risk_default_df.loc[:, low_risk_default_df.columns.difference(['GRID'])] == 'High-risk').sum(1)
#
# ptb_df = input_data.loc[:,['GRID','label']].copy()
# ptb_df['delivery_type'] = ptb_df['label'].apply(lambda x: 1 if x == 'preterm' else 0)
# label_risk_df = pd.merge(ptb_df[['GRID','delivery_type']], low_risk_default_df.loc[:,['GRID','num_risk_fx']], on='GRID', how='outer')
#
# cross_df = pd.crosstab(label_risk_df['delivery_type'], label_risk_df['num_risk_fx'])
# cross_df = cross_df.append(pd.DataFrame(cross_df.loc[1,:]/cross_df.sum(0)).transpose())
# cross_df.index =['not-ptb','ptb', 'ptb_percent']
#
# # write
# # cross_df.to_csv(os.path.join(OUTPUT_DIR, 'ptb_prev_by_num_risk_28wk_since_concep.tsv'), sep="\t", index=False)
