
#!/bin/python
# This script will ...
#
#
#
# Abin Abraham
# created on: 2020-06-09 22:59:06


import os, sys
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import glob
import pickle
import xgboost as xgb
from textwrap import wrap
from scipy import stats
import shap

from datetime import datetime

sys.path.append('/dors/capra_lab/users/abraha1/projects/PTB_phenotyping/scripts/rand_forest_ptb_classification')
from hyperparam_tune import validate_best_model

sys.path.append("/dors/capra_lab/users/abraha1/projects/PTB_phenotyping/scripts/rand_forest_ptb_classification/manuscript/0_helper_func")
from manip_trained_models_funcs import unpack_input_data, get_preds

sys.path.append('/dors/capra_lab/users/abraha1/projects/PTB_phenotyping/scripts/rand_forest_ptb_classification/manuscript/no_twins_analyses')
from helper_manu_figs import set_up_manu_roc, manu_roc_format

sys.path.append('/dors/capra_lab/users/abraha1/projects/PTB_phenotyping/scripts/rand_forest_ptb_classification/shap_feat_importance')
from shaply_funcs import calc_shap,calc_shap_matrix



import matplotlib.font_manager as fm
fpath='/dors/capra_lab/users/abraha1/conda/envs/py36_r_ml/lib/python3.6/site-packages/matplotlib/mpl-data/fonts/ttf/Arial.ttf'
bprop = fm.FontProperties(fname=fpath, size=10)
sprop = fm.FontProperties(fname=fpath, size=8)

import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

DATE = datetime.now().strftime('%Y-%m-%d')

###
### PATHS
###


BY_TYPE_SINCE_CONCEP_DIR="/dors/capra_lab/users/abraha1/projects/PTB_phenotyping/results/ptb_predict_machine_learning/2020_06_03_since_concep_cs_vg_delivery_no_twins"
CSEC_DIR=os.path.join(BY_TYPE_SINCE_CONCEP_DIR, 'since_concep_28_weeks_no_twins_csection')
VG_DIR=os.path.join(BY_TYPE_SINCE_CONCEP_DIR, 'since_concep_28_weeks_no_twins_vaginal_delivery')

# RF_DIR = "/dors/capra_lab/users/abraha1/projects/PTB_phenotyping/results/ptb_predict_machine_learning/2019_05_20_time_since_conception/up_to_28_weeks_since_preg_start_icd9_cpt_count"
# CLIN_RISK_DIR = "/dors/capra_lab/users/abraha1/projects/PTB_phenotyping/results/ptb_predict_machine_learning/2020_03_24_time_since_concep_up_to_28wks_w_riskfx"
OUTPUT_FIG_DIR = "/dors/capra_lab/users/abraha1/projects/PTB_phenotyping/scripts/rand_forest_ptb_classification/manuscript/no_twins_analyses/28wk_csec_vg_delivery/figures"


###
###    functions
###

def check_shap_adds_up(X, y, sk_xgb_model, shap_mat, tol=0.5):

    dtrain = xgb.DMatrix(X, label=y)
    shap_vals= sk_xgb_model.get_booster().predict(dtrain, output_margin=True, pred_contribs=False, pred_interactions=False, validate_features=False)
    return np.all(np.isclose(shap_mat.sum(1),shap_vals, rtol=tol))


# %%

###
### main
###

csec_input_file=os.path.join(CSEC_DIR,'input_data_csection_up_to_28_weeks_since_preg_start_icd9_cpt_no_twins_count-2020-06-04.tsv')
vg_input_file=os.path.join(VG_DIR,'input_data_vaginal_delivery_up_to_28_weeks_since_preg_start_icd9_cpt_no_twins_count-2020-06-04.tsv')

csec_model_file=os.path.join(CSEC_DIR,'best_xgb_model_csection_up_to_28_weeks_since_preg_start_icd9_cpt_no_twins_count-2020-06-04.pickle')
vg_model_file=os.path.join(VG_DIR,'best_xgb_model_vaginal_delivery_up_to_28_weeks_since_preg_start_icd9_cpt_no_twins_count-2020-06-04.pickle')

# load models and input files
csec_X_train, csec_y_train, csec_X_test, csec_y_test, csec_xgb_model, csec_input_data = unpack_input_data(csec_input_file, csec_model_file)
vg_X_train, vg_y_train, vg_X_test, vg_y_test, vg_xgb_model, vg_input_data = unpack_input_data(vg_input_file, vg_model_file)


csec_no_ptb_test_grids = csec_X_test.reset_index().loc[csec_y_test==0, 'GRID'].values
csec_ptb_test_grids = csec_X_test.reset_index().loc[csec_y_test==1, 'GRID'].values
vg_no_ptb_test_grids = vg_X_test.reset_index().loc[vg_y_test==0, 'GRID'].values
vg_ptb_test_grids = vg_X_test.reset_index().loc[vg_y_test==1, 'GRID'].values

# consider working in probability space..
# explainer = shap.TreeExplainer(csec_xgb_model.get_booster(), data=shap.sample(csec_X_train, 100), model_output='probability')
# shap_values = explainer.shap_values(csec_X_train)


# %%

###
###    SHAP
###

csec_shap_mat = calc_shap_matrix(csec_X_test.values, csec_y_test, csec_xgb_model.get_booster())
vg_shap_mat = calc_shap_matrix(vg_X_test.values, vg_y_test, vg_xgb_model.get_booster())

cs_shap_df = pd.DataFrame(csec_shap_mat[:,:-1], columns=csec_X_test.columns, index=csec_X_test.index)
vg_shap_df = pd.DataFrame(vg_shap_mat[:,:-1], columns=vg_X_test.columns, index=vg_X_test.index)

csec_top_feats_df = calc_shap(csec_X_test.values, csec_y_test, csec_xgb_model.get_booster(), csec_X_test, top_n=10)
vg_top_feats_df = calc_shap(vg_X_test.values, vg_y_test, vg_xgb_model.get_booster(), vg_X_test, top_n=10)


# check shap valeus are good
check_shap_adds_up(vg_X_test.values, vg_y_test, vg_xgb_model,vg_shap_mat, tol=0.5)
check_shap_adds_up(csec_X_test.values, csec_y_test, csec_xgb_model,csec_shap_mat, tol=0.5)

# -----------
# get top features
# -----------

csec_top_feats_df['delivery_type'] = 'csec'
vg_top_feats_df['delivery_type'] = 'vg'


csec_top_feats_df.sort_values('mean_abs_shap', ascending=False, inplace=True)
vg_top_feats_df.sort_values('mean_abs_shap', ascending=False, inplace=True)
csec_top_feats_df['rank'] = np.arange(1,csec_top_feats_df.shape[0]+1)
vg_top_feats_df['rank'] = np.arange(1,vg_top_feats_df.shape[0]+1)

merged_df = pd.concat([csec_top_feats_df, vg_top_feats_df])
wide_df = merged_df.pivot(index='long_descrip', columns='delivery_type', values='rank')
wide_df.reset_index(inplace=True)
wide_df['feature'] = wide_df.long_descrip.apply(lambda x: x.split('-')[0])
wide_df['desecrip'] = wide_df.long_descrip.apply(lambda x: '-'.join(x.split('-')[1:]))
wide_df.fillna(0, inplace=True)

merged_df.feature.values
long_names = dict(zip(merged_df.feature , ['\n'.join(wrap(x,70)) for x in merged_df.long_descrip]))
short_names = {'76820': 'Fetal doppler of umbilical artery',
  '0502F': 'Subsequent prenatal care visit (Prenatal)',
  '88175': 'Cytopathology, cervical or vaginal',
  '76805': 'Ultrasound, after first trimester ...',
  '76816': 'Ultrasound, pregnant uterus ...follow-up',
  '36415': 'Venipuncture',
  '76825': 'Fetal cardiovascular echocardiography',
  '0500F': 'Initial prenatal care visit',
  'V22.0': 'Supervis normal 1st preg',
  '86592': 'Syphilis test, non-treponemal antibody',
  'V22.1': 'Supervis oth normal preg',
  '84156': 'Total urine protein',
  '86762': 'Rueblla antibody',
  'V28.81': 'Scrn fetal anatmc survey',
  '86900': 'ABO blood typing'}

# %% compare frequency of top 10 codes in either dataset

csec_freq_df = pd.DataFrame((csec_X_test.loc[:, csec_X_test.columns.isin(merged_df.feature.values)] > 0).sum(0)/csec_X_test.shape[0], columns=['csec_freq']).reset_index()
vg_freq_df = pd.DataFrame((vg_X_test.loc[:, vg_X_test.columns.isin(merged_df.feature.values)] > 0).sum(0)/vg_X_test.shape[0], columns=['vg_freq']).reset_index()
freq_df = pd.merge(csec_freq_df, vg_freq_df, on='index', how='outer').rename({'index':'feature'}, axis=1)
long_freq_df = pd.melt(freq_df, id_vars='feature')

sns.barplot(y='feature', x='value', hue='variable', data=long_freq_df)


# %%
# compare mean count of codes bw/ cs/ and vg

csec_counts_df = csec_X_test.loc[:, csec_X_test.columns.isin(merged_df.feature.values)].reset_index()
long_csec_counts_df = pd.melt(csec_counts_df,id_vars='GRID')
long_csec_counts_df['delivery_type'] = 'csection'

vg_counts_df =  vg_X_test.loc[:, vg_X_test.columns.isin(merged_df.feature.values)].reset_index()
long_vg_counts_df = pd.melt(vg_counts_df,id_vars='GRID')
long_vg_counts_df['delivery_type'] = 'vaginal'

all_counts_df = pd.concat([long_csec_counts_df, long_vg_counts_df], axis=0)
all_counts_df['short_name'] = all_counts_df.variable.map(short_names)
all_counts_df
# t-test
ttest_df = pd.DataFrame()
for feat in all_counts_df.short_name.unique():
    vdist = all_counts_df.loc[ (all_counts_df['short_name']==feat) &
                            (all_counts_df['delivery_type']=='vaginal'),'value'].values
    cdist = all_counts_df.loc[ (all_counts_df['short_name']==feat) &
                            (all_counts_df['delivery_type']=='csection'),'value'].values

    tstat, pval = stats.ttest_ind(vdist,cdist, equal_var = False)
    ttest_df = ttest_df.append(pd.DataFrame({'feat':[feat], 'tstat':[tstat], 'pval':[pval]}))

ttest_df['signif'] = ttest_df.pval < 0.05
ttest_df

# %% - compare odds ratio for each of the top features

for feat in csec_counts_df.columns.difference(['GRID']):
    ptb_counts = csec_counts_df.loc[csec_y_test==1, feat].values
    no_ptb_counts = csec_counts_df.loc[csec_y_test==0, feat].values


vg_counts_df


csec_y_test
vg_y_test

# %%
sns.set(style="whitegrid",  font_scale=1.4, rc={"figure.figsize": (12, 8)})
fig, ax = plt.subplots()
ax = sns.barplot(y='short_name', x='value', hue='delivery_type', data=all_counts_df, ax=ax, palette=['dimgray','lightsteelblue'])
ax.set_xlabel('Mean number of codes in held-out set')
ax.set_ylabel('ICD-9 or CPT')
for feat in ttest_df.loc[ttest_df['signif']==True, 'feat'].values:
    xc = [t.get_text() for t in ax.get_yticklabels()].index(feat)
    x1, x2 = xc-0.35, xc+0.35   # columns 'Sat' and 'Sun' (first column: 0, see plt.xticks())
    hmax = np.max([[p.get_width() for p in ax.patches][0:15][xc], [p.get_width() for p in ax.patches][15:][xc]])+0.25

    y, h, col = hmax, 0.1, 'indianred'
    plt.plot([y, y, y, y], [x1, x1, x2, x2], lw=2.5, c=col)
    plt.text( y+0.05, xc+0.75, "*", ha='center', va='bottom', color=col, fontsize=30)

    ax.get_yticklabels()[xc].set_color("indianred")
plt.savefig(os.path.join(OUTPUT_FIG_DIR, 'mean_num_codes_for_top_feats_in_test_set_csec_vg.pdf'))

# %%


top_cs_shap_df = pd.melt(cs_shap_df.loc[:, cs_shap_df.columns.isin(merged_df.feature.values)].reset_index(), id_vars='GRID', var_name='feature', value_name='shap')
top_vg_shap_df = pd.melt(vg_shap_df.loc[:, vg_shap_df.columns.isin(merged_df.feature.values)].reset_index(), id_vars='GRID', var_name='feature', value_name='shap')
top_cs_shap_df['delivery'] = 'csection'
top_vg_shap_df['delivery'] = 'vaginal'

top_combined_df = pd.concat([top_cs_shap_df, top_vg_shap_df], axis=0)
top_combined_df['descrip'] = top_combined_df.feature.map(short_names)


top_cs_count_df = pd.melt(csec_X_test.loc[:, csec_X_test.columns.isin(merged_df.feature.values)].reset_index(), id_vars='GRID', var_name='feature', value_name='count')
top_vg_count_df = pd.melt(vg_X_test.loc[:, vg_X_test.columns.isin(merged_df.feature.values)].reset_index(), id_vars='GRID', var_name='feature', value_name='count')


# 1 / (1 + np.exp(-vg_shap_df.sum(axis=1)))  # Predictions as probabilities



  # predictions[i] = 1 / (1 + np.exp(-a.sum(axis=1)[0]))  # Predictions as probabilities

# %%
sns.set(style="ticks",  font_scale=1.0, rc={"figure.figsize": (10, 10)})
# sns.boxplot(y='shap', x='feature', hue='delivery', data=combined_df)
fig, ax = plt.subplots()
ax.axvline(0, color='k', lw=1)
ax = sns.boxplot(x='shap', y='descrip', hue='delivery', data=top_combined_df, fliersize=0.8, palette="Set2", ax=ax, showfliers=True, showmeans=False, showcaps=False, width=0.5)
_ = plt.setp(ax.artists, alpha=.9, linewidth=0)

sns.despine(ax=ax, top=True, right=True, left=False)
plt.tight_layout()
# plt.savefig(os.path.join(OUTPUT_FIG_DIR,  f'{DATE}_testset_shap_dist_by_delivery_type_top_10_w_means.pdf'))



# %% shap vs. count
from matplotlib.ticker import MaxNLocator
top_cs_shap_df.head()
top_vg_shap_df
top_cs_count_df.head()
top_vg_count_df

cs_shap_count_df = pd.merge(top_cs_shap_df, top_cs_count_df, on=['GRID','feature'], how='outer')
vg_shap_count_df = pd.merge(top_vg_shap_df, top_vg_count_df, on=['GRID','feature'], how='outer')

cs_shap_count_df['ptb_bool'] = 0
cs_shap_count_df.loc[cs_shap_count_df.GRID.isin(csec_ptb_test_grids), 'ptb_bool'] = 1
vg_shap_count_df['ptb_bool'] = 0
vg_shap_count_df.loc[vg_shap_count_df.GRID.isin(vg_ptb_test_grids), 'ptb_bool'] = 1


vg_shap_count_df.head()
sns.set(style="ticks",  font_scale=1., rc={"figure.figsize": (10, 5)})



for feat in vg_shap_count_df.feature.unique():
    print(feat)
    fig, axs = plt.subplots(ncols=2, sharex=True, sharey=True)

    sns.scatterplot(x='count', y='shap', data= vg_shap_count_df.loc[vg_shap_count_df['feature']==feat], s=120, alpha=0.5, hue='ptb_bool', style='ptb_bool', ax=axs[0])
    sns.regplot(x="count", y="shap",  data=vg_shap_count_df.loc[vg_shap_count_df['feature']==feat], scatter=False, ci=0, ax=axs[0],color='k')
    sns.scatterplot(x='count', y='shap', data= cs_shap_count_df.loc[cs_shap_count_df['feature']==feat], s=120, alpha=0.5, hue='ptb_bool', style='ptb_bool', ax=axs[1])
    sns.regplot(x="count", y="shap",  data=cs_shap_count_df.loc[cs_shap_count_df['feature']==feat], scatter=False, ci=0, ax=axs[1], color='k')
    axs[0].set_title('Vaginal Delivery')
    axs[1].set_title('Csection Delivery')
    [ax.xaxis.set_major_locator(MaxNLocator(integer=True)) for ax in axs]
    [ax.axhline(0, color='k', ls='--') for ax in axs]
    [ax.set_xlabel('Code Count') for ax in axs]

    plt.suptitle(f'{feat}: {short_names[feat]}', fontsize=18)

    plt.savefig(os.path.join(OUTPUT_FIG_DIR, f'{DATE}_{feat}_count_v_shap.pdf'))




#
# # %%
# # split csection by preterm vs. not preterm
# col_indices = np.where(csec_X_test.columns.isin(csec_top_feats_df.feature.values))[0]
# noptb_incides = np.where(csec_y_test==0)[0]
# ptb_incides = np.where(csec_y_test==1)[0]
#
#
# mask_ptb = np.ix_(ptb_incides, col_indices)
# mask_noptb = np.ix_(noptb_incides, col_indices)
#
# ptb_shap_df = pd.DataFrame(csec_shap_mat[mask_ptb], columns=csec_top_feats_df.feature.values, index=csec_input_data.iloc[ptb_incides, 0].values).reset_index()
# noptb_shap_df = pd.DataFrame(csec_shap_mat[mask_noptb], columns=csec_top_feats_df.feature.values, index=csec_input_data.iloc[noptb_incides, 0].values).reset_index()
#
# melted_ptb_df = pd.melt(ptb_shap_df, id_vars='index', var_name='feature', value_name='shap')
# melted_ptb_df['delivery'] = 'preterm'
# melted_noptb_df = pd.melt(noptb_shap_df, id_vars='index', var_name='feature', value_name='shap')
# melted_noptb_df['delivery'] = 'not_preterm'
#
#
# combined_df = pd.concat([melted_ptb_df, melted_noptb_df], axis=0)
# combined_df.rename({'index':'GRID'}, axis=1, inplace=True)
# combined_df.head()
# # %%
# sns.set(style="whitegrid",  font_scale=1.0, rc={"figure.figsize": (24, 8)})
# # sns.boxplot(y='shap', x='feature', hue='delivery', data=combined_df)
# sns.violinplot(y='shap', x='feature', hue='delivery', data=combined_df, split=True, inner="quart", alpha=0.5)
#
# # %%
# sns.set(style="whitegrid",  font_scale=1.0, rc={"figure.figsize": (8, 8)})
# sns.scatterplot(x="csec", y="vg", data=wide_df, s=100)

