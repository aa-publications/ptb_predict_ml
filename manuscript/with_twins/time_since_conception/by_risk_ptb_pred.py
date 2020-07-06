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

RESULTS_DIR="/dors/capra_lab/users/abraha1/projects/PTB_phenotyping/results/ptb_predict_machine_learning/2019_05_20_time_since_conception/up_to_28_weeks_since_preg_start_icd9_cpt_count"
RISK_DIR="/dors/capra_lab/users/abraha1/projects/PTB_phenotyping/data/ptb_predict_machine_learning/quantify_high_risk_ptb"
DEMO_FILE="/dors/capra_lab/users/abraha1/projects/PTB_phenotyping/data/ptb_predict_machine_learning/demographics/complete_demographics.tsv"

OUTPUT_DIR="/dors/capra_lab/users/abraha1/projects/PTB_phenotyping/scripts/rand_forest_ptb_classification/manuscript/time_since_conception/figures"

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




timepoint='28_weeks'
input_file = os.path.join(RESULTS_DIR, f'input_data_up_to_{timepoint}_since_preg_start_icd9_cpt_count-2019-06-19.tsv')
model_file = os.path.join(RESULTS_DIR, f'best_xgb_model_up_to_{timepoint}_since_preg_start_icd9_cpt_count-2019-06-19.pickle')



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


###
###    MAIN
###


# %%
# get model predictions

X_train, y_train, X_test, y_test, xgb_model, input_data = unpack_input_data(input_file, model_file)
metrics_results, metrics_df, model_params = validate_best_model(xgb_model, X_test, y_test)
y_pred, y_proba = get_preds(xgb_model, X_test)

pred_df = pd.DataFrame({'GRIDS':X_test.index.tolist(), 'y_true':y_test, 'y_pred':y_pred, 'y_proba':y_proba[:,1]})



# %% - distribution of predicted probabilties
fit, ax = plt.subplots()
sns.set(style="whitegrid",  font_scale=1.0, rc={"figure.figsize": (6,6)})
sns.distplot(pred_df.query("y_pred==0").y_proba, ax=ax, label="not-preterm")
sns.distplot(pred_df.query("y_pred==1").y_proba, ax=ax, label="preterm")
ax.legend()


# plt.savefig(os.path.join(OUTPUT_DIR, f'{DATE}_ptb_proba_dist_for_28wk_since_concep.pdf'))

# %%
###
###    load risk fx
###

long_risk_df = harmonize_risk_fx(risk_file_dict, risk_cols_to_keep_dict)
long_risk_df['RISK_CAT_LABEL'] = long_risk_df.RISK_CATEGORY +", "+ long_risk_df.RISK_LABEL
wide_risk_df = long_risk_df.pivot(index='GRID', columns='RISK_LABEL', values='RISK_CATEGORY')

risk_cols = long_risk_df.RISK_LABEL.unique()


# wide_risk_df.to_csv(os.path.join(OUTPUT_DIR, 'risk_fx_matrix_w_na.tsv'), sep="\t", index=False, na_rep="NaN")

# %%
###
###    merge prediction data with risk fx data
###

all_counts_df, all_merged_df = count_merged_df(wide_risk_df, input_data, risk_cols)
train_counts_df, train_merged_df = count_merged_df(wide_risk_df, X_train.reset_index(), risk_cols)
test_counts_df, test_merged_df = count_merged_df(wide_risk_df, X_test.reset_index(), risk_cols)



# add count of total individual in train or test set
train_counts_df = train_counts_df.append(pd.DataFrame(dict(zip(train_counts_df.columns, ['total_X_train']+list(np.ones(train_counts_df.shape[1]-1)*X_train.shape[0]))), index=[1]))
test_counts_df = test_counts_df.append(pd.DataFrame(dict(zip(test_counts_df.columns, ['total_X_test']+list(np.ones(test_counts_df.shape[1]-1)*X_test.shape[0]))), index=[1]))

# train_counts_df.to_csv(os.path.join(OUTPUT_DIR, 'risk_fx_counts_train_28_weeks_since_concep.tsv'), sep="\t", index=False, na_rep="NA")
# test_counts_df.to_csv(os.path.join(OUTPUT_DIR, 'risk_fx_counts_test_28_weeks_since_concep.tsv'), sep="\t", index=False, na_rep="NA")

# %%
# -----------
###    convert NaN to 'Low-Risk'
# -----------

all_merged_df.columns.name=''
low_risk_default_df = all_merged_df.fillna('Low-risk')



###
###    calc risk (OR) in dataset ...
###

calc_or_by_risk(input_data, risk_cols, low_risk_default_df)
or_held_out_df = calc_or_by_risk(input_data.query('partition=="held_out"'), risk_cols, low_risk_default_df)
or_grid_cv_df = calc_or_by_risk(input_data.query('partition=="grid_cv"'), risk_cols, low_risk_default_df)


# or_held_out_df.to_csv(os.path.join(OUTPUT_DIR, 'or_held_out_for_risk_fx_28_wk_since_concep.tsv'), sep="\t", index=False, float_format="%.2f")
# or_grid_cv_df.to_csv(os.path.join(OUTPUT_DIR, 'or_grid_cv_for_risk_fx_28_wk_since_concep.tsv'), sep="\t", index=False,float_format="%.2f")


# calc by number of risk factors
low_risk_default_df['num_risk_fx'] = (low_risk_default_df.loc[:, low_risk_default_df.columns.difference(['GRID'])] == 'High-risk').sum(1)

ptb_df = input_data.loc[:,['GRID','label']].copy()
ptb_df['delivery_type'] = ptb_df['label'].apply(lambda x: 1 if x == 'preterm' else 0)
label_risk_df = pd.merge(ptb_df[['GRID','delivery_type']], low_risk_default_df.loc[:,['GRID','num_risk_fx']], on='GRID', how='outer')

cross_df = pd.crosstab(label_risk_df['delivery_type'], label_risk_df['num_risk_fx'])
cross_df = cross_df.append(pd.DataFrame(cross_df.loc[1,:]/cross_df.sum(0)).transpose())
cross_df.index =['not-ptb','ptb', 'ptb_percent']

# write
# cross_df.to_csv(os.path.join(OUTPUT_DIR, 'ptb_prev_by_num_risk_28wk_since_concep.tsv'), sep="\t", index=False)


# run logistic regression
X=sm.add_constant(label_risk_df.num_risk_fx.values)
y=label_risk_df.delivery_type.values
logit_model=sm.Logit(y,X)

result=logit_model.fit()
print(result.summary())




###
###    model predictino stratified by risk status
###

held_out_ptb_prev =pred_df.y_true.sum()/pred_df.shape[0]

# %%
# -----------
# high vs. low risk
# -----------

all_metrics_df = pd.DataFrame()
for risk_fx in risk_cols:


    high_risk, low_risk = low_risk_default_df.groupby(risk_fx)['GRID'].apply(lambda x: list(x))
    map_risk_fx = dict(zip(low_risk_default_df.GRID, low_risk_default_df[risk_fx]))

    # map risk column to prediction
    risk_pred_df = pred_df.copy()
    risk_pred_df['risk'] = risk_pred_df.GRIDS.map(map_risk_fx)


    for risk_status in ['Low-risk','High-risk']:

        strat_risk_df = risk_pred_df.loc[risk_pred_df['risk']==risk_status]
        ppv = metrics.precision_score(strat_risk_df['y_true'], strat_risk_df['y_pred'])
        npv = calc_npv(strat_risk_df['y_true'], strat_risk_df['y_pred'])

        n_total = strat_risk_df['y_true'].shape[0]
        ptb_prev = np.sum(strat_risk_df['y_true'])/n_total

        all_metrics_df = all_metrics_df.append(pd.DataFrame({'risk_fx':[risk_fx], 'risk_status':[risk_status], 'ppv':[ppv], 'npv':[npv], 'ptb_prev':[ptb_prev], 'n_total':[n_total]  }))



all_metrics_df

# %%
# -----------
# ppv and npv for each risk status
# -----------
# fig, ax = plt.subplots()

sns.set(style="whitegrid",  font_scale=1.2)
melt_metrics_df = pd.melt(all_metrics_df, id_vars=['risk_fx','risk_status'], value_vars=['ppv','npv','ptb_prev','n_total'])


plot_df = melt_metrics_df.loc[melt_metrics_df['variable'].isin(['ppv','ptb_prev'])]
g = sns.catplot(x="risk_status", y="value", hue="variable", col="risk_fx", data=plot_df,
                 kind="point", palette="muted", col_wrap=3, height=3, sharex=False, legend=True, legend_out = True)

g.set_ylabels("")
g.set_titles("{col_name}")
g.despine(left=True)
axes = g.axes
axes[-3].set_xlabel("")
axes[-2].set_xlabel("")
axes[-1].set_xlabel("")

# g.add_legend( loc=1, title='')
g._legend.set_title("")


for ax in axes:
    ax.axhline(held_out_ptb_prev, color="black", lw=3)
    ax.set_title(ax.get_title(), color='indianred')

for ind, riskfx in enumerate(all_metrics_df.risk_fx.unique()):

    low_n, high_n = all_metrics_df.loc[(all_metrics_df['risk_fx'] == riskfx), 'n_total' ]

    if ind > 5:
        axes[ind].set_xticklabels([f'LowRisk\nn={low_n:,}', f'HighRisk\nn={high_n:,}'])
    else:
        axes[ind].set_xticklabels([f'n={low_n:,}', f'n={high_n:,}'])


plt.subplots_adjust(hspace=0.7)
# plt.savefig(os.path.join(OUTPUT_DIR,f"{DATE}_risk_stratified_ppv.pdf"))


# %%
# -----------
# ppv by by number of risk fx
# -----------
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

# all_metrics_risk_num_df
# %% -- plot groupbed barplot

mult=1
sns.set(style="ticks", context='paper', font_scale=1.0, rc={"figure.figsize": (2.3, 2.3), 'figure.dpi':300})
fig, ax  = plt.subplots()

sns.barplot(x='risk_status', y="ppv", data=all_metrics_risk_num_df, palette=sns.light_palette("darkslateblue", reverse=False), ax=ax, linewidth=0.5, edgecolor="black")
sns.barplot(x='risk_status', y="ptb_prev", data=all_metrics_risk_num_df, color='gray', ax=ax, linewidth=0.5, edgecolor='black')
ax.axhline(held_out_ptb_prev, color="indianred", lw=1, linestyle=':', label=f"Pop. PTB Prevalence ({held_out_ptb_prev*100:.0f}%)")

sns.despine(ax=ax, top=True, right=True)

sprop = fm.FontProperties(fname=fpath, size=8)
ax.set_ylabel("PPV", fontproperties=bprop)
ax.set_yticks(np.arange(0,1.5,0.5))
ax.set_yticklabels(np.arange(0,1.5,0.5), fontproperties=sprop)
# ax.set_xticklabels(xticklabels_w_n, fontproperties=sprop)
a = ax.set_xticklabels(['0','1','2','3','4+'], fontproperties=sprop)
ax.set_xlabel("Number of Risk Factors", fontproperties=bprop)


ax.spines['left'].set_linewidth(0.5)
ax.spines['bottom'].set_linewidth(0.5)
ax.tick_params(width=0.5, length=2.5)

counter = 0
for ind, row in all_metrics_risk_num_df.iterrows():
    ax.annotate("{:.2}".format(row.ppv), xy=(counter, row.ppv), ha='center', va='bottom', fontproperties=sprop)
    counter += 1

# plt.subplots_adjust(left=0.2,right=0.9, top=0.95, bottom=0.25)
plt.savefig(os.path.join(OUTPUT_DIR, f'{DATE}_post_hoc_ppv_by_num_risk_fx_icd9_cpt_28wks.pdf'), bbox_inches = 'tight', pad_inches=0, transparent=True)


# %% -- plot dumbell plot

sns.set(style="whitegrid",  font_scale=1.3, rc={"figure.figsize": (12, 8)})
fig, ax  = plt.subplots()


ax.scatter(x=all_metrics_risk_num_df.risk_status, y=all_metrics_risk_num_df.ppv, marker='o', color="royalblue", label='PPV')
ax.scatter(x=all_metrics_risk_num_df.risk_status, y=all_metrics_risk_num_df.ptb_prev, marker='o', color="gray", label='Stratified PTB Prevalence')
ax.axhline(held_out_ptb_prev, color="indianred", lw=1, label=f"Pop. PTB Prevalence ({held_out_ptb_prev*100:.0f}%)")

xlabels_ =[]
# add vertical liens
for ind, row in all_metrics_risk_num_df.iterrows():
    ax.vlines(x=row.risk_status, ymin=row.ptb_prev, ymax=row.ppv, color='black', alpha=1, linewidth=1, linestyles='dotted')
    xlabels_.append(f"{int(row.risk_status)}\n(n={int(row.n_total)})")




sns.despine(ax=ax, top=True, right=True, left=True)
ax.grid(which='major', axis='x', linestyle='')
ax.grid(which='major', axis='y', linestyle='--')
ax.set_xlabel('Number of Risk Factors', labelpad=23)
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
ax.set_title("PPV of PTB Prediction based on number of risk factors.")


labels = [item.get_text() for item in ax.get_xticklabels()]
ax.set_xticks(np.arange(6))
ax.set_xticklabels(xlabels_)


plt.tight_layout()
# plt.savefig(os.path.join(OUTPUT_DIR,f"{DATE}_stratified_by_num_risk_ppv.pdf"))


# %% -- plot npv, dumbell plot
# -----------
# npv
# -----------


all_metrics_risk_num_df['no_ptb_prev'] = 1-all_metrics_risk_num_df.ptb_prev


sns.set(style="whitegrid",  font_scale=1.3, rc={"figure.figsize": (12, 8)})
fig, ax  = plt.subplots()


ax.scatter(x=all_metrics_risk_num_df.risk_status, y=all_metrics_risk_num_df.npv, marker='o', color="royalblue", label='NPV')
ax.scatter(x=all_metrics_risk_num_df.risk_status, y=all_metrics_risk_num_df.no_ptb_prev, marker='o', color="gray", label='Stratified not-PTB Prevalence')
ax.axhline(1-held_out_ptb_prev, color="indianred", lw=1, label=f"Pop. Not-PTB Prevalence ({(1-held_out_ptb_prev)*100:.0f}%)")

sns.despine(ax=ax, top=True, right=True, left=True)

ax.grid(which='major', axis='x', linestyle='')
ax.grid(which='major', axis='y', linestyle='--')

ax.set_xlabel('Number of Risk Factors', labelpad=23)
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
ax.set_title("NPV of PTB Prediction based on number of risk factors.")



xlabels_ =[]
for ind, row in all_metrics_risk_num_df.iterrows():
    ax.vlines(x=row.risk_status, ymin=row.no_ptb_prev, ymax=row.npv, color='black', alpha=1, linewidth=1, linestyles='dotted')
    xlabels_.append(f"{int(row.risk_status)}\n(n={int(row.n_total)})")


labels = [item.get_text() for item in ax.get_xticklabels()]

_ = ax.set_ylim(0,1.05)
_ = ax.set_xticks(np.arange(6))
_ = ax.set_xticklabels(xlabels_)

plt.tight_layout()
# plt.savefig(os.path.join(OUTPUT_DIR,f"{DATE}_stratified_by_num_risk_npv.pdf"))


