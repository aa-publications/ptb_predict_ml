#!/bin/python
# This script will ...
#
#
#
# Abin Abraham
# created on: 2020-03-31 12:51:36



import os, sys
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from glob import glob
import pickle

import matplotlib.font_manager as fm
fpath='/dors/capra_lab/users/abraha1/conda/envs/py36_r_ml/lib/python3.6/site-packages/matplotlib/mpl-data/fonts/ttf/Arial.ttf'

from scipy import stats

sys.path.append('/dors/capra_lab/users/abraha1/projects/PTB_phenotyping/scripts/rand_forest_ptb_classification')
from hyperparam_tune import validate_best_model

sys.path.append("/dors/capra_lab/users/abraha1/projects/PTB_phenotyping/scripts/rand_forest_ptb_classification/manuscript/0_helper_func")
from manip_trained_models_funcs import unpack_input_data, get_preds

import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

%matplotlib inline


from datetime import datetime
DATE = datetime.now().strftime('%Y-%m-%d')


from scipy import stats



# 28-week
# icd-9 and icd-10




###
### paths
###

# ucsf files
UCSF_DIR="/dors/capra_lab/users/abraha1/projects/PTB_phenotyping/scripts/rand_forest_ptb_classification/manuscript/no_twins_analyses/28wk_ucsf_replication/uscf_results"

UCSF_CHECK_SHAP=os.path.join(UCSF_DIR, "shap_raw")
UCSF_SHAP_BY_FEAT=os.path.join(UCSF_DIR, 'shap_by_features')
UCSF_SHAP_ARRAY_DIR=os.path.join(UCSF_DIR, 'no_bias_shap')
UCSF_SHAP_BY_CASES_DIR=os.path.join(UCSF_DIR, 'by_case_control')

uc_shap_pred_file=os.path.join(UCSF_DIR, '28_weeks_icd9_shap_raw_df_2020_06_08.csv') # timpoint, # feature
uc_shap_by_feat_file = os.path.join(UCSF_DIR, '28_weeks_icd9_shap_scores_2020_06_08.tsv') # timpoint, # feature
uc_shap_array_file=os.path.join(UCSF_DIR,'28_weeks_icd9_no_bias_shap_2020_06_08.csv')# timpoint, # feature
uc_shap_ptb_file=os.path.join(UCSF_DIR, '28_weeks_icd9_ptb_shap_2020_06_08.csv')
uc_shap_no_ptb_file=os.path.join(UCSF_DIR, '28_weeks_icd9_not_ptb_shap_2020_06_08.csv')



# vu files
VU_DIR="/dors/capra_lab/users/abraha1/projects/PTB_phenotyping/results/ptb_predict_machine_learning/2020_05_26_time_since_conception_icd9_10_phecode_shared_ucsf_vu_codes_no_twins/icd9_28_weeks_since_concep_shared_codes_no_twins_updated_ega"


# ICD descriptors
DESCRIP_FILE="/dors/capra_lab/users/abraha1/projects/PTB_phenotyping/data/ptb_predict_machine_learning/icd_cpt_descrip_mapping/descrip_master-col_names.txt"
ICD10_DESC_FILE="/dors/capra_lab/users/abraha1/projects/PTB_phenotyping/data/icd_codes_manual/icd10cm_descriptions.tsv"
OUTPUT_DIR="/dors/capra_lab/users/abraha1/projects/PTB_phenotyping/scripts/rand_forest_ptb_classification/manuscript/no_twins_analyses/28wk_ucsf_replication/figures"


timepoints=['28_weeks']

features=['icd9' ]


###
###    functions
###


def get_path_dicts():

    vu_shap_array_files=dict()
    vu_input_files=dict()
    uc_check_shap_files=dict()
    uc_shap_by_feat_files=dict()
    uc_shap_array_files=dict()
    uc_ptb_shap_array_files=dict()
    uc_not_ptb_shap_array_files=dict()

    print("Only loading 28 weeks for icd9 codes...")

    for timepoint in timepoints:
        for feature in features:

            vu_shap_file = os.path.join(VU_DIR, 'test_shap_icd9_28_weeks_since_concep_shared_codes_no_twins_updated_ega-2020-06-01.pickle')
            vu_input_file = os.path.join(VU_DIR, 'input_data_icd9_28_weeks_since_concep_shared_codes_no_twins_updated_ega-2020-06-01.tsv')


            vu_shap_array_files['{}_{}'.format(timepoint, feature)] = vu_shap_file
            vu_input_files['{}_{}'.format(timepoint, feature)] = vu_input_file



            uc_shap_pred_file=os.path.join(UCSF_DIR, '28_weeks_icd9_shap_raw_df_2020_06_08.csv') # timpoint, # feature
            uc_shap_by_feat_file = os.path.join(UCSF_DIR, '28_weeks_icd9_shap_scores_2020_06_08.tsv') # timpoint, # feature
            uc_shap_array_file=os.path.join(UCSF_DIR,'28_weeks_icd9_no_bias_shap_2020_06_08.csv')# timpoint, # feature
            uc_shap_ptb_file=os.path.join(UCSF_DIR, '28_weeks_icd9_ptb_shap_2020_06_08.csv')
            uc_shap_no_ptb_file=os.path.join(UCSF_DIR, '28_weeks_icd9_not_ptb_shap_2020_06_08.csv')


            uc_check_shap_files['{}_{}'.format(timepoint, feature)] = uc_shap_pred_file
            uc_shap_by_feat_files['{}_{}'.format(timepoint, feature)] = uc_shap_by_feat_file
            uc_shap_array_files['{}_{}'.format(timepoint, feature)] = uc_shap_array_file
            uc_ptb_shap_array_files['{}_{}'.format(timepoint, feature)] = uc_shap_ptb_file
            uc_not_ptb_shap_array_files['{}_{}'.format(timepoint, feature)] = uc_shap_no_ptb_file

    return vu_shap_array_files, vu_input_files, uc_check_shap_files, uc_shap_by_feat_files, uc_shap_array_files, uc_ptb_shap_array_files, uc_not_ptb_shap_array_files

def get_mean_abs_shap(uc_shap_df, vu_shap_df):

    # mean of absolute shap value across all women
    uc_abs_shap_mean_df = uc_shap_df.abs().mean(0).reset_index().rename(columns={'index': 'feature', 0:'uc_mean_shap_abs'})
    vu_abs_shap_mean_df = vu_shap_df.abs().mean(0).reset_index().rename(columns={'index': 'feature', 0:'vu_mean_shap_abs'})

    # std
    uc_abs_shap_std_df = uc_shap_df.abs().std(0).reset_index().rename(columns={'index': 'feature', 0:'uc_std_shap_abs'})
    vu_abs_shap_std_df = vu_shap_df.abs().std(0).reset_index().rename(columns={'index': 'feature', 0:'vu_std_shap_abs'})

    # merge vu and ucsf then mean and std
    mean_shap_df = pd.merge(uc_abs_shap_mean_df, vu_abs_shap_mean_df, on='feature', how='inner')
    std_shap_df = pd.merge(uc_abs_shap_std_df, vu_abs_shap_std_df, on='feature', how='inner')
    shap_df = pd.merge(mean_shap_df, std_shap_df, on='feature',how='inner')

    # add rank by dataset
    shap_df.sort_values('uc_mean_shap_abs',ascending=False, inplace=True)
    shap_df['uc_rank'] = np.arange(1, shap_df.shape[0]+1)
    shap_df.sort_values('vu_mean_shap_abs',ascending=False, inplace=True)
    shap_df['vu_rank'] = np.arange(1, shap_df.shape[0]+1)

    return shap_df

def load_shap_df(this_model):
    # load shap array and assign feature names

    vu_shap_array = pickle.load(open(vu_shap_array_files[this_model], 'rb'))
    vu_input_df = pd.read_csv(vu_input_files[this_model],sep="\t")


    feat_cols = vu_input_df.columns.difference(['GRID','label','partition']).values.tolist()
    vu_shap_df = pd.DataFrame(vu_shap_array[:,:-1], columns=feat_cols)

    uc_by_feat_df = pd.read_csv(uc_shap_by_feat_files[this_model],sep="\t")
    uc_shap_df = pd.read_csv(uc_shap_array_files[this_model], sep=",", names=uc_by_feat_df.iloc[:, 0].values.tolist())

    return uc_shap_df, vu_shap_df

def load_shap_by_case_control_df(this_model):

    # load shap array and assign feature names
    vu_shap_array = pickle.load(open(vu_shap_array_files[this_model], 'rb'))
    vu_input_df = pd.read_csv(vu_input_files[this_model],sep="\t")

    vu_input_df.head()
    preterm_bool = (vu_input_df.loc[(vu_input_df['partition']=='held_out')]['label'] =="preterm").tolist()
    not_preterm_bool = (vu_input_df.loc[(vu_input_df['partition']=='held_out')]['label'] !="preterm").tolist()




    feat_cols = vu_input_df.columns.difference(['GRID','label','partition']).values.tolist()
    vu_shap_df = pd.DataFrame(vu_shap_array[:,:-1], columns=feat_cols)

    vu_ptb_shap_df = vu_shap_df[preterm_bool].copy()
    vu_not_ptb_shap_df = vu_shap_df[not_preterm_bool].copy()

    #UCSF
    uc_by_feat_df = pd.read_csv(uc_shap_by_feat_files[this_model],sep="\t")

    uc_ptb_shap_df = pd.read_csv(uc_ptb_shap_array_files[this_model], sep=",", names=uc_by_feat_df.iloc[:, 0].values.tolist())
    uc_not_ptb_shap_df = pd.read_csv(uc_not_ptb_shap_array_files[this_model], sep=",", names=uc_by_feat_df.iloc[:, 0].values.tolist())

    return uc_ptb_shap_df,uc_not_ptb_shap_df, vu_ptb_shap_df, vu_not_ptb_shap_df

def extract_top15(shap_df):

    # -----------
    # compare top 15
    # -----------
    uc_top15_df = shap_df.sort_values('uc_mean_shap_abs', ascending=False)[0:15].copy()
    vu_top15_df = shap_df.sort_values('vu_mean_shap_abs', ascending=False)[0:15].copy()

    # combine shared codes
    temp_shared = pd.merge(uc_top15_df, vu_top15_df.loc[:, ['feature']], on='feature', how='inner' )
    temp_shared['dataset'] = 'shared'

    # uc top15 not in vu
    temp_uc_only = uc_top15_df[~uc_top15_df.feature.isin(vu_top15_df.feature)].copy()
    temp_uc_only['dataset'] = 'ucsf_only'

    # vu top15 not in uc
    temp_vu_only = vu_top15_df[~vu_top15_df.feature.isin(uc_top15_df.feature)].copy()
    temp_vu_only['dataset'] = 'vu_only'


    top15_df = pd.concat([temp_shared, temp_uc_only, temp_vu_only], axis=0)

    return top15_df

def plot_top15(cat_df):
    sns.set(style="ticks", context='paper', font_scale=1.0, rc={ 'figure.dpi':300})
    prop = fm.FontProperties(fname=fpath, size=8)
    fig = plt.figure(figsize=(1.5,6))
    gs = fig.add_gridspec(20, 3, wspace=0, hspace=0.01)

    cum_len = 0
    nrows= cat_df.shape[0]
    for index, row in cat_df.iterrows():
        this_cat = row.cat_name
        print(f"{row.category} ==> {cum_len} to {cum_len+row.category-1}")
        this_ax = fig.add_subplot(gs[cum_len:cum_len+(row.category -1), :])
        ax2 = this_ax.twinx()


        # get data
        plot_df = top15_df.loc[top15_df['category']==this_cat].copy()
        plot_df.sort_values('order', inplace=True, ascending=False)
        plot_df.loc[plot_df['vu_rank']>15, 'vu_rank']=np.nan
        plot_df.loc[plot_df['uc_rank']>15, 'uc_rank']=np.nan

        # plot
        this_ax.scatter(x=plot_df.uc_rank, y=plot_df.manu_descript, marker='o', s=50, facecolors='none', edgecolor='royalblue')
        this_ax.scatter(x=plot_df.vu_rank, y=plot_df.manu_descript, marker='s', s=80, color='goldenrod', facecolors='none', edgecolor='goldenrod')
        this_ax.set_yticklabels(plot_df.descrip_only, fontproperties=prop)

        ax2.scatter(x=plot_df.uc_rank, y=plot_df.feature, marker='o', s=50, facecolors='none', edgecolor='royalblue')
        ax2.scatter(x=plot_df.vu_rank, y=plot_df.feature, marker='s', s=80, color='goldenrod', facecolors='none', edgecolor='goldenrod')

        # draw line connecting both points
        if this_cat == 'shared':
            for iind, rrow in plot_df.iterrows():
                this_ax.plot([rrow.uc_rank, rrow.vu_rank], [rrow.manu_descript, rrow.manu_descript], '-', linewidth=1, color='gray')

        # turn off x axix
        if (index != (nrows-1)):
             this_ax.axes.get_xaxis().set_visible(False)
        else:
            this_ax.set_xticks(np.arange(1,16,2))
            this_ax.tick_params(axis='x', bottom=True)

        # set axis and ticks
        this_ax.set_xlim(0,17)
        sns.despine(ax=this_ax, top=True, bottom=True, left=True, right=True)
        sns.despine(ax=ax2, top=True, bottom=True, left=True, right=True)
        this_ax.tick_params(axis='y',left=False)
        ax2.tick_params(axis='y',right=False)
        this_ax.grid(b=True, which='major', axis='y', linestyle='--')

        cum_len +=row.category

    return fig, gs


# %%
###
###    main
###

# -----------
# load icd description for 9 and 10
# -----------
desc9_df = pd.read_csv(DESCRIP_FILE, sep="\t")
desc10_df = pd.read_csv(ICD10_DESC_FILE, sep="\t", usecols=['code_decimal','short_desc','major'])
desc10_df.rename(columns={'code_decimal':'feature'},inplace=True)

desc9_dict = dict(zip(desc9_df.feature, desc9_df.short_desc))
desc10_dict = dict(zip(desc10_df.feature, desc10_df.short_desc))


# -----------
# load file paths @ 28 weeks
# -----------
# load file paths
vu_shap_array_files, vu_input_files, uc_check_shap_files, uc_shap_by_feat_files, uc_shap_array_files, uc_ptb_shap_array_files, uc_not_ptb_shap_array_files = get_path_dicts()
# TODO: fix error with ICD-10 at 28 weeks


icd9_model='28_weeks_icd9'


# focus on icd9_model
uc_shap_df, vu_shap_df = load_shap_df(icd9_model)
shap_df = get_mean_abs_shap(uc_shap_df, vu_shap_df)


# %%
###
###    correlation between features and SHAP scores
###

# sp_r, sp_p = stats.spearmanr(shap_df.uc_mean_shap_abs, shap_df.vu_mean_shap_abs)
sp_r, sp_p = stats.pearsonr(shap_df.uc_mean_shap_abs, shap_df.vu_mean_shap_abs)

if sp_p == 0:
    sp_p = '<2.2e-308'

sns.set(style="ticks", context='paper', font_scale=1.0, rc={"figure.figsize": (2.3, 2.3), 'figure.dpi':300})
# sns.set(style="ticks", context='paper',  font_scale=1.0, rc={"figure.figsize": (3, 3)})
prop = fm.FontProperties(fname=fpath, size=8)
fig, ax = plt.subplots()
sns.scatterplot(x=shap_df.uc_mean_shap_abs, y=shap_df.vu_mean_shap_abs, ax=ax, size=1, alpha=0.7, **{'edgecolor':'none'})

ax.legend().set_visible(False)
ax.set_xlabel('UCSF SHAP value', fontproperties=prop)
ax.set_ylabel('Vanderbilt SHAP value', fontproperties=prop)
ax.annotate(f"pearson R: {str(sp_r)[0:4]}\np-value: {sp_p:}", xy=(0.02,0.85), xycoords='axes fraction' , fontproperties=prop)
# ax.set_ylim(0,0.05)
ax.set_yticks(np.arange(0,0.30,0.05))
ax.set_yticklabels([f"{x:.2f}" for x in np.arange(0, 0.3, 0.05)], fontproperties=prop)
# ax.set_xlim(0,0.05)
ax.set_xticks(np.arange(0,0.30,0.05))
ax.set_xticklabels([f"{x:.2f}" for x in np.arange(0, 0.3, 0.05)], fontproperties=prop)
# ax.set_aspect('equal')
sns.despine(ax=ax, top=True, right=True, trim=True)

plt.subplots_adjust(left=0.2,right=1, top=0.9, bottom=0.2)
# plt.savefig(os.path.join(OUTPUT_DIR, f'{DATE}_shap_feat_pear_correlation_28wks_icd9.eps'), bbox_inches = None, pad_inches=0, transparent=True)
plt.savefig(os.path.join(OUTPUT_DIR, f'{DATE}_shap_feat_pear_correlation_28wks_icd9.pdf'), bbox_inches = None, pad_inches=0, transparent=True)



# %% - repeat on raw shaw values

# %%
###
###    TOP 15 FEATURES
###


top15_df = extract_top15(shap_df)
top15_df['description'] = top15_df.feature.map(desc9_dict)

top15_df.head()
top15_df[top15_df.feature.isin(temp_feats)]

temp_feats = set(top15_df.feature.unique()) - set(manual_descriptions_dict.keys())

manual_descriptions_dict = {"V22.1": "Supervision of pregnancy (V22.1)",
                            "V22.0": "Supervision of pregnancy (V22.0)",
                            "V28.3": "Routine fetal ultrasound (V28.3)",
                            "655.83": "Known fetal abnormality (655.83)",
                            "651.03": "Twins (651.03)",
                            "V28.89": "Antenatal Screening (V28.89)",
                            "V28.81": "Antenatal Screening (V28.81)",
                            "V76.2": "Screen for cervix malignancy (V76.2)",
                            "V23.41": "History of PTB (V23.41)",
                            "648.03": "Diabetes (648.03)",
                            "V28.9": "Antenatal Screening (V28.9)",
                            "V74.1": "Pulmonary TB screen (V74.1)",
                            "V23.82": "Super. elderly mult. pregnancy (V23.82)",
                            "V23.9": "Super. high-risk pregnancy (V23.9)",
                            "659.63": "Elderly multi-gravid complication (659.63)",
                            "V72.31": "Routine gyn exam (V72.31)",
                            "V77.1": "Diabetes screen (V77.1)",
                            
                            "401.9": "Hypertension (401.9)",
                            "642.03": "Hypertension Antepartum (642.03)",
                            "648.93": "Pregnancy complication (648.93)",
                            "649.73": "Cervical shortening (649.73)",
                            "656.53": "Poor fetal growth (656.35)",
                            
                            
                            
                            "644.03": "Threatened premature labor (644.03)",
                            "282.9": "Hereditary hemolytic anemia (282.9)",
                            "649.13": "Antepartum obesity (649.13)"}

category_dict = {"V22.1": "Pregnancy Supervision",
                "V22.0": "Pregnancy Supervision",
                "V28.3": "Pregnancy Supervision",
                "655.83": "Increase PTB Risk",
                "651.03": "Increase PTB Risk",
                "V28.89": "Screening",
                "V28.81": "Screening",
                "V76.2": "Screening",
                "V23.41": "Increase PTB Risk",
                "648.03": "Increase PTB Risk",
                "V28.9": "Screening",
                "V74.1": "Screening",
                
                "401.9": "Increase PTB Risk",
                "642.03": "Increase PTB Risk",
                "648.93": "Pregnancy Supervision",
                "649.73": "Increase PTB Risk",
                "656.53": "Increase PTB Risk",
                
                "V23.82": "High-risk Pregnancy",
                "V23.9": "High-risk Pregnancy",
                "659.63": "High-risk Pregnancy",
                "V72.31": "Pregnancy Supervision",
                "V77.1": "Screening",
                "644.03": "Increase PTB Risk",
                "282.9": "Increase PTB Risk",
                "649.13": "Increase PTB Risk"}

top15_df['manu_descript'] = top15_df.feature.map(manual_descriptions_dict)
top15_df['category'] = top15_df.feature.map(category_dict)

data_cat_df = pd.value_counts(top15_df.dataset).reset_index()
data_cat_df.columns=['cat_name','category']

top15_df['order'] = top15_df.apply(lambda x: x.uc_rank if (x.uc_rank < x.vu_rank) else x.vu_rank, axis=1)

top15_df['descrip_only'] = top15_df.manu_descript.apply(lambda x: x.split(" (")[0])

type_cat_df = pd.value_counts(top15_df.category).reset_index()
type_cat_df.columns=['cat_name','category']


#write
# top15_df.to_csv(os.path.join(OUTPUT_DIR, 'top15_icd9_28wks_uc_vu_no_twins.tsv'), sep="\t", index=False)


# -----------
# plot
# -----------
# %%

plot_top15(type_cat_df)
plt.savefig(os.path.join(OUTPUT_DIR, '{}_top_15_icd9_28_weeks_ucsf_vu_no_twins.pdf'.format(DATE)), bbox_inches = "tight")




# %%
###
###    shap by ptb vs. not-ptb
###
#
# ptb_shap = pd.read_csv(uc_shap_ptb_file.format('28_weeks','icd9'))
#
#
# this_model='28_weeks_icd9'
# uc_ptb_shap_df, uc_not_ptb_shap_df, vu_ptb_shap_df, vu_not_ptb_shap_df = load_shap_by_case_control_df(this_model)
#
#
# shap_ptb_df =    extract_top15(get_mean_abs_shap(uc_ptb_shap_df, vu_ptb_shap_df))
# shap_no_ptb_df = extract_top15(get_mean_abs_shap(uc_not_ptb_shap_df, vu_not_ptb_shap_df))
#
# shap_ptb_df['description'] = shap_ptb_df.feature.map(desc9_dict)
# shap_no_ptb_df['description'] = shap_no_ptb_df.feature.map(desc9_dict)
#
#
#
# set(shap_no_ptb_df.description.values).difference(set(shap_ptb_df.description.values))
# set(shap_ptb_df.description.values).difference(set(shap_no_ptb_df.description.values))
# set(shap_ptb_df.description.values).intersection(set(shap_no_ptb_df.description.values))
#
#
#
#
# import shap
#
#
#
# shap.decision_plot(0, shap_ptb_df)
#
# # %%
# fig, axs = plt.subplots(nrows=3)
#
#
# for counter, dataset in enumerate(['shared', 'vu_only','ucsf_only']):
#     plot_df = shap_no_ptb_df.loc[shap_no_ptb_df['dataset']==dataset]
#     axs[counter].scatter(x=plot_df.uc_mean_shap_abs, y=plot_df.feature, marker='o', label='ucsf', color='royalblue')
#     axs[counter].scatter(x=plot_df.vu_mean_shap_abs, y=plot_df.feature, marker='o', label='vu', color='goldenrod')
#
#
#
