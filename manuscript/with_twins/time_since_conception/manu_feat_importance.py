#!/bin/python
# This script will ...
#
#
#
# Abin Abraham
# created on: 2020-04-06 09:39:30


import os, sys
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from glob import glob
import pickle

import matplotlib.font_manager as fm
fpath='/dors/capra_lab/users/abraha1/conda/envs/py36_r_ml/lib/python3.6/site-packages/matplotlib/mpl-data/fonts/ttf/Arial.ttf'

sys.path.append('/dors/capra_lab/users/abraha1/projects/PTB_phenotyping/scripts/rand_forest_ptb_classification')
from hyperparam_tune import validate_best_model

sys.path.append("/dors/capra_lab/users/abraha1/projects/PTB_phenotyping/scripts/rand_forest_ptb_classification/manuscript/0_helper_func")
from manip_trained_models_funcs import unpack_input_data, get_preds

%matplotlib inline

from datetime import datetime
DATE = datetime.now().strftime('%Y-%m-%d')


###
###    paths
###
MODEL_DIR="/dors/capra_lab/users/abraha1/projects/PTB_phenotyping/results/ptb_predict_machine_learning/2019_05_20_time_since_conception/up_to_28_weeks_since_preg_start_icd9_cpt_count"
SHAP_FILE=os.path.join(MODEL_DIR, "test_shap_up_to_28_weeks_since_preg_start_icd9_cpt_count-2019-06-19.pickle")
INPUT_FILE=os.path.join(MODEL_DIR, "input_data_up_to_28_weeks_since_preg_start_icd9_cpt_count-2019-06-19.tsv")

DESCRIP_FILE="/dors/capra_lab/users/abraha1/projects/PTB_phenotyping/data/ptb_predict_machine_learning/icd_cpt_descrip_mapping/descrip_master-col_names.txt"

OUTPUT_DIR="/dors/capra_lab/users/abraha1/projects/PTB_phenotyping/scripts/rand_forest_ptb_classification/manuscript/time_since_conception/figures"


###
###    functions
###

def load_shap_df(shap_file, input_file):
    # load shap array and assign feature names

    vu_shap_array = pickle.load(open(shap_file, 'rb'))
    vu_input_df = pd.read_csv(input_file,sep="\t")
    feat_cols = vu_input_df.columns.difference(['GRID','label','partition']).values.tolist()
    vu_shap_df = pd.DataFrame(vu_shap_array[:,:-1], columns=feat_cols)


    return  vu_shap_df

def get_mean_abs_shap(vu_shap_df):

    # mean of absolute shap value across all women
    vu_abs_shap_mean_df = vu_shap_df.abs().mean(0).reset_index().rename(columns={'index': 'feature', 0:'vu_mean_shap_abs'})

    # std
    vu_abs_shap_std_df = vu_shap_df.abs().std(0).reset_index().rename(columns={'index': 'feature', 0:'vu_std_shap_abs'})

    # merge vu mean and std
    shap_df = pd.merge(vu_abs_shap_mean_df, vu_abs_shap_std_df, on='feature',how='inner')

    # add rank by dataset
    shap_df.sort_values('vu_mean_shap_abs',ascending=False, inplace=True)
    shap_df['vu_rank'] = np.arange(1, shap_df.shape[0]+1)

    return shap_df

def extract_top15(shap_df):
    # compare top 15
    vu_top15_df = shap_df.sort_values('vu_mean_shap_abs', ascending=False)[0:15].copy()

    return vu_top15_df

def plot_top15(cat_df, top15_df):

    # plot properties
    sns.set(style="white",  font_scale=1.0, rc={"figure.figsize": (8, 8)})
    prop = fm.FontProperties(fname=fpath, size=12)
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

        # plot
        this_ax.scatter(x=plot_df.vu_rank, y=plot_df.feature, marker='s', s=80, color='goldenrod', facecolors='none', edgecolor='goldenrod')
        this_ax.set_yticklabels(plot_df.manu_descript, fontproperties=prop)
        ax2.scatter(x=plot_df.vu_rank, y=plot_df.feature, marker='s', s=80, color='goldenrod', facecolors='none', edgecolor='goldenrod')



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


# load icd-9 descriptions
desc9_df = pd.read_csv(DESCRIP_FILE, sep="\t")
desc9_dict = dict(zip(desc9_df.feature, desc9_df.short_desc))


vu_shap_df = load_shap_df(SHAP_FILE, INPUT_FILE)
mean_shap_df = get_mean_abs_shap(vu_shap_df)
top15_df = extract_top15(mean_shap_df)
top15_df['description'] = top15_df.feature.map(desc9_dict)
top15_df['order'] = top15_df['vu_rank']

# rename CPT codes ...
cpt_dict= {"88307":"Pathology examination fo tissue usign a microscope, moderately high complexity",
    "76812":"Ultrasound, pregnant uterus, real time with image documentation, fetal and maternal evaluation plus detailed fetal anatomic examination, transabdominal approach; each additional gestation (List separately in addition to code for primary procedure)",
    "76820":"Doppler velocimetry, fetal; umbilical artery",
    "82570":"Creatinine; other source",
    "36415":"Collection of venous blood by venipuncture",
    "76825":"Ultrasound, pregnant uterus, real time with image documentation, follow-up (eg, re-evaluation of fetal size by measuring standard growth parameters and amniotic fluid volume, re-evaluation of organ system(s) suspected or confirmed to be abnormal on a prev"}

top15_df.loc[top15_df.feature.isin(cpt_dict.keys()), 'description'] = top15_df.loc[top15_df.feature.isin(cpt_dict.keys()), 'feature'].map(cpt_dict)

# manual descriptions
manual_descript_dict = {'Supervis normal 1st preg': "Supervision of pregnancy",
    'Supervis oth normal preg': "Supervision of pregnancy",
    'Ultrasound, pregnant uterus, real time with image documentation, follow-up (eg, re-evaluation of fetal size by measuring standard growth parameters and amniotic fluid volume, re-evaluation of organ system(s) suspected or confirmed to be abnormal on a prev': "Pregnancy ultrasound",
    'Initial prenatal care visit (report at first prenatal encounter with health care professional providing obstetrical care. Report also date of visit and, in a separate field, the date of the last menstrual period [LMP]) (Prenatal)':"Prenatal visit",
    'Ultrasound, pregnant uterus, real time with image documentation, follow-up (eg, re-evaluation of fetal size by measuring standard growth parameters and amniotic fluid volume, re-evaluation of organ system(s) suspected or confirmed to be abnormal on a prev': 'Pregnancy ultrasound',
    'Scrn fetal anatmc survey': 'Pregnancy ultrasound',
    'Scr fetl malfrm-ultrasnd': 'Pregnancy ultrasound',
    'Thrt prem labor-antepart': 'Threatened premature labor',
    'Threat labor NEC-antepar': 'Threatened labor',
    'Pathology examination fo tissue usign a microscope, moderately high complexity': 'Pathology of tissue biopsy',
    'Ultrasound, pregnant uterus, real time with image documentation, fetal and maternal evaluation plus detailed fetal anatomic examination, transabdominal approach; each additional gestation (List separately in addition to code for primary procedure)': 'Pregnancy ultrasound',
    'Preg w hx pre-term labor': 'History of PTB',
    'Doppler velocimetry, fetal; umbilical artery':"Umbilical artery doppler",
    'Creatinine; other source': "Creatinine lab",
    'Collection of venous blood by venipuncture': "Venipuncture"}

manual_categories_dict = {'Supervis normal 1st preg': "Pregnancy supervision",
    'Supervis oth normal preg': "Pregnancy supervision",
    'Ultrasound, pregnant uterus, real time with image documentation, follow-up (eg, re-evaluation of fetal size by measuring standard growth parameters and amniotic fluid volume, re-evaluation of organ system(s) suspected or confirmed to be abnormal on a prev': "Pregnancy ultrasound",
    'Initial prenatal care visit (report at first prenatal encounter with health care professional providing obstetrical care. Report also date of visit and, in a separate field, the date of the last menstrual period [LMP]) (Prenatal)':"Pregnancy supervision",
    'Ultrasound, pregnant uterus, real time with image documentation, follow-up (eg, re-evaluation of fetal size by measuring standard growth parameters and amniotic fluid volume, re-evaluation of organ system(s) suspected or confirmed to be abnormal on a prev': 'Pregnancy ultrasound',
    'Scrn fetal anatmc survey': 'Pregnancy ultrasound',
    'Scr fetl malfrm-ultrasnd': 'Pregnancy ultrasound',
    'Thrt prem labor-antepart': 'Preterm risk',
    'Threat labor NEC-antepar': 'Preterm risk',
    'Pathology examination fo tissue usign a microscope, moderately high complexity': 'Other',
    'Ultrasound, pregnant uterus, real time with image documentation, fetal and maternal evaluation plus detailed fetal anatomic examination, transabdominal approach; each additional gestation (List separately in addition to code for primary procedure)': 'Pregnancy ultrasound',
    'Preg w hx pre-term labor': 'Preterm risk',
    'Doppler velocimetry, fetal; umbilical artery':'Pregnancy ultrasound',
    'Creatinine; other source': "Other",
    'Collection of venous blood by venipuncture': "Other"}
top15_df['manu_descript'] = top15_df.description.map(manual_descript_dict)
top15_df['category'] = top15_df.description.map(manual_categories_dict)



# %%
# -----------
# plot
# -----------

type_cat_df = pd.value_counts(top15_df.category).reset_index()
type_cat_df.columns=['cat_name','category']
plot_top15(type_cat_df, top15_df)





plt.savefig(os.path.join(OUTPUT_DIR, f"{DATE}_top15feat_vu_28weeks_icd9_cpt.pdf"), bbox_inches = "tight")