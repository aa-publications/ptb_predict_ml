#!/usr/bin/env python
# coding: utf-8

# # Combine EHR + Demographics + PRS + OB Notes

# Evaluating sample overlap when combining different sources of data.

# In[1]:


import os, sys
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pickle

sys.path.append("/dors/capra_lab/users/abraha1/bin/python_modules/pyvenn")
import venn
from collections import OrderedDict

sys.path.append( '/dors/capra_lab/users/abraha1/projects/PTB_phenotyping/scripts/rand_forest_ptb_classification')
from train_test_rf import load_labels

sys.path.append('/dors/capra_lab/users/abraha1/projects/PTB_phenotyping/scripts/rand_forest_ptb_classification/to_dmatrix')
from create_dmatrix_func import convert_to_xgbDataset


# OUTPUT_DIR = "/dors/capra_lab/users/abraha1/projects/PTB_phenotyping/data/ptb_predict_machine_learning/feature_matrices/no_twins/add_ehr_since_preg_start_28wks"
OUTPUT_DIR = "/dors/capra_lab/users/abraha1/projects/PTB_phenotyping/data/ptb_predict_machine_learning/feature_matrices/no_twins/add_ehr_since_preg_start_28wks_v1"
root_path="/dors/capra_lab/users/"
DELIVERY_LABELS_FILE = os.path.join(root_path,"abraha1/projects/PTB_phenotyping/data/combined_cohorts_07_12_2018/full_dataset_characterization/est_delivery_date_at_least_one_icd_cpt_ega.tsv")
DEMOGRAPHICS_FILE = os.path.join(root_path,"abraha1/projects/PTB_phenotyping/data/ptb_predict_machine_learning/demographics/complete_demographics.tsv")

short_hand_files = {'GRID_PRS.tsv':os.path.join(root_path,"abraha1/projects/PTB_phenotyping/scripts/polygenic_risk_scores/zhang_with_covars/GRID_PRS.tsv"),
                    'years_at_delivery_matrix.tsv':"/dors/capra_lab/users/abraha1/projects/PTB_phenotyping/data/ptb_predict_machine_learning/demographics/years_at_delivery_matrix.tsv",
                    'demographics_matrix.tsv':os.path.join(root_path,"abraha1/projects/PTB_phenotyping/data/ptb_predict_machine_learning/demographics/demographics_matrix.tsv"),
                    'unstruct_':"/dors/capra_lab/users/abraha1/projects/PTB_phenotyping/data/ptb_predict_machine_learning/feature_matrices/ob_notes_unstruct/since_concep_upto_28_weeks/filtered_cui_counts_ob_notes_since_concep_up_to_28wk_feat_mat.tsv",
                    'struct':"/dors/capra_lab/users/abraha1/projects/PTB_phenotyping/data/ptb_predict_machine_learning/ob_notes_variables/ob_notes_struc_since_concep_up_to_28wk.tsv",
                    'clin_labs':"/dors/capra_lab/users/abraha1/projects/PTB_phenotyping/data/ptb_predict_machine_learning/feature_matrices/clincal_labs/no_twins_since_concep_upto_28wk/first_preg_all_stats_labs_feat_mat.tsv"}



# load feature file from 28 weeks 
# checked that there is no twins
feat_file="/dors/capra_lab/users/abraha1/projects/PTB_phenotyping/data/ptb_predict_machine_learning/feature_matrices/no_twins/time_since_preg_start_icd_cpt_compare_risk_fx_no_twins/icd_cpt/up_to_28_weeks_since_preg_start_icd9_cpt_no_twins_count_compare_riskfx_feat_mat.tsv"
# OLD FILE feat_file="/dors/capra_lab/users/abraha1/projects/PTB_phenotyping/data/ptb_predict_machine_learning/feature_matrices/no_twins/add_ehr_since_preg_start_28wks/up_to_28_weeks_since_preg_start_icd9_cpt_no_twins_count_feat_mat_for_add_ehr.tsv"


###
###    functions
###

def merge_two_df(df1, df2):
    df1_c = df1.copy()
    df2_c = df2.copy()

    intersect = check_for_col_collisions(df1_c, df2_c)
    if  len(intersect) > 0:
        df1_c.drop(intersect, axis=1, inplace=True)
        df2_c.drop(intersect, axis=1, inplace=True)

        print("removed conflicting columns: {}".format(intersect))


    merged_df = pd.merge(df1_c, df2_c, on="GRID", how="inner")

    return merged_df


def check_for_col_collisions(df1, df2):
    old_cols = set(df1.columns)
    new_cols = set(df2.columns)
    old_cols.remove("GRID")
    new_cols.remove("GRID")
    intersect = old_cols.intersection(new_cols)

    return intersect


def create_pairwise_datasets(data1_df, data2_df, data1_grids, data2_grids):
    ''' - data1_df and data2_df are merged on "GRID"
        - GRIDS shared between data1_df and data2_df are identified
        - three datasets are generated for shared GRIDS:
            - data1_df w/ only shared GRIDS
            - data2_df 2/ only shared GRIDS
            - merged data1_df and data2_Df w/ shared GRIDS
        '''
    keep_grids = data1_grids.intersection(data2_grids)
    data_1_2_df = merge_two_df(data1_df, data2_df)

    vs_data2__data1  = data1_df.loc[data1_df.GRID.isin(keep_grids),:].copy()
    vs_data2__data2 = data2_df.loc[data2_df.GRID.isin(keep_grids),:].copy()
    vs_data2__all  = data_1_2_df.loc[data_1_2_df.GRID.isin(keep_grids),:].copy()

    return vs_data2__data1, vs_data2__data2, vs_data2__all

def wrapper_create_pairwise(feat_df, ehr_df):


    vs_age_race__icd_cpt, vs_age_race__age_race, vs_age_race__all = create_pairwise_datasets(feat_df,
                                                                                             ehr_df,
                                                                                             set(feat_df.GRID.unique()),
                                                                                             set(ehr_df.GRID.unique()))

    return vs_age_race__icd_cpt, vs_age_race__age_race, vs_age_race__all

# %%
###
###    main
###


# load delivery labels
final_labels_df = load_labels(DELIVERY_LABELS_FILE)

#
# LOAD FILES
#




# icd_cpt vs. age_race
feat_df = pd.read_csv(feat_file, sep='\t')


# icd_cpt  + prs
df_prs = pd.read_csv(short_hand_files['GRID_PRS.tsv'], sep='\t')
df_struct = pd.read_csv(short_hand_files['struct'], sep='\t')
df_unstruct = pd.read_csv(short_hand_files['unstruct_'], sep='\t')
df_unstruct.rename({'grid':'GRID'}, inplace=True, axis=1)
clin_lab_df = pd.read_csv(short_hand_files['clin_labs'], sep='\t')

df_yob = pd.read_csv(short_hand_files['years_at_delivery_matrix.tsv'], sep='\t')
df_demo = pd.read_csv(short_hand_files['demographics_matrix.tsv'], sep='\t')
age_race_df = merge_two_df(df_yob, df_demo)

vs_age_race__icd_cpt, vs_age_race__age_race, vs_age_race__all = wrapper_create_pairwise(feat_df, age_race_df)
vs_obnotes__icd_cpt, vs_obnotes__obnotes, vs_obnotes__all = wrapper_create_pairwise(feat_df, df_struct) # very low counts...
vs_unstruc__icd_cpt, vs_unstruc__unstruc, vs_unstruc__all = wrapper_create_pairwise(feat_df, df_unstruct)
vs_clin_lab__icd_cpt, vs_clin_lab__clin_lab, vs_clin_lab__all = wrapper_create_pairwise(feat_df, clin_lab_df)
vs_prs__icd_cpt, vs_prs__prs, vs_prs__all = wrapper_create_pairwise(feat_df, df_prs)

# 
all_dfs_dict = {'vs_age_race__icd_cpt': vs_age_race__icd_cpt,
                'vs_age_race__age_race': vs_age_race__age_race,
                'vs_age_race__all': vs_age_race__all,
                'vs_prs__icd_cpt': vs_prs__icd_cpt,
                'vs_prs__prs': vs_prs__prs,
                'vs_prs__all': vs_prs__all,
                'vs_obnotes__icd_cpt': vs_obnotes__icd_cpt,
                'vs_obnotes__obnotes': vs_obnotes__obnotes,
                'vs_obnotes__all': vs_obnotes__all,
                'vs_unstruc__icd_cpt': vs_unstruc__icd_cpt,
                'vs_unstruc__unstruc': vs_unstruc__unstruc,
                'vs_unstruc__all': vs_unstruc__all,
                'vs_clin_lab__icd_cpt': vs_clin_lab__icd_cpt,
                'vs_clin_lab__unstruc': vs_clin_lab__clin_lab,
                'vs_clin_lab__all': vs_clin_lab__all}


# all_dfs_dict = {'vs_clin__lab_icd_cpt': vs_clin_lab__icd_cpt,
#                 'vs_clin__lab_clin_lab': vs_clin_lab__clin_lab,
#                 'vs_clin__lab_all': vs_clin_lab__all}


for label, feat_df in all_dfs_dict.items():

    print("{}: {}".format(label, feat_df.shape[0]))

    if label.startswith('vs_obnotes__'):
        print(f"Skipping {label}")
        continue
    # create a different directory for each type of dataset...
    output_feat_dir = os.path.join(OUTPUT_DIR, label.split('__')[0])
    if not os.path.isdir(output_feat_dir):
        os.mkdir(output_feat_dir)

    feat_file = os.path.join(output_feat_dir, f'{label}_since_concep_28wk_no_twins_feat_mat.tsv')
    feat_df.to_csv(feat_file, sep="\t", index=False)
    print(feat_file)
    # convert to dmatrix

    xgb_train, xgb_eval, xgb_train_eval,  xgb_test, annotated_df, mapped_col_df = convert_to_xgbDataset(feat_file, final_labels_df)

    xgb_train_file = os.path.join(output_feat_dir, f'{label}_since_concep_28wk_no_twins_dtrain.dmatrix')
    xgb_eval_file = os.path.join(output_feat_dir, f'{label}_since_concep_28wk_no_twins_deval.dmatrix')
    xgb_train_eval_file = os.path.join(output_feat_dir, f'{label}_since_concep_28wk_no_twins_dtrain_deval.dmatrix')
    xgb_test_file = os.path.join(output_feat_dir, f'{label}_since_concep_28wk_no_twins_dtest.dmatrix')
    this_annotated_file = os.path.join(output_feat_dir, f'{label}_since_concep_28wk_no_twins_annotated.tsv.pickle')
    this_new_col_name_mapping_file = os.path.join(output_feat_dir, f'{label}_since_concep_28wk_no_twins_new_col_name_mapping.tsv')


    #write
    annotated_df.reset_index(drop=True).to_pickle(this_annotated_file)
    mapped_col_df.to_csv(this_new_col_name_mapping_file)
    print("num bytes in df = {}".format(annotated_df.memory_usage().sum()))
    xgb_train.save_binary(xgb_train_file)
    xgb_eval.save_binary(xgb_eval_file)
    xgb_train_eval.save_binary(xgb_train_eval_file)
    xgb_test.save_binary(xgb_test_file)
    print("DONE\n\n")

