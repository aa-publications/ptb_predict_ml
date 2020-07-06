#!/bin/python
# This script will calculate summary statistics for a given cohort.
#
#
#
# Abin Abraham
# created on: 2019-05-22 14:41:58



# %%
import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime

DATE = datetime.now().strftime('%Y-%m-%d')


# %%
# -----------
# FUNCTION
# -----------

ROOT_DIR="/dors/capra_lab/users/abraha1/projects/PTB_phenotyping/results/ptb_predict_machine_learning"
COHORT_DIR= os.path.join(ROOT_DIR,"2019-01-16_xgboost_hyperopt_icd_cpt_raw_counts")
INPUT_FILE = os.path.join(COHORT_DIR, 'input_data/input_data_all_icd9_cpt_count_subset-2019-01-26.tsv')

RACE_FILE = "/dors/capra_lab/users/abraha1/projects/PTB_phenotyping/data/ptb_predict_machine_learning/demographics/demographics_matrix.tsv"
AGE_FILE="/dors/capra_lab/users/abraha1/projects/PTB_phenotyping/data/ptb_predict_machine_learning/demographics/years_at_delivery_matrix.tsv"

OB_STRU_FILE="/dors/capra_lab/users/abraha1/projects/PTB_phenotyping/data/ptb_predict_machine_learning/ob_notes_variables/filtered_wide_ob_notes_w_na_binary_and_counts.tsv"
OB_UNSTRUC_FILE="/dors/capra_lab/users/abraha1/projects/PTB_phenotyping/data/ptb_predict_machine_learning/feature_matrices/ob_notes_unstruct/'filtered_wide_ob_notes_w_na_binary_and_counts.tsv"

# %%
# -----------
# FUNCTIONS
# -----------

def format_demo_df(og_demo_df):

    demo_df = og_demo_df.copy()
    demo_df['more_than_one_race'] = demo_df.sum(1) > 1
    demo_df['consensus_race'] = np.nan

    demo_df.loc[demo_df['more_than_one_race'], 'consensus_race'] = 'MORE_THAN_ONE'
    cols = demo_df.columns.to_list()
    cols.remove('GRID')
    cols.remove('consensus_race')
    cols.remove('more_than_one_race')

    for column in cols:
        print("on col:{}".format(column))
        demo_df.loc[(demo_df[column] ==1) & (demo_df['more_than_one_race'] == False), 'consensus_race'] = column


    demo_df['consensus_race_clean'] = demo_df.consensus_race.apply(lambda x: x.replace('RACE_LIST_','') if (type(x) != float) else 'FLOAT')

    final_df = demo_df.loc[:, ['GRID','consensus_race_clean']].copy()

    return final_df

def get_counts_by_race(merged_df):
    """

    Parameters
    ----------
    merged_df : pandas.DataFrame
        should contain one GRID per row. Following columns are required: ['GRID', 'consensus_race_clean', 'years_at_delivery', 'binary_labels']

    Returns
    -------
    count_by_race: pandas.DataFrame
        summary of counts by RACE

    note: uses groupby().count(), therefore does NOT include NaN values

    """

    count_by_race = merged_df.loc[: , ['binary_labels','consensus_race_clean', 'GRID']].groupby(['binary_labels', 'consensus_race_clean'], as_index=False).count().pivot(index='consensus_race_clean', columns = 'binary_labels', values='GRID').reset_index()
    count_by_race.columns.name = ''

    tots = pd.DataFrame({'consensus_race_clean':['total'],
                         'not-preterm':count_by_race['not-preterm'].sum(),
                          'preterm':count_by_race['preterm'].sum()})

    count_by_race = count_by_race.append(tots)

    # calc percent total


    nptb_total = count_by_race.loc[count_by_race['consensus_race_clean'] == 'total','not-preterm' ].values
    ptb_total = count_by_race.loc[count_by_race['consensus_race_clean'] == 'total','preterm' ].values

    count_by_race['per_not-preterm'] = np.around(count_by_race['not-preterm']/nptb_total,5)
    count_by_race['per_preterm'] = np.around(count_by_race['preterm']/ptb_total,5)




    return count_by_race.sort_values('preterm',ascending=False)


def summarize_years_at_delivery(og_merged_df):
    """

    Parameters
    ----------
    merged_df : pd.DataFrame
        Should contain one GRID per row.
        Following columns are required: ['GRID', 'binary_labels', 'years_at_delivery']

    Returns
    -------
    final_df: pd.DataFrame
        summary stats on year at delivery

    note: remove NaN before any calculations...

    """
    merged_df = og_merged_df.copy()

    merged_df.loc[merged_df['years_at_delivery']=="NA"]
    merged_df.replace("NA", np.nan, inplace=True)

    summary_by_label = merged_df.loc[:, ['GRID','binary_labels', 'years_at_delivery']].groupby(['binary_labels']).agg({
        'years_at_delivery':[_min_, _5th_, _mean_, _median_, _95th_, _max_, _num_na_, _num_total_]}).transpose()

    final_df = summary_by_label.reset_index().drop('level_0', axis=1).rename(columns={'level_1':'stats'})
    final_df.columns.names=['index']


    return final_df

def _min_(x):
    no_na_x = x[~x.isna()]
    return np.median(no_na_x)

def _5th_(x):
    no_na_x = x[~x.isna()]
    return np.quantile(no_na_x, 0.05)

def _mean_(x):
    no_na_x = x[~x.isna()]
    return np.mean(no_na_x)

def _median_(x):
    no_na_x = x[~x.isna()]
    return np.median(no_na_x)


def _95th_(x):
    no_na_x = x[~x.isna()]
    return np.quantile(no_na_x, 0.95)

def _max_(x):
    no_na_x = x[~x.isna()]
    return np.median(no_na_x)

def _num_na_(x):
    return  np.sum(x.isna())

def _num_total_(x):
    return  len(x)

def format_merged_df(og_merge_df):
    # convert NaN to 'NA', and have binary labels
    merged_df = og_merge_df.copy()

    # convert nan to "NA"
    merged_df.fillna('NA', inplace=True)
    # convert to preterm  vs. not_preterm
    merged_df['binary_labels'] = merged_df.label.apply(lambda x: 'preterm' if x =='preterm' else 'not-preterm')
    merged_df.drop(['label'], axis=1, inplace=True)

    return merged_df


# %%
# -----------
# MAIN
# -----------

# load data
#       - note:


input_df = pd.read_csv(INPUT_FILE, sep="\t", usecols=['GRID','label'])
demo_df = pd.read_csv(RACE_FILE, sep="\t")
clean_demo_df = format_demo_df(demo_df)
assert not 'FLOAT' in clean_demo_df.consensus_race_clean.unique(), 'check race formatting...'
age_df = pd.read_csv(AGE_FILE, sep="\t")

# load billing code data


# %%
# merge
dmerge_df = pd.merge(input_df, clean_demo_df, on='GRID',how="left")
raw_merged_df = pd.merge(dmerge_df, age_df, on='GRID',how="left")
merged_df = format_merged_df(raw_merged_df)

# %%

counts_by_race = get_counts_by_race(merged_df)
year_at_delivery_summary = summarize_years_at_delivery(merged_df)



# billing codes
    # total number of codes between ptb vs non_ptb
    #   compare distributions

    # total number of unique codes
    #   compare distributions

    # code density over time relative to delivery

    #
    # mean of total number of codes per person in EHR for ptb vs non_ptb
        # same as above but for before the first delivery

    # median+/-95ci code density in time bins relative to first delivery for ptb vs non-ptb
