#!/bin/python
# This script will compare how delivery labels (preterm vs. not-preterm) agree between chart reviewed sets and algorithm based call.
#
#
#
# Abin Abraham
# created on: 2019-09-03 15:15:41


# NOTE:
#           > The chart reviewed deliveries were temporally matched to the deliveries ascertained using ICD-EGA-CPT algorithm. (Requried the delivery dates to be within 15 days of each other )
#           > The classificaiton metrics are calculated on data that was present in both Chart Reviewed data and ICD-EGA-CPT algorithm; all other data points were dropped.


import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime


from sklearn.metrics import confusion_matrix, classification_report

DATE = datetime.now().strftime('%Y-%m-%d')


### PATHS

review_file_1="/dors/capra_lab/users/abraha1/projects/PTB_phenotyping/data/ptb_predict_machine_learning/chart_review/random_cohort_1_for_export.csv"
review_file_2="/dors/capra_lab/users/abraha1/projects/PTB_phenotyping/data/ptb_predict_machine_learning/chart_review/random_cohort_2_for_export.csv"
review_file_3="/dors/capra_lab/users/abraha1/projects/PTB_phenotyping/data/ptb_predict_machine_learning/chart_review/random_cohort_3.csv"

root_billing_file = "/dors/capra_lab/users/abraha1/projects/PTB_phenotyping/data/combined_cohorts_07_12_2018/full_dataset_characterization/"
ega_data_file = os.path.join(root_billing_file, "EGA_w-in_3days_of_delivery.tsv")
delivery_file = os.path.join(root_billing_file, "est_delivery_date_at_least_one_icd_cpt_ega.tsv")

OUTPUT_DIR="/dors/capra_lab/users/abraha1/projects/PTB_phenotyping/scripts/rand_forest_ptb_classification/manuscript/chart_review"


# -----------
# FUNCTIONS

# -----------

def delivery_by_ega(x):

    if x >= 42:
        return 'postterm'

    elif (x >=37) & (x < 42):
        return 'term'

    elif (x >=20) & (x < 37):
        return 'preterm'

    else:
        return np.nan

def only_icd_based_label(delivery_df):

    # delivery classication will be the latest gestational classification when mutliple classification are present
    only_icd_df = delivery_df.copy()
    only_icd_df['icd_based_delivery_type'] = np.nan
    only_icd_df.loc[only_icd_df['icd_label']=='preterm, term', 'icd_based_delivery_type'] = 'term'
    only_icd_df.loc[only_icd_df['icd_label']=='preterm', 'icd_based_delivery_type'] = 'preterm'
    only_icd_df.loc[only_icd_df['icd_label']=='term', 'icd_based_delivery_type'] = 'term'
    only_icd_df.loc[only_icd_df['icd_label']=='postterm, term', 'icd_based_delivery_type'] = 'postterm'
    only_icd_df.loc[only_icd_df['icd_label']=='postterm', 'icd_based_delivery_type'] = 'postterm'
    only_icd_df.loc[only_icd_df['icd_label']=='postterm, preterm, term', 'icd_based_delivery_type'] = 'postterm'
    only_icd_df.loc[only_icd_df['icd_label']=='postterm, preterm', 'icd_based_delivery_type'] = 'postterm'

    only_icd_df.consensus_delivery = pd.to_datetime(only_icd_df.consensus_delivery)


    # keep only GRID w/ a ICD-label
    clean_only_icd_df = only_icd_df[~only_icd_df.icd_based_delivery_type.isna()].copy()
    # # keep only the first deliery on file
    # clean_only_icd_df.sort_values( ['GRID','consensus_delivery'], inplace=True)
    # no_dups_icd_df = clean_only_icd_df[~clean_only_icd_df.duplicated('GRID', keep='first')].copy()



    return clean_only_icd_df

def label_ega_df(raw_ega_df):
    ega_df = raw_ega_df.copy()
    ega_df['ega_based_delivery_type'] = 'term'
    ega_df.loc[ega_df['closest_ega'] <=37, 'ega_based_delivery_type'] = 'preterm'
    ega_df.loc[ega_df['closest_ega'] >=42, 'ega_based_delivery_type'] = 'postterm'
    ega_df['delivery_id'] = ega_df.delivery_date +"_" +  ega_df.GRID

    return ega_df

def load_chart_reviews(review_file_1, review_file_2, review_file_3=None):
    chart1_df = pd.read_csv(review_file_1)
    chart2_df = pd.read_csv(review_file_2)

    if review_file_3:
        chart3_df = pd.read_csv(review_file_3)
        all_chart_df = pd.concat([chart1_df, chart2_df,chart3_df])
    else:
        all_chart_df = pd.concat([chart1_df, chart2_df])



    all_chart_df['chart_clasif'] = all_chart_df.EGA.apply(lambda x: delivery_by_ega(float(x)))

    # remove any NaNs
    chart_df = all_chart_df[~all_chart_df.isnull().any(1)].copy()
    chart_df.iloc[:, 2] = pd.to_datetime(chart_df.iloc[:, 2])



    return chart_df

def merge_icd_within15d(chart_df, clean_only_icd_df):

    # for each chart reviewed delivery, see if there is an ICD_label within 15 days, if so, add it to that row
    merged_chart_icd_df = pd.DataFrame()
    for index, row in chart_df.iterrows():

        grid = row['GRID']
        chart_tstamp = row['est. date of delivery']
        chart_clasif = row['chart_clasif']


        # check if grid exists in icd_df
        if clean_only_icd_df[clean_only_icd_df['GRID']==grid].shape[0] > 0:
            for iindex, icd_row in clean_only_icd_df[clean_only_icd_df['GRID']==grid].iterrows():


                icd_tstamps = clean_only_icd_df.loc[clean_only_icd_df['GRID']==grid, 'consensus_delivery'].values[0]
                icd_clasif = clean_only_icd_df.loc[clean_only_icd_df['GRID']==grid, 'icd_based_delivery_type'].values[0]


                # within 15 days of each other
                days_diff = np.abs(np.datetime64(chart_tstamp) - icd_tstamps).astype('timedelta64[D]')
                if (days_diff < np.timedelta64('15','D')):

                    temp_df = pd.DataFrame({'GRID':[grid], 'chart_clasif':[row['chart_clasif']], 'icd_clasif':[icd_clasif], 'chart_delivery_date':[chart_tstamp], 'icd_delivery_date':[icd_tstamps], 'within_15d':[True]})
                else:
                    temp_df = pd.DataFrame({'GRID':[grid], 'chart_clasif':[row['chart_clasif']], 'icd_clasif':[np.nan], 'chart_delivery_date':[np.nan], 'icd_delivery_date':[np.nan], 'within_15d':[False]})

            merged_chart_icd_df = merged_chart_icd_df.append(temp_df)

    return merged_chart_icd_df

def merge_icd_ega_within15d(chart_df, full_delivery_df):

    # for each chart reviewed delivery, see if there is an ICD_label within 15 days, if so, add it to that row
    merged_chart_icd_df = pd.DataFrame()
    for index, row in chart_df.iterrows():

        grid = row['GRID']
        chart_tstamp = row['est. date of delivery']
        chart_clasif = row['chart_clasif']


        # check if grid exists in icd_df
        if full_delivery_df[full_delivery_df['GRID']==grid].shape[0] > 0:
            for iindex, icd_row in full_delivery_df[full_delivery_df['GRID']==grid].iterrows():


                _tstamps = full_delivery_df.loc[full_delivery_df['GRID']==grid, 'consensus_delivery'].values[0]
                _clasif = full_delivery_df.loc[full_delivery_df['GRID']==grid, 'consensus_label'].values[0]


                # within 15 days of each other
                days_diff = np.abs(np.datetime64(chart_tstamp) - _tstamps).astype('timedelta64[D]')
                if (days_diff < np.timedelta64('15','D')):

                    temp_df = pd.DataFrame({'GRID':[grid], 'chart_clasif':[row['chart_clasif']], 'icd_ega_clasif':[_clasif], 'chart_delivery_date':[chart_tstamp], 'icd_ega_delivery_date':[_tstamps], 'within_15d':[True]})
                else:
                    temp_df = pd.DataFrame({'GRID':[grid], 'chart_clasif':[row['chart_clasif']], 'icd_ega_clasif':[np.nan], 'chart_delivery_date':[np.nan], 'icd_ega_delivery_date':[np.nan], 'within_15d':[False]})

            merged_chart_icd_df = merged_chart_icd_df.append(temp_df)

    return merged_chart_icd_df

# %%
# -----------
# MAIN
# -----------

###
#   LOAD DATA
###

ega_df = pd.read_csv(ega_data_file, sep="\t")
full_delivery_df = pd.read_csv(delivery_file, sep="\t")
full_delivery_df.consensus_delivery = pd.to_datetime(full_delivery_df.consensus_delivery)

# icd only based label
clean_only_icd_df = only_icd_based_label(full_delivery_df)
clean_only_icd_df = clean_only_icd_df.loc[:, ['GRID','consensus_delivery','icd_based_delivery_type']].copy()

# evalute delivery type based on ega
#   this ega was confirmed to be measured closest to delivery date (w/in 3 days)
ega_df = label_ega_df(ega_df)


# load chart reviewed data
chart_df = load_chart_reviews(review_file_1, review_file_2,review_file_3)
chart_df[~chart_df.isna().any(1)].GRID.nunique()




# %%
###
#   COMPARE CHART REVIEWED LABELS TO ALGORITHM CALL
###


chart_w_icd_ega_df = merge_icd_ega_within15d(chart_df, full_delivery_df)
time_locked_df = chart_w_icd_ega_df.loc[chart_w_icd_ega_df['within_15d']].copy()
print("{} :number of uniq grids".format(chart_w_icd_ega_df.loc[chart_w_icd_ega_df['within_15d']].GRID.nunique()))

chart_w_icd_ega_df.head()

# -----------  convert to binary prediction  -----------
time_locked_df['chart_clasif_binary'] = time_locked_df.chart_clasif.apply(lambda x: 'preterm' if x == 'preterm' else 'not-preterm')
time_locked_df['icd_ega_clasif_binary'] = time_locked_df.icd_ega_clasif.apply(lambda x: 'preterm' if x == 'preterm' else 'not-preterm')

conf_matrix_df = pd.DataFrame(confusion_matrix(time_locked_df['chart_clasif_binary'], time_locked_df['icd_ega_clasif_binary'], labels=['preterm','not-preterm']), columns=['preterm','not-preterm'], index=['preterm','not-preterm'])
clasif_report_df = pd.DataFrame(classification_report(time_locked_df['chart_clasif_binary'], time_locked_df['icd_ega_clasif_binary'], output_dict=True))
clasif_report_df.drop(columns=['micro avg','macro avg','weighted avg'], inplace=True)



clasif_report_df.to_csv(os.path.join(OUTPUT_DIR, 'classif_report_chart_review1_2_3_vs_algo_labels.tsv'),sep="\t", index=True, header=True)
conf_matrix_df.to_csv(os.path.join(OUTPUT_DIR, 'conf_matrix_chart_review1_2_3_vs_algo_labels.tsv'),sep="\t", index=True, header=True)

