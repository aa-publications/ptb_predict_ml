#!/bin/python
# This script will compare how ICD-9 based delivery label concord with EGA based labels
#
#
#
# Abin Abraham
# created on: 2019-08-14 22:35:52



# NOTE:
#           > We use EGA values recorded w/in 3 days of delivery.
#           >


import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime


from sklearn.metrics import confusion_matrix, classification_report

DATE = datetime.now().strftime('%Y-%m-%d')


### PATHS
root_billing_file = "/dors/capra_lab/users/abraha1/projects/PTB_phenotyping/data/combined_cohorts_07_12_2018/full_dataset_characterization/"
ega_data_file = os.path.join(root_billing_file, "EGA_w-in_3days_of_delivery.tsv")
delivery_file = os.path.join(root_billing_file, "est_delivery_date_at_least_one_icd_cpt_ega.tsv")

OUTPUT_DIR="/dors/capra_lab/users/abraha1/projects/PTB_phenotyping/scripts/rand_forest_ptb_classification/manuscript/ega_icd_concordance"

# %%
# -----------
# FUNCIONS
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
    # delivery label is based only on the ICD data

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


    return clean_only_icd_df

def label_ega_df(raw_ega_df):
    ega_df = raw_ega_df.copy()
    ega_df['ega_based_delivery_type'] = 'term'
    ega_df.loc[ega_df['closest_ega'] <=37, 'ega_based_delivery_type'] = 'preterm'
    ega_df.loc[ega_df['closest_ega'] >=42, 'ega_based_delivery_type'] = 'postterm'
    # ega_df['delivery_id'] = ega_df.delivery_date +"_" +  ega_df.GRID


    return ega_df

def merge_icd_ega_within15d(ega_df, clean_icd_df):

    # for each chart reviewed delivery, see if there is an ICD_label within 15 days, if so, add it to that row
    merged_chart_icd_df = pd.DataFrame()
    for index, row in ega_df.iterrows():

        grid = row['GRID']
        ega_ddate = row['delivery_date']
        ega = row['closest_ega']


        # check if grid exists in icd_df
        if clean_icd_df[clean_icd_df['GRID']==grid].shape[0] > 0:
            for iindex, icd_row in clean_icd_df[clean_icd_df['GRID']==grid].iterrows():


                _tstamps = clean_icd_df.loc[clean_icd_df['GRID']==grid, 'consensus_delivery'].values[0]
                _clasif = clean_icd_df.loc[clean_icd_df['GRID']==grid, 'icd_label'].values[0]


                # within 15 days of each other
                days_diff = np.abs(np.datetime64(ega_ddate) - _tstamps).astype('timedelta64[D]')
                if (days_diff < np.timedelta64('15','D')):

                    temp_df = pd.DataFrame({'GRID':[grid], 'ega':[ega], 'icd_clasif':[_clasif], 'ega_delivery_date':[ega_ddate], 'icd_delivery_date':[_tstamps], 'within_15d':[True]})
                else:
                    temp_df = pd.DataFrame({'GRID':[grid], 'ega':[ega], 'icd_clasif':[np.nan], 'ega_delivery_date':[np.nan], 'icd_delivery_date':[np.nan], 'within_15d':[False]})

            merged_chart_icd_df = merged_chart_icd_df.append(temp_df)

    return merged_chart_icd_df




# %%
# -----------
# MAIN
# -----------

# load
ega_df = pd.read_csv(ega_data_file, sep="\t")
delivery_df = pd.read_csv(delivery_file, sep="\t")

assert ega_df.GRID.duplicated().sum() == 0, 'dups in ega_df'
assert delivery_df.GRID.duplicated().sum() == 0, 'dups in delivery_df'

print("Num deliveries with EGA and ICD based classificaiton: {:,}".format(delivery_df.loc[ (delivery_df['ega_label'] != 'None') & (delivery_df['icd_label'] != 'None'), 'GRID'].shape[0]))


# select GRIDS w/ ICD based delivery classifications
#       > here choose deliveryies with ICD data
#       >
only_icd_df = delivery_df.loc[(delivery_df['icd_exists'] == True)].copy()
clean_icd_df = only_icd_based_label(only_icd_df)

# evalute delivery type based on ega
#   this ega was confirmed to be measured closest to delivery date (w/in 3 days)
ega_df = label_ega_df(ega_df)


# merge ega data with icd data if delivery date is w/in 15 d
chart_w_icd_ega_df = merge_icd_ega_within15d(ega_df, clean_icd_df)

# I forgot to use the icd based classification (that has only one label), so now I am re-parsing the icd_clasif column to output only one delivery label
chart_w_icd_ega_df.rename(columns={'icd_clasif':'icd_label', 'icd_delivery_date':'consensus_delivery', 'ega':'closest_ega'}, inplace=True)
t_clean_icd_ega_df = only_icd_based_label(chart_w_icd_ega_df)

# same with the ega
clean_icd_ega_df = label_ega_df(t_clean_icd_ega_df)

# write
clean_icd_ega_df.to_csv(os.path.join(OUTPUT_DIR, 'icd_ega_cocord__delivery_w_in_15d.tsv'), sep="\t", index=False)

assert clean_icd_ega_df[clean_icd_ega_df.loc[:, ['icd_based_delivery_type','ega_based_delivery_type']].isna().any(1)].shape[0] == 0, 'na values present'

clean_icd_ega_df.head(2)
# convert to binary
clean_icd_ega_df['icd_based_delivery_type_binary'] = clean_icd_ega_df.icd_based_delivery_type.apply(lambda x: 'preterm' if x == 'preterm' else 'not-preterm')
clean_icd_ega_df['ega_based_delivery_type_binary'] = clean_icd_ega_df.ega_based_delivery_type.apply(lambda x: 'preterm' if x == 'preterm' else 'not-preterm')

conf_matrix_df = pd.DataFrame(confusion_matrix(clean_icd_ega_df['ega_based_delivery_type_binary'], clean_icd_ega_df['icd_based_delivery_type_binary'], labels=['preterm','not-preterm']), columns=['preterm','not-preterm'], index=['preterm','not-preterm'])
clasif_report_df = pd.DataFrame(classification_report(clean_icd_ega_df['ega_based_delivery_type_binary'], clean_icd_ega_df['icd_based_delivery_type_binary'], output_dict=True))
clasif_report_df.drop(columns=['micro avg','macro avg','weighted avg'], inplace=True)

conf_matrix_df.to_csv(os.path.join(OUTPUT_DIR, 'conf_matrix_icd_ega_concord.tsv'),sep="\t", header=True, index=True)
clasif_report_df.to_csv(os.path.join(OUTPUT_DIR, 'classif_ega_vs_icd_labels.tsv'),sep="\t", header=True, index=True) 
