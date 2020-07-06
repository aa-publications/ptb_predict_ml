


#!/bin/python
# This script will contains funtions for merging different types of data (icd, cpt, labs, prs etc..) to generate feature matrices for xgboost.
#
#
#
# Abin Abraham
# created on: 2019-09-05 08:49:31


import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime

DATE = datetime.now().strftime('%Y-%m-%d')




# -----------
# FUNCTIONS
# -----------

def check_for_col_collisions(df1, df2):
    old_cols = set(df1.columns)
    new_cols = set(df2.columns)





    
    old_cols.remove("GRID")
    new_cols.remove("GRID")
    intersect = old_cols.intersection(new_cols)

    return intersect

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
