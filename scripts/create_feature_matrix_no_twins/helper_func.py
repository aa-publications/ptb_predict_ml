#!/bin/python
# This script will ...
#
#
#
# Abin Abraham
# created on: 2020-06-03 09:17:06


import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime

DATE = datetime.now().strftime('%Y-%m-%d')



def ascertainment_and_twin_codes_to_exclude():

    PRETERM_ICD9_CODES = ['644.2', '644.20', '644.21']
    PRETERM_ICD10_CODES = ['O60.1', 'O60.10', 'O60.10X0', 'O60.10X1', 'O60.10X2', 'O60.10X3', 'O60.10X4',
                           'O60.10X5', 'O60.10X9', 'O60.12', 'O60.12X0', 'O60.12X1', 'O60.12X2', 'O60.12X3',
                           'O60.12X4', 'O60.12X5', 'O60.12X9', 'O60.13', 'O60.13X0', 'O60.13X1', 'O60.13X2',
                           'O60.13X3', 'O60.13X4', 'O60.13X5', 'O60.13X9', 'O60.14', 'O60.14X0', 'O60.14X1',
                           'O60.14X2', 'O60.14X3', 'O60.14X4', 'O60.14X5', 'O60.14X9']

    TERM_ICD9_CODES = ['650', '645.1', '645.10', '645.11', '645.13', '649.8', '649.81', '649.82']
    TERM_ICD10_CODES = ['O60.20', 'O60.20X0', 'O60.20X1', 'O60.20X2', 'O60.20X3', 'O60.20X4', 'O60.20X5',
                        'O60.20X9', 'O60.22', 'O60.22X0', 'O60.22X1', 'O60.22X2', 'O60.22X3', 'O60.22X4',
                        'O60.22X5', 'O60.22X9', 'O60.23', 'O60.23X0', 'O60.23X1', 'O60.23X2', 'O60.23X3',
                        'O60.23X4', 'O60.23X5', 'O60.23X9', 'O80', 'O48.0', '650', '645.1', '645.10',
                        '645.11', '645.13', '649.8', '649.81', '649.82']

    POSTTERM_ICD9_CODES = ['645.2', '645.20', '645.21', '645.23', '645.00', '645.01', '645.03']
    POSTTERM_ICD10_CODES = ['O48.1']

    CPT_DELIVERY_CODES = ['59400', '59409', '59410', '59414', '59510', '59514',
                     '59515', '59525', '59610', '59612', '59614', '59618', '59620', '59622']

    # ZA3 codes sepereated by gestational age (icd10 codes)
    LESS_20WK_CODES = ['Z3A.0', 'Z3A.00', 'Z3A.01', 'Z3A.08', 'Z3A.09', 'Z3A.1', 'Z3A.10', 'Z3A.11',
                       'Z3A.12', 'Z3A.13', 'Z3A.14', 'Z3A.15', 'Z3A.16', 'Z3A.17', 'Z3A.18', 'Z3A.19']
    BW_20_37WK_CODES = ['Z3A.2', 'Z3A.20', 'Z3A.21', 'Z3A.22', 'Z3A.23', 'Z3A.24', 'Z3A.25', 'Z3A.26',
                        'Z3A.27', 'Z3A.28', 'Z3A.29', 'Z3A.3', 'Z3A.30', 'Z3A.31', 'Z3A.32', 'Z3A.33',
                        'Z3A.34', 'Z3A.35', 'Z3A.36', 'Z3A.37']
    BW_37_42WK_CODES = ['Z3A.38', 'Z3A.39', 'Z3A.4', 'Z3A.40', 'Z3A.41']
    BW_42_HIGHER_CODES = ['Z3A.42', 'Z3A.49']

    # twins, triplets etc. (multiple gestation codes)
    print("included codes for twins and multiple gestation!")
    ICD9_MULT_GEST = ['651','651.7','651.70','651.71','651.8','651.81','651.83','651.9','651.91','651.93','652.6','652.60','652.61','652.63','V91','V91.9','V91.90','V91.91','V91.92','V91.99']
    MORE_ICD9_TWINS_CODES=["651","651.0","651.00","651.01","651.03","651.1","651.10","651.11","651.13","651.2","651.20","651.21","651.23","651.3","651.30","651.31","651.33","651.4","651.40","651.41","651.43","651.5","651.50", "651.51", "651.53"]
    COMPLETE_VCODES_TWINS_ICD9 =  ["V91","V91.0","V91.00","V91.01","V91.02","V91.03","V91.09","V91.1","V91.10","V91.11","V91.12","V91.19","V91.2","V91.20","V91.21","V91.22","V91.29","V91.9","V91.90","V91.91","V91.92", "V91.99"]
    CPT_TWINS_CODES = ["74713","76802","76810","76812","76814"]
    ICD10_MULT_GEST = ['BY4BZZZ','BY4DZZZ','BY4GZZZ','O30.801','O30.802','O30.803','O30.809','O30.811','O30.812','O30.813','O30.819','O30.821','O30.822','O30.823','O30.829','O30.891','O30.892','O30.893','O30.899','O30.91','O30.92','O30.93','O31.BX10','O31.BX11','O31.BX12','O31.BX13','O31.BX14','O31.BX15','O31.BX19','O31.BX20','O31.BX21','O31.BX22','O31.BX23','O31.BX24','O31.BX25','O31.BX29','O31.BX30','O31.BX31','O31.BX32','O31.BX33','O31.BX34','O31.BX35','O31.BX39','O31.BX90','O31.BX91','O31.BX92','O31.BX93','O31.BX94','O31.BX95','O31.BX99']


    exclude_codes = PRETERM_ICD9_CODES + PRETERM_ICD10_CODES + TERM_ICD9_CODES + TERM_ICD10_CODES + POSTTERM_ICD9_CODES + \
        POSTTERM_ICD10_CODES + LESS_20WK_CODES + BW_20_37WK_CODES + BW_37_42WK_CODES + BW_42_HIGHER_CODES + BW_42_HIGHER_CODES + CPT_DELIVERY_CODES + \
        ICD9_MULT_GEST + ICD10_MULT_GEST + MORE_ICD9_TWINS_CODES + COMPLETE_VCODES_TWINS_ICD9

    return exclude_codes


def get_mult_gest_and_twin_codes():
    # twins, triplets etc. (multiple gestation codes)
    ICD9_MULT_GEST = ['651','651.7','651.70','651.71','651.8','651.81','651.83','651.9','651.91','651.93','652.6','652.60','652.61','652.63','V91','V91.9','V91.90','V91.91','V91.92','V91.99']
    MORE_ICD9_TWINS_CODES=["651","651.0","651.00","651.01","651.03","651.1","651.10","651.11","651.13","651.2","651.20","651.21","651.23","651.3","651.30","651.31","651.33","651.4","651.40","651.41","651.43","651.5","651.50", "651.51", "651.53"]
    COMPLETE_VCODES_TWINS_ICD9 =  ["V91","V91.0","V91.00","V91.01","V91.02","V91.03","V91.09","V91.1","V91.10","V91.11","V91.12","V91.19","V91.2","V91.20","V91.21","V91.22","V91.29","V91.9","V91.90","V91.91","V91.92", "V91.99"]
    CPT_TWINS_CODES = ["74713","76802","76810","76812","76814"]
    ICD10_MULT_GEST = ['BY4BZZZ','BY4DZZZ','BY4GZZZ','O30.801','O30.802','O30.803','O30.809','O30.811','O30.812','O30.813','O30.819','O30.821','O30.822','O30.823','O30.829','O30.891','O30.892','O30.893','O30.899','O30.91','O30.92','O30.93','O31.BX10','O31.BX11','O31.BX12','O31.BX13','O31.BX14','O31.BX15','O31.BX19','O31.BX20','O31.BX21','O31.BX22','O31.BX23','O31.BX24','O31.BX25','O31.BX29','O31.BX30','O31.BX31','O31.BX32','O31.BX33','O31.BX34','O31.BX35','O31.BX39','O31.BX90','O31.BX91','O31.BX92','O31.BX93','O31.BX94','O31.BX95','O31.BX99']

    mult_gest_codes = ICD9_MULT_GEST + MORE_ICD9_TWINS_CODES + COMPLETE_VCODES_TWINS_ICD9 + CPT_TWINS_CODES + ICD10_MULT_GEST
    return mult_gest_codes

def keep_only_singletons(mult_gest_codes, df):
    # df should be a long dataframe. one row per GRID-ICD-DATE comobo
    # return a df excluding GRIDs with ≥ 1 code indicating multiple gestations.
    mult_gest_grids = df.loc[df.ICD.isin(mult_gest_codes), 'GRID'].unique()
    singletons_df = df.loc[~df.GRID.isin(mult_gest_grids)].copy()
    print(f"Removed {len(mult_gest_grids):,} out of {df.GRID.nunique():,} women due to multiple pregnancies.")

    return singletons_df


def get_earliest_preg_start(final_labels_df, ega_df):

    # load EGA data  -- keep only first/earliest delivery
    ega_df.delivery_date = pd.to_datetime(ega_df.delivery_date)
    ega_df.sort_values(['GRID','delivery_date'],inplace=True, ascending=True)
    earliest_ega_df = ega_df[~ega_df.duplicated(['GRID'], keep='first')].copy()

    #
    # keep only ega values for delivery date matching in final_labels_df
    #

    temp_first_label_df = final_labels_df.copy()
    temp_first_label_df['GRID_DDATE'] = temp_first_label_df.GRID +"_"+temp_first_label_df.delivery_date

    temp_early_ega_df = earliest_ega_df.copy()
    temp_early_ega_df['GRID_DDATE'] = temp_early_ega_df.GRID +"_"+ temp_early_ega_df.delivery_date.dt.strftime( "%Y-%m-%d")

    # align delivery dates
    keep_early_ega_df = temp_early_ega_df[temp_early_ega_df.GRID_DDATE.isin(temp_first_label_df.GRID_DDATE)].copy()


    delivery_date_dict = dict(zip(keep_early_ega_df.GRID, keep_early_ega_df.delivery_date.astype('str')))
    preg_start_date_dict = dict(zip(keep_early_ega_df.GRID, keep_early_ega_df.date_of_conception.astype('str')))

    return delivery_date_dict, preg_start_date_dict