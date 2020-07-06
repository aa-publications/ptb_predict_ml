#!/bin/python
# This script will ...
#
#
#
# Abin Abraham
# created on: 2020-03-27 08:38:00



import os
import sys
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import glob
import pickle
from datetime import datetime

sys.path.append('/dors/capra_lab/users/abraha1/projects/PTB_phenotyping/scripts/rand_forest_ptb_classification')
from hyperparam_tune import validate_best_model

sys.path.append("/dors/capra_lab/users/abraha1/projects/PTB_phenotyping/scripts/rand_forest_ptb_classification/manuscript/0_helper_func")
from manip_trained_models_funcs import unpack_input_data, get_preds

from sklearn import metrics
from sklearn.metrics import accuracy_score

DATE = datetime.now().strftime('%Y-%m-%d')

%matplotlib inline

###
###    PATHS
###
RF_DIR = "/dors/capra_lab/users/abraha1/projects/PTB_phenotyping/results/ptb_predict_machine_learning/2019_05_20_time_since_conception/up_to_28_weeks_since_preg_start_icd9_cpt_count"
CLIN_RISK_DIR = "/dors/capra_lab/users/abraha1/projects/PTB_phenotyping/results/ptb_predict_machine_learning/2020_03_24_time_since_concep_up_to_28wks_w_riskfx"
OUTPUT_DIR = "/dors/capra_lab/users/abraha1/projects/PTB_phenotyping/scripts/rand_forest_ptb_classification/manuscript/quantify_by_clin_risk_fx"



# -----------
# FUNCSION
# -----------

def calc_npv(y_true, y_pred):
    tn, fp, fn, tp = metrics.confusion_matrix(y_true, y_pred).ravel()
    return tn/(tn+fn)




###
###    MAIN
###

### define file paths
ehr_input_file=os.path.join(RF_DIR,'input_data_up_to_28_weeks_since_preg_start_icd9_cpt_count-2019-06-19.tsv')
riskfx_input_file=os.path.join(CLIN_RISK_DIR,'input_data_up_to_28_weeks_since_preg_start_risk_fx-2020-03-24.tsv')


### load models and input files
ehr_model_file=os.path.join(RF_DIR,'best_xgb_model_up_to_28_weeks_since_preg_start_icd9_cpt_count-2019-06-19.pickle')
riskfx_model_file=os.path.join(CLIN_RISK_DIR,'best_xgb_model_up_to_28_weeks_since_preg_start_risk_fx-2020-03-24.pickle')


ehr_X_train, ehr_y_train, ehr_X_test, ehr_y_test, ehr_xgb_model, ehr_input_data = unpack_input_data(ehr_input_file, ehr_model_file)
riskfx_X_train, riskfx_y_train, riskfx_X_test, riskfx_y_test, riskfx_xgb_model, riskfx_input_data = unpack_input_data(riskfx_input_file, riskfx_model_file)

# check that grids match!
np.all(ehr_input_data.GRID == riskfx_input_data.GRID)

# %%
###
###    compare predictions
###

ehr_y_pred, ehr_y_proba = get_preds(ehr_xgb_model, ehr_X_test)
riskfx_y_pred, riskfx_y_proba = get_preds(riskfx_xgb_model, riskfx_X_test)

# measure ppv and npv
ehr_ppv = metrics.precision_score(ehr_y_test, ehr_y_pred)
riskfx_ppv = metrics.precision_score(ehr_y_test, riskfx_y_pred)
ehr_npr = calc_npv(ehr_y_test, ehr_y_pred)
riskfx_npv = calc_npv(ehr_y_test, riskfx_y_pred)

ehr_ppv
riskfx_ppv
ehr_npr
riskfx_npv


accu_df = pd.DataFrame()
for ehr_state in [0, 1]:
    for riskfx_state in [0,1]:

        this_bool = (ehr_y_pred ==ehr_state) & (riskfx_y_pred==riskfx_state)


        ehr_acu = accuracy_score(ehr_y_test[this_bool], ehr_y_pred[this_bool])
        riskfx_acu = accuracy_score(ehr_y_test[this_bool], riskfx_y_pred[this_bool])


        temp_df = pd.DataFrame({'ehr_bool':[ehr_state], 'riskfx_bool':[riskfx_state], 'ehr_acu':[ehr_acu], 'riskfx_acu':[riskfx_acu], 'count':[np.sum(this_bool)]})
        accu_df = accu_df.append(temp_df)


# -----------
# measure ppv and npv
# -----------



###
###    predicted probabilities
###

# %%
plt.rcParams["font.family"] = "Liberation Sans"
sns.set(style="ticks",  font_scale=1.0, rc={"figure.figsize": (8, 6)})
fig, ax = plt.subplots()

true_ptb_riskfx_bool = riskfx_y_test == 1
true_ptb_ehr_bool = ehr_y_test == 1 # same as above


ax.scatter(x=riskfx_y_proba[true_ptb_riskfx_bool][:,1], y=ehr_y_proba[true_ptb_riskfx_bool][:,1], alpha=0.3, s=3, color='indianred', label='true ptb')
ax.scatter(x=riskfx_y_proba[~true_ptb_riskfx_bool][:,1], y=ehr_y_proba[~true_ptb_riskfx_bool][:,1], alpha=0.3, s=3, color='royalblue', label='true not-ptb')

ax.set_ylim(0,1)
ax.set_xlim(0,1)
ax.axvline(0.5, color='black', linestyle=":", lw=2)
ax.axhline(0.5, color='black', linestyle=":", lw=2)

sns.despine(top=True, right=True)

ax.set_xlabel("Based on clincal risk factors")
ax.set_ylabel("Based on EHR billing codes")
ax.set_title("PTB probability at 28 weeks since conception")

ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

ax.axis('equal')
plt.tight_layout()
# plt.savefig(os.path.join(OUTPUT_DIR, f'{DATE}_ptb_proba_by_billing_codes_vs_riskfx_at_28wks_since_concep.png'), dpi=300)


###
###    CALIBRATION CURVE
###
# %%
from sklearn.calibration import calibration_curve


sns.set(style="whitegrid",  font_scale=1.0, rc={"figure.figsize": (8, 8)})
fig, ax = plt.subplots()


risk_fx_fraction_of_positives, risk_fx_mean_predicted_value = calibration_curve(riskfx_y_test, riskfx_y_proba[:,1], n_bins=20)
ax.plot(mean_predicted_value, fraction_of_positives, label='Risk Fx', color='orange', marker='o')

ehr_fraction_of_positives, ehr_mean_predicted_value = calibration_curve(ehr_y_test, ehr_y_proba[:,1], n_bins=20)
ax.plot(ehr_fraction_of_positives, ehr_mean_predicted_value, label='Billing Codes', color='green', marker='o')

ax.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")

ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))


ax.set_ylabel("Fraction of positives")
ax.set_xlabel("Mean predicted value")
ax.set_ylim([-0.05, 1.05])
ax.set_xlim([0, 1])
