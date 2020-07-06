#!/bin/python
# Run xgboost with hyeropt and mongoDB
#           # USES NATIVE PYTHON IMPLEMENTATION
#
#
#
#
#
#
#
#               SCRIPT SETTING THAT YOU MIGHT WANT TO MODIFY:
#                   - specific hyperparmamters
#                   - fmin settings
#
#
#
#
#
# Abin Abraham
# created on: 2018-12-27 11:37:41
# Py 3.6.2
# Scikit  0.19.0
# Pandas 0.23.4
# Numpy 1.13.1
# xgboost 0.81
# hyperopt 0.1.1



import os
import sys
import pickle
import time
import logging
import numpy as np
import pandas as pd
import xgboost as xgb

from functools import partial
from datetime import datetime
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK, STATUS_FAIL, rand, space_eval
from hyperopt.mongoexp import MongoTrials
from xgboost.sklearn import XGBClassifier

sys.path.append('/dors/capra_lab/users/abraha1/projects/PTB_phenotyping/scripts/rand_forest_ptb_classification')
from hyperparam_tune import initialize, validate_best_model, create_held_out
from get_feature_importances import get_feature_importance, barplot_feat_importance
from train_test_rf import load_labels, load_X_y, compute_metrics, metrics_to_df, plot_roc, plot_pr

sys.path.append('/dors/capra_lab/users/abraha1/projects/PTB_phenotyping/scripts/rand_forest_ptb_classification/shap_feat_importance')
from shaply_funcs import calc_write_shap


# import pdb
# from hyperopt.pyll.stochastic import sample

DATE = datetime.now().strftime('%Y-%m-%d')

# parameters for native implementation of xgboost
BOOSTER_PARAMS = ['max_depth', 'min_child_weight', 'subsample', 'colsample_bytree', 'eta', 'gamma', 'objective', 'alpha', 'lambda']

# -----------
# FUNCTIONS
# -----------


def fpreproc(dtrain, dtest, param):
    ''' update scale_pos_weight for each fold in cv'''

    label = dtrain.get_label()

    updated_ratio = float(np.sum(label == 0)) / np.sum(label == 1)
    param['scale_pos_weight'] = updated_ratio

    return (dtrain, dtest, param)


def objective(space, dtrain_file=None):
    import time
    import numpy as np
    import pandas as pd
    import xgboost as xgb
    from hyperopt import STATUS_OK
    # from hyperopt.pyll.stochastic import sample
    start_o = time.time()


    def format_params(space):

        formated_space = {'booster': {  'max_depth': int(space['max_depth']),  # booster param
                                        'min_child_weight': int(space['min_child_weight']),  # booster param
                                        'subsample': float(space['subsample']),  # booster param
                                        'colsample_bytree': float(space['colsample_bytree']),  # booster param
                                        'eta': float(space['eta']),  # booster param (learning_rate)
                                        'gamma': float(space['gamma']),  # booster pram (alias min_split_loss)
                                        'objective': space['objective'],  # learning task
                                        'alpha': float(space['alpha']),  # booster param (reg_alpha)
                                        "lambda":float(space['lambda']), # booster param
                                    },

                        'other': {  'verbosity': int(space['verbosity']),  # general param
                                    'booster':  space['booster'],  # general param
                                    'seed': int(space['seed']),  # learning task
                                    "num_boost_rounds": int(space['num_boost_rounds']),
                                    "eval_metric": str(space['eval_metric'])  # learning task
                                    }
                        }

        return formated_space

    def fpreproc(dtrain, dtest, param):
        ''' update scale_pos_weight for each fold in cv'''

        label = dtrain.get_label()

        updated_ratio = float(np.sum(label == 0)) / np.sum(label == 1)
        param['scale_pos_weight'] = updated_ratio

        return (dtrain, dtest, param)

    #note:  scale_pos_weight will be updated on the fly with each cv iteration

    dtrain = xgb.DMatrix(dtrain_file)
    params = format_params(space)
    optim_xgb = xgb.cv(params['booster'], dtrain,
                    num_boost_round=params['other']['num_boost_rounds'],
                    nfold=2, stratified=True,
                    fpreproc=fpreproc,
                    metrics=params['other']['eval_metric'],
                    verbose_eval=params['other']['verbosity'],
                    seed=params['other']['seed'])


    end_o = time.time()
    result = {'loss': -1*optim_xgb['test-map-mean'].values[-1],
            'test_mean_map': -1*optim_xgb['test-map-mean'].values[-1],
            'test_std_map': -1*optim_xgb['test-map-std'].values[-1],
            'train_mean_map': -1*optim_xgb['train-map-mean'].values[-1],
            'train_std_map': -1*optim_xgb['train-map-std'].values[-1],
            'status': STATUS_OK,
            'params': params,
            'time': round(end_o, 8)} # i switched this from elapsed time to end time on 2019-03-21 12:31:46

    print("Done w/ objective functions in {:,} mins.".format(round((end_o - start_o)/60, 3)))

    return result


def flatten_trial_results(one_result):
    r_dict = one_result.to_dict()
    r_params = r_dict.pop('params')
    r_booster = r_params.pop('booster')
    r_other = r_params.pop('other')

    r_flat = {**r_dict, **r_booster, **r_other}

    return r_flat


def to_sklearn_params(space):
    ''' pass in parameter dictionary for native python xgboost implementation  and return sklearn compatible parameters'''

    sk_params = {'n_estimators': int(space['num_boost_rounds']),
                     'max_depth': int(space['max_depth']),
                     'min_child_weight': int(space['min_child_weight']),
                     'subsample': float(space['subsample']),
                     'colsample_bytree': float(space['colsample_bytree']),
                     'learning_rate': float(space['eta']),
                     'gamma': float(space['gamma']),
                     'reg_alpha': float(space['alpha']),
                     'reg_lambda': float(space['lambda']),
                     'random_state': int(space['seed']),
                     'objective': space['objective'],
                     'booster':  space['booster'],
                     'scale_pos_weight': space['scale_pos_weight'],
                     'importance_type':'gain',
                     'silent': 1}

    return sk_params

def load_held_out_data(annotated_df_file):
    ''' assummes that annotated_df file is ...
                * tsv
                * first column = GRIDS
                * last column is "partition" w/ held_out or grid_cv
                * second to last column is 'label' with either preterm or term
                * all other columns are features '''

    if annotated_df_file.endswith('.pickle'):
        annotated_df = pd.read_pickle(annotated_df_file)
    else:
        annotated_df = pd.read_feather(annotated_df_file)

    train_df = annotated_df.loc[annotated_df['partition'] == 'grid_cv'].copy()
    test_df = annotated_df.loc[annotated_df['partition']  == 'held_out'].copy()

    X_train = train_df.iloc[:,1:-2]
    y_train = train_df.label.apply(lambda x: 1 if x == 'preterm' else 0).values

    X_test = test_df.iloc[:,1:-2]
    y_test = test_df.label.apply(lambda x: 1 if x == 'preterm' else 0).values

    return X_train, y_train, X_test, y_test, annotated_df

def extract_train_df(input_df):

    # create train and test df
    train_df = input_df.loc[input_df['partition']=='grid_cv'].copy()
    train_df_no_labels = train_df.drop(['GRID','label','partition'], axis=1, inplace=False)

    return train_df_no_labels, train_df

def extract_test_df(input_df):

    # create train and test df
    test_df = input_df.loc[input_df['partition']=='held_out'].copy()
    test_df_no_labels = test_df.drop(['GRID','label','partition'], axis=1, inplace=False)

    return test_df_no_labels, test_df


# -----------
# MAIN
# -----------


if __name__ == "__main__":
    start_m = time.time()

    ###
    #   FILE PATHS
    ###

    # if no args passed, run with dummy/demo, else use argparse
    FEATURE_FILE, DELIVERY_LABELS_FILE, output_suffix, OUTPUT_DIR = initialize()

    # output file paths

    # print(" HEY ABIN, FIX THE REDIRECTIONG OF THE OUTPUT FILE IN THE XGOOST SCRIPT! ")
    # if (os.path.basename(OUTPUT_DIR) == "icd9_90d_before"):
    #     MOD_OUTPUT_DIR = "/scratch/abraha1/xgboost_temp/icd9_90d_before"
    # elif (os.path.basename(OUTPUT_DIR) == "icd10_90d_before"):
    #     MOD_OUTPUT_DIR = "/scratch/abraha1/xgboost_temp/icd10_90d_before"
    # elif (os.path.basename(OUTPUT_DIR) == "icd9_to_phe_90d_before_redo"):
    #     MOD_OUTPUT_DIR = "/scratch/abraha1/xgboost_temp/icd9_to_phe_90d_before_redo"
    # else:
    MOD_OUTPUT_DIR = OUTPUT_DIR


    # CHANGE ALL THE MODS BACK TO OUTPUT DIR
    INPUT_DATA_FILE = os.path.join(MOD_OUTPUT_DIR, 'input_data_{}-{}.tsv'.format(output_suffix, DATE))
    HYPEROPT_TRIALS_FILE = os.path.join(MOD_OUTPUT_DIR, 'hyperopt_trials_{}-{}.tsv'.format(output_suffix, DATE))
    BEST_MODEL_FILE = os.path.join(MOD_OUTPUT_DIR, "best_xgb_model_{}-{}.pickle".format(output_suffix, DATE))
    ROC_FIG_FILE = os.path.join(MOD_OUTPUT_DIR, "roc_auc_optimized_ptb_{}-{}.png".format(output_suffix, DATE))
    PR_FIG_FILE = os.path.join(MOD_OUTPUT_DIR, "pr_auc_optimized_ptb_{}-{}.png".format(output_suffix, DATE))
    BEST_MODEL_PARAM_FILE = os.path.join(MOD_OUTPUT_DIR, "best_hyperparam_{}-{}.tsv".format(output_suffix, DATE))
    VALIDATION_FILE = os.path.join(MOD_OUTPUT_DIR, "held_out_metrics_{}-{}.tsv".format(output_suffix, DATE))
    TRAIN_SHAP_FILE = os.path.join(MOD_OUTPUT_DIR, 'train_shap_{}-{}.pickle'.format(output_suffix, DATE))
    TEST_SHAP_FILE = os.path.join(MOD_OUTPUT_DIR, 'test_shap_{}-{}.pickle'.format(output_suffix, DATE))
    TRAIN_TOP_FEAT_SHAP_FILE = os.path.join(MOD_OUTPUT_DIR, 'train_shap_top_feat_{}-{}.tsv'.format(output_suffix, DATE))
    TEST_TOP_FEAT_SHAP_FILE = os.path.join(MOD_OUTPUT_DIR, 'test_shap_top_feat_{}-{}.tsv'.format(output_suffix, DATE))
    FEATURE_IMPORT_FILE = os.path.join(MOD_OUTPUT_DIR, 'feature_importance_{}-{}.tsv'.format(output_suffix, DATE))
    FEATURE_FIG_FILE = os.path.join(MOD_OUTPUT_DIR, 'plot_feat_importance_{}-{}.png'.format(output_suffix, DATE))

    # set up logging file
    log_file = open(os.path.join(MOD_OUTPUT_DIR,'{}-{}.log'.format(output_suffix,DATE)), 'w')
    log_file.writelines("{}\n\n".format(sys.argv))


    log_file.write('FEATURE_FILE: {}'.format(FEATURE_FILE)+"\n")
    log_file.write('DELIVERY_LABELS_FILE: {}'.format(DELIVERY_LABELS_FILE)+"\n")
    log_file.write('output_suffix: {}'.format(output_suffix)+"\n")
    log_file.write('OUTPUT_DIR: {}'.format(OUTPUT_DIR)+"\n")
    log_file.flush()


    ###
    #   LOAD DATA
    ###

    root_path = os.path.dirname((FEATURE_FILE))
    dtrain_file  = os.path.join(root_path,  os.path.basename(FEATURE_FILE).replace('_feat_mat.tsv', '_dtrain.dmatrix'))
    # annotated_df_file = os.path.join(root_path,  os.path.basename(FEATURE_FILE).replace('_feat_mat.tsv', '_annotated.tsv.feather'))
    annotated_df_file = os.path.join(root_path,  os.path.basename(FEATURE_FILE).replace('_feat_mat.tsv', '_annotated.tsv.pickle'))

    log_file.write("Created and saved DMatrix files.\n")

    if (not os.path.isfile(dtrain_file)) or not (os.path.isfile(annotated_df_file)):
        print("either the Dmatrix training file or test set is not found..")
        sys.exit("either the Dmatrix training file or test set is not found..")

    # ! # ! # ! # ! #! # ! # ! # ! # !
    # ! # ! # ! TESTING ONLY ! # ! # !
    # ! # ! # ! # ! #! # ! # ! # ! # !
    # DELIVERY_LABELS_FILE = "/dors/capra_lab/users/abraha1/projects/PTB_phenotyping/data/combined_cohorts_07_12_2018/full_dataset_characterization/est_delivery_date_at_least_one_icd_cpt_ega.tsv"
    # OUTPUT_DIR = "/dors/capra_lab/users/abraha1/projects/PTB_phenotyping/scripts/rand_forest_ptb_classification/xgboost_rf/mongo/slurm_out"
    # FEATURE_FILE = "/dors/capra_lab/users/abraha1/projects/PTB_phenotyping/data/ptb_predict_machine_learning/feature_matrices/icd_cpt/all_icd9_cpt_count_subset_feat_mat.tsv"
    # output_suffix = "all_icd9_cpt_raw_count_test"
    # dtrain_file="/dors/capra_lab/users/abraha1/projects/PTB_phenotyping/data/ptb_predict_machine_learning/feature_matrices/icd_cpt/all_icd_cpt_dtrain_500.buffer"

    ###
    #   HYPER-PARAMETER TUNING W/ HYPEROPT
    ###


    # =============  set up hyperparamters =============
    # REMEMBER TO UPDATE PARAMETERS IN FORMAT FUNCTION
    space = {
        "max_depth": hp.randint('max_depth', 98),
        "min_child_weight": hp.quniform('min_child_weight', 1, 25, 5),  # min samples*wt require at node for split
        "subsample": hp.uniform('subsample', 0.5, 1),
        "colsample_bytree": hp.uniform('colsample_bytree', 0.5, 1),
        "eta": 2*hp.loguniform('eta', np.log(0.0001), np.log(0.2)),
        "gamma": 2*hp.loguniform('gamma', np.log(0.0001), np.log(0.2)),
        "alpha": 2*hp.loguniform('alpha', np.log(0.0001), np.log(0.4)),
        "lambda":2*hp.loguniform('lambda', np.log(0.0001), np.log(0.4)),
        'booster': 'gbtree',
        'verbosity': 1,
        'seed': 32,
        'objective': 'binary:logistic',
        "num_boost_rounds": hp.quniform('n_estimators', 3, 75, 2),
        "eval_metric": 'map'}

    # ============= hyperparameter optimization =============

    seed = 2019
    exp_name = output_suffix
    db_name = os.path.basename(OUTPUT_DIR)

    print("DB_NAME: {}".format(db_name))
    log_file.write("About to launch mongo jobs...\n\tdb name: {}\n\texp_key: {}\n".format(db_name, exp_name))
    log_file.flush()
    # trials = MongoTrials('mongo://vgi01.vampire:1234/{}/jobs'.format(db_name), exp_key=exp_name)
    trials = MongoTrials('mongo://capra1.vampire:1234/{}/jobs'.format(db_name), exp_key=exp_name)
    # trials = MongoTrials('mongo://cn1277:1234/{}/jobs'.format(db_name), exp_key=exp_name)


    partial_obj = partial(objective, dtrain_file=dtrain_file)
    fmin_settings = {'max_evals': 1000,
                    'algo': tpe.suggest,
                    'rstate': np.random.RandomState(seed)}


    _ = fmin(fn=partial_obj, space=space,
                       algo=fmin_settings['algo'],
                       max_evals=fmin_settings['max_evals'],
                       trials=trials, rstate=fmin_settings['rstate'],
                       verbose=1)

    log_file.write("\tDone w/ hyperopt. Took {} minutes".format(round((time.time() - start_m)/60, 2)))
    log_file.flush()
    # =============  optimize num_rounds  =============

    # get the best set of hyperparameters
    trial_df = pd.DataFrame([flatten_trial_results(x) for x in trials.results])
    sorted_df = trial_df.sort_values(by='loss').reset_index(drop=True)
    best_params = sorted_df.loc[0, list(space.keys())].to_dict()

    # rerun xgb.cv to optimize num_rounds
    optim_booster_params = {your_key:best_params[your_key] for your_key in BOOSTER_PARAMS}
    opt_rounds = xgb.cv(optim_booster_params, xgb.DMatrix(dtrain_file),
                        nfold=3, stratified=True,
                        metrics=best_params['eval_metric'],
                        early_stopping_rounds=8, num_boost_round=best_params['num_boost_rounds'],
                        fpreproc=fpreproc,
                        verbose_eval=1,
                        seed=best_params['seed'] )

    # update
    best_params['num_boost_rounds'] = opt_rounds.shape[0]


    ###
    #   VALIDATE ON HELD OUT SET
    ###

    # load held out data
    X_train, y_train, X_test, y_test, annotated_df = load_held_out_data(annotated_df_file)

    # fit
    best_params['scale_pos_weight'] =  np.sum(y_train == 0)/np.sum(y_train == 1)
    sk_params = to_sklearn_params(best_params)
    log_file.write("Fitting train data with best model...")
    best_xgb_rf = XGBClassifier(**sk_params)
    best_xgb_rf.fit(X_train, y_train)

    # predict on  held out
    metrics_results, metrics_df, model_params = validate_best_model(best_xgb_rf, X_test, y_test)

    # plot curves
    pos_prop = np.sum(y_test == 1)/len(y_test)
    plot_roc([metrics_results['fpr']], [metrics_results['tpr']], [metrics_results['roc_auc']],
             output_suffix, roc_fig_file=ROC_FIG_FILE)
    plot_pr([metrics_results['pr_curve']], [metrics_results['rc_curve']], [metrics_results['avg_prec']],
            output_suffix, pr_fig_file=PR_FIG_FILE, pos_prop=pos_prop)


    ###
    #   FEATURE IMPORTANCE
    ###

    feat_df = get_feature_importance(best_xgb_rf, annotated_df, cols_to_drop=['GRID', 'label', 'partition'])
    barplot_feat_importance(feat_df, top_n=25, plt_prefix=output_suffix, fig_file=FEATURE_FIG_FILE)

    # shap values
    train_df, _ = extract_train_df(annotated_df)
    test_df, _ = extract_test_df(annotated_df)

    xgb_booster = best_xgb_rf.get_booster()
    calc_write_shap(X_train, y_train, xgb_booster, train_df, shap_pickle_file=TRAIN_SHAP_FILE, top_feat_file=TRAIN_TOP_FEAT_SHAP_FILE)
    calc_write_shap(X_test, y_test, xgb_booster, test_df, shap_pickle_file=TEST_SHAP_FILE, top_feat_file=TEST_TOP_FEAT_SHAP_FILE)



    ###
    #   WRITE
    ###

    # write model
    pickle.dump(best_xgb_rf, open(BEST_MODEL_FILE, 'wb'))

    # write dataset used with train/test annotated
    annotated_df.to_csv(INPUT_DATA_FILE, sep="\t", index=False)

    # write tuning_summary
    trial_df.to_csv(HYPEROPT_TRIALS_FILE, sep="\t", index=False)

    # write model performance

    # write final model evaluation
    metrics_df.to_csv(VALIDATION_FILE, sep="\t", index=False)

    # write feature importance
    feat_df.to_csv(FEATURE_IMPORT_FILE, sep="\t", index=False)

    # write run summary and model details
    with open(BEST_MODEL_PARAM_FILE, 'w') as fopen:
        fopen.write("Trained and optimized XGBoost Random Forest with Hyperopt to predict PTB vs. non-PTB.\n")
        fopen.write("Run details:\n\t{}\n".format(output_suffix))
        fopen.write("Feature file used:\n\t{}\n".format(FEATURE_FILE))
        fopen.write("Label file used:\n\t{}\n".format(DELIVERY_LABELS_FILE))
        fopen.write("Generated on {}\n\n".format(DATE))
        fopen.write("Random Forest Model Parameters:\n")

        for key, value in model_params.items():
            fopen.write("{}:{}\n".format(key, value))
        fopen.write("-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#:\n".format(DATE))
        fopen.write("Hyperopt Settings:\n".format(DATE))
        for key, value in fmin_settings.items():
            fopen.write("{}:{}\n".format(key, value))

    log_file.write("\tWrote model settings to:\n\t{}".format(BEST_MODEL_FILE))
    end_m = time.time()
    log_file.write("Output files written to:\n{}".format(OUTPUT_DIR))
    log_file.write("DONE, took {:.2f} minutes.".format((end_m-start_m)/60))
    log_file.close()
