

import os, sys
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import glob
import pickle


from scipy.interpolate import interp1d
from datetime import datetime

sys.path.append('/dors/capra_lab/users/abraha1/projects/PTB_phenotyping/scripts/rand_forest_ptb_classification')
from hyperparam_tune import validate_best_model

sys.path.append("/dors/capra_lab/users/abraha1/projects/PTB_phenotyping/scripts/rand_forest_ptb_classification/manuscript/0_helper_func")
from manip_trained_models_funcs import unpack_input_data, get_preds

import matplotlib.font_manager as fm
fpath='/dors/capra_lab/users/abraha1/conda/envs/py36_r_ml/lib/python3.6/site-packages/matplotlib/mpl-data/fonts/ttf/Arial.ttf'
bprop = fm.FontProperties(fname=fpath, size=10)
sprop = fm.FontProperties(fname=fpath, size=8)

import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42



###
### FUNCTIONS
###



def set_up_manu_roc(fpath, figsize=(2.25,2.25)):
    
    sns.set( style='ticks',  font_scale=1.0, rc={'figure.figsize':(2.25,2.25)} )
    sns.set_style( {'axes.grid': True, 'axes.edgecolor': 'k', 'axes.linewidth': 10,  'grid.color': '#e1e1e1'})
    sprop = fm.FontProperties(fname=fpath, size=6, )


    # plot
    fig ,ax = plt.subplots()
    
    return ax,sprop,


def manu_roc_format(ax, sprop, type, legend_loc, lg_title=None):    

    if type == "roc":
        _ = ax.set_xlabel('False Positive Rate', fontproperties=sprop,labelpad=0.59)
        _ = ax.set_ylabel('True Positive Rate', fontproperties=sprop,labelpad=0.59)
        if not lg_title: 
            lg_title="Features (ROC-AUC)"
    elif type == "pr":     
        _ = ax.set_xlabel('Recall', fontproperties=sprop,labelpad=0.59)
        _ = ax.set_ylabel('Precision', fontproperties=sprop,labelpad=0.59)
        if not lg_title: 
            lg_title="Features (PR-AUC)"
        
    _ = lg = ax.legend(prop=sprop, facecolor='white', edgecolor='white', frameon=True, fancybox=False, framealpha=1, shadow=False, borderpad=0.25, loc=legend_loc)

    _ = lg.set_title(title=lg_title, prop=sprop)
    _ = lg._legend_box.align = "left"
    _ = ax.tick_params(width=0.5, length=2.5)

    ax.grid(axis='both', linewidth=0.5, color='gainsboro')
    ax.spines['left'].set_linewidth(0.5)
    ax.spines['bottom'].set_linewidth(0.5)
    ax.spines['top'].set_linewidth(0.5)
    ax.spines['right'].set_linewidth(0.5)

    ax.set_xlim(0,1.0)
    ax.set_ylim(0,1.0)
    ax.set_xticks(np.arange(0,1.2,0.2))
    ax.set_yticks(np.arange(0,1.2,0.2))
    ax.set_xticklabels(['{:.1f}'.format(x) for x in np.arange(0,1.2,0.2)], fontproperties=sprop)
    ax.set_yticklabels(['{:.1f}'.format(x) for x in np.arange(0,1.2,0.2)], fontproperties=sprop)
    ax.set_aspect('equal','box')


    plt.subplots_adjust(left=0.15,right=0.95, top=.95, bottom=0.10)
    
    return ax 

