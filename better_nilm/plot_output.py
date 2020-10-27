import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import re

from itertools import groupby
from operator import itemgetter

import seaborn as sns
sns.set_style('white')


PATH_DATA = '../data'
PATH_OUTPUT = '../outputs'

assert os.path.isdir(PATH_DATA)
assert os.path.isdir(PATH_OUTPUT)

DICT_APPLIANCES = {'dish_washer': 'Dishwasher',
                  'fridge': 'Fridge',
                  'washing_machine': 'Washing machine'}
DICT_MODEL = {'TPNILM': 'CONV',
             'BiGRU': 'GRU'}
DICT_MAE = {'Dishwasher': [5, 25],
           'Fridge': [20, 35],
           'Washing machine': [10, 45]}


def list_dir_sorted(path_input, model):
    files = []
    for i in os.listdir(path_input):
        if (os.path.isdir(os.path.join(path_input, i))
            and model in i):
            files.append((os.path.join(path_input, i),
                          float(re.search('clas_(.*)_reg', i).group(1))))
    files.sort(key=lambda tup: tup[1])
    return files


def list_scores(path_dir):
    list_scores_txt = []
    for i in os.listdir(path_dir):
        if (os.path.isfile(os.path.join(path_dir, i))
            and 'scores_' in i):
            list_scores_txt.append(os.path.join(path_dir, i))
    return list_scores_txt


def extract_scores_from_file(path_file):
    with open(path_file, 'r') as search:
        next_line_is = None
        approach = None
        appliance = None
        scores = {}
        for line in search:
            line = line.rstrip()
            if line.startswith('==='):
                next_line_is = 'approach'
                approach = None
            elif line.startswith('---'):
                next_line_is = 'appliance'
                appliance = None
            elif next_line_is == 'approach':
                approach = line
                scores[approach] = {}
                next_line_is = None
            elif next_line_is == 'appliance':
                appliance = line
                scores[approach][appliance] = {}
                next_line_is = 'score'
            elif next_line_is == 'score':
                key = line.split(': ', 1)[0]
                value = float(line.split(': ', 1)[1])
                scores[approach][appliance][key] = value
    return scores


def get_f1_mae_from_scores(list_scores_txt):
    list_f1 = []
    list_mae = []
    for path_file in list_scores_txt:
        dict_scores = extract_scores_from_file(path_file)
        # Include F1 from classification
        # If missing, get F1 from regression
        if 'classification' in dict_scores.keys():
            for app, dic in dict_scores['classification'].items():
                list_f1.append((app, dic['f1']))
        elif 'regression' in dict_scores.keys():
            for app, dic in dict_scores['regression'].items():
                list_f1.append((app, dic['f1']))
        # Include MAE from regression
        # If missing, get MAE from classification
        if 'regression' in dict_scores.keys():
            for app, dic in dict_scores['regression'].items():
                list_mae.append((app, dic['mae']))
        elif 'classification' in dict_scores.keys():
            for app, dic in dict_scores['classification'].items():
                list_mae.append((app, dic['mae']))
    list_f1.sort(key=lambda tup: tup[0])
    dict_f1 = dict([(k, list(list(zip(*g))[1]))
                    for k, g in groupby(list_f1, itemgetter(0))])
    list_mae.sort(key=lambda tup: tup[0])
    dict_mae = dict([(k, list(list(zip(*g))[1]))
                     for k, g in groupby(list_mae, itemgetter(0))])
    return dict_f1, dict_mae


def get_arrays(list_values):
    w = np.zeros(len(list_values))
    mean = np.zeros(len(list_values))
    std = np.zeros(len(list_values))
    for i, (weight, values) in enumerate(list_values):
        w[i] = weight
        mean[i] = np.mean(values)
        std[i] = np.std(values)
    return w, mean, std


def moving_average(a, n=3) :
    ret = np.cumsum(a)
    ret[n:] = ret[n:] - ret[:-n]
    ret[n - 1:] = ret[n - 1:] / n
    return ret
 

def subplot_f1(ax, w, y, std, ignore_extreme):
    color = 'tab:red'
    ax.set_xlabel('Classification weight')
    ax.set_ylabel('F1', color=color)
    up = y + std
    down = y - std
    if ignore_extreme or w.min() != 0:
        ax.plot(w[1:], y[1:], color=color)
        ax.fill_between(w[1:], down[1:], up[1:],
                        color=color, alpha=0.2)
    else:
        ax.plot(w[1:], y[1:], color=color)
        ax.fill_between(w[1:], down[1:], up[1:],
                        color=color, alpha=0.2)
        ax.errorbar(w[0], y[0], std[0], color=color,
                    linestyle='None', marker='.')
    ax.tick_params(axis='y', labelcolor=color)
    ax.grid(axis='y')
    ax.grid(axis='x')
    ax.set_ylim([.5, 1])
    return ax


def subplot_mae(ax, w, y, std, ignore_extreme, app):
    color = 'tab:blue'
    ax.set_ylabel('MAE (watts)', color=color)  # we already handled the x-label with ax1
    up = y + std
    down = y - std
    if ignore_extreme or w.max() != 1:
        ax.plot(w[:-1], y[:-1], color=color)
        ax.fill_between(w[:-1], down[:-1], up[:-1],
                        color=color, alpha=0.2)
    else:
        ax.plot(w[:-1], y[:-1], color=color)
        ax.fill_between(w[:-1], down[:-1], up[:-1],
                        color=color, alpha=0.2)
        ax.errorbar(w[-1], y[-1], std[-1], color=color,
                    linestyle='None', marker='.')
    ax.tick_params(axis='y', labelcolor=color)
    ax.set_ylim(DICT_MAE[app])
    return ax


def plot_arrays(w_f1, f1, f1_std, w_mae, mae, mae_std,
                app, model,
                movavg=1, ignore_extreme=True, figsize=(6, 4),
               savefig=None):
    app = DICT_APPLIANCES.get(app, app)
    if movavg > 1:
        f1 = moving_average(f1, n=movavg)
        mae = moving_average(mae, n=movavg)

    fig, ax1 = plt.subplots(figsize=figsize)

    ax1 = subplot_f1(ax1, w_f1, f1, f1_std, ignore_extreme)
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    ax2 = subplot_mae(ax2, w_mae, mae, mae_std, ignore_extreme, app)
    

    ax1.set_title(model + ' ' + app)
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    if savefig is None:
        plt.show()
    else:
        plt.savefig(savefig)


def plot_weights(path_input: str, app: str,
                 model: str='seq_480_1min', movavg: int=1,
                ignore_extreme: bool=False, figsize=(8, 4),
                savefig=None):
    assert os.path.isdir(path_input)
    
    # List files and sort by class weight
    files = list_dir_sorted(path_input, model)
    
    list_f1 = []
    list_mae = []
    
    for path_dir, clas_w in files:
        list_scores_txt = list_scores(path_dir)
        dict_f1, dict_mae = get_f1_mae_from_scores(list_scores_txt)
        if app in dict_f1.keys():
            list_f1.append((clas_w, dict_f1[app]))
        if app in dict_mae.keys():
            list_mae.append((clas_w, dict_mae[app]))
    
    # Build arrays
    w_f1, f1, f1_std = get_arrays(list_f1)
    w_mae, mae, mae_std = get_arrays(list_mae)
    
    # Plot arrays
    model = path_input.rsplit('/', 1)[1]
    model = model[:-5]
    model = DICT_MODEL.get(model, model)
    plot_arrays(w_f1, f1, f1_std, w_mae, mae, mae_std,
                app, model,
                movavg=movavg, ignore_extreme=ignore_extreme,
               figsize=figsize, savefig=savefig)    
