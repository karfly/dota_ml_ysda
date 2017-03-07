import os
from datetime import datetime
import json

import numpy as np
import pandas as pd

from sklearn.model_selection import ParameterGrid

import matplotlib.pylab as plt


def make_submission(test_df, model, root, title, params={}, score=None):
    predictions = model.predict_proba(test_df)[:, 1]
    submission = pd.DataFrame({'mid': test_df.index,
                               'radiant_won': predictions})

    datetime_now = datetime.now()

    dir_name = title
    dir_name += '[score={:.5}]'.format(score) if score is not None else ''
    dir_name += '[{}]'.format(datetime_now.strftime('%d-%m-%Y %H:%M:%S'))
    dir_path = os.path.join(root, dir_name)
    os.makedirs(dir_path)

    # submission.csv
    submission.to_csv(os.path.join(dir_path, 'submission.csv'), index=False)

    # params.json
    with open(os.path.join(dir_path, 'params.json'), 'w') as fout:
        json.dump(params, fout, indent=4)

    # TODO: Dump model here


def plot_feature_ranking(importances, feature_names, max_n_importances=20):
    fig, ax = plt.subplots(figsize=(10, min(len(importances), max_n_importances) * 0.5))

    importances_sorted_indexes = np.argsort(importances)[::-1][:max_n_importances][::-1]
    importances_sorted = importances[importances_sorted_indexes]

    ax.barh(range(len(importances_sorted)), importances_sorted,
            color='blue', align='center')

    ax.set_title('Feature ranking', fontsize=20)
    ax.set_yticks(range(len(importances_sorted)))
    ax.set_yticklabels(feature_names[importances_sorted_indexes], fontsize=15)
    ax.set_ylim([-1, len(importances_sorted)])
    ax.set_xlabel("importance", fontsize=18)


def generate_grid(param_grid, grid_max_size=None):
    grid = list(ParameterGrid(param_grid))
    if grid_max_size is not None:
        grid = np.random.choice(grid, size=min(len(grid), grid_max_size), replace=False)

    return grid
