import os
from datetime import datetime
import json

import numpy as np
import pandas as pd

from sklearn.model_selection import ParameterGrid

import matplotlib.pylab as plt


# Constants
DATA_URL = 'https://dl.dropboxusercontent.com/u/67618204/dota_ml/data.tar.gz'

player_columns = ['player_{}'.format(i) for i in range(10)]
radiant_player_columns = player_columns[:5]
dire_player_columns = player_columns[5:]


def add_gold_features(data_dir_path, train_df, test_df,
                      last_gold_by_player=True, last_gold_by_team=True,
                      gold_speed_by_player=True, gold_speed_by_team=True,
                      max_gold_by_player=True, max_gold_by_team=True):
    if last_gold_by_player:
        print('Adding \'last_gold_by_player\'...')
        gold_df = pd.read_csv(os.path.join(data_dir_path, 'gold.csv'), index_col='mid')
        last_gold_by_player_df = gold_df.groupby(gold_df.index).agg(lambda x: x.values[-1])[player_columns].add_prefix(
            'last_gold_')
        train_df = pd.merge(train_df, last_gold_by_player_df, left_index=True, right_index=True)
        test_df = pd.merge(test_df, last_gold_by_player_df, left_index=True, right_index=True)

    if last_gold_by_team:
        print('Adding \'last_gold_by_team\'...')
        if not last_gold_by_player:
            gold_df = pd.read_csv(os.path.join(data_dir_path, 'gold.csv'), index_col='mid')
            last_gold_by_player_df = gold_df.groupby(gold_df.index).agg(lambda x: x.values[-1])[
                player_columns].add_prefix('last_gold_')

        last_gold_radiant_df = last_gold_by_player_df[
            ['last_gold_{}'.format(player) for player in radiant_player_columns]].sum(axis=1).to_frame().rename(
            columns={0: 'last_gold_radiant'})
        last_gold_dire_df = last_gold_by_player_df[
            ['last_gold_{}'.format(player) for player in dire_player_columns]].sum(axis=1).to_frame().rename(
            columns={0: 'last_gold_dire'})

        last_gold_by_team_df = pd.merge(last_gold_radiant_df, last_gold_dire_df, left_index=True, right_index=True)
        train_df = pd.merge(train_df, last_gold_by_team_df, left_index=True, right_index=True)
        test_df = pd.merge(test_df, last_gold_by_team_df, left_index=True, right_index=True)

    if gold_speed_by_player:
        print('Adding \'gold_speed_by_player\'...')
        gold_df = pd.read_csv(os.path.join(data_dir_path, 'gold.csv'), index_col='mid')
        gold_speed_by_player_df = gold_df.groupby(gold_df.index).agg(lambda x: np.diff(x.values).mean())[
            player_columns].add_prefix('gold_speed_')
        train_df = pd.merge(train_df, gold_speed_by_player_df, left_index=True, right_index=True)
        test_df = pd.merge(test_df, gold_speed_by_player_df, left_index=True, right_index=True)

    if gold_speed_by_team:
        print('Adding \'gold_speed_by_team\'...')
        if not gold_speed_by_player:
            gold_df = pd.read_csv(os.path.join(data_dir_path, 'gold.csv'), index_col='mid')
            gold_speed_by_player_df = gold_df.groupby(gold_df.index).agg(lambda x: np.diff(x.values).mean())[
                player_columns].add_prefix('gold_speed_')

        gold_speed_radiant_df = gold_speed_by_player_df[
            ['gold_speed_{}'.format(player) for player in radiant_player_columns]].mean(axis=1).to_frame().rename(
            columns={0: 'gold_speed_radiant'})

        gold_speed_dire_df = gold_speed_by_player_df[
            ['gold_speed_{}'.format(player) for player in dire_player_columns]].mean(axis=1).to_frame().rename(
            columns={0: 'gold_speed_dire'})

        gold_speed_by_team_df = pd.merge(gold_speed_radiant_df, gold_speed_dire_df, left_index=True, right_index=True)
        train_df = pd.merge(train_df, gold_speed_by_team_df, left_index=True, right_index=True)
        test_df = pd.merge(test_df, gold_speed_by_team_df, left_index=True, right_index=True)

    if max_gold_by_player:
        print('Adding \'max_gold_by_player\'...')
        gold_df = pd.read_csv(os.path.join(data_dir_path, 'gold.csv'), index_col='mid')
        max_gold_by_player_df = gold_df.groupby(gold_df.index).agg(lambda x: x.values.max())[player_columns].add_prefix(
            'max_gold_')
        train_df = pd.merge(train_df, max_gold_by_player_df, left_index=True, right_index=True)
        test_df = pd.merge(test_df, max_gold_by_player_df, left_index=True, right_index=True)

    if max_gold_by_team:
        print('Adding \'max_gold_by_team\'...')
        if not max_gold_by_player:
            gold_df = pd.read_csv(os.path.join(data_dir_path, 'gold.csv'), index_col='mid')
            max_gold_by_player_df = gold_df.groupby(gold_df.index).agg(lambda x: x.values.max())[
                player_columns].add_prefix('max_gold_')

        max_gold_radiant_df = max_gold_by_player_df[
            ['max_gold_{}'.format(player) for player in radiant_player_columns]].max(axis=1).to_frame().rename(
            columns={0: 'max_gold_radiant'})
        max_gold_dire_df = max_gold_by_player_df[
            ['max_gold_{}'.format(player) for player in dire_player_columns]].max(axis=1).to_frame().rename(
            columns={0: 'max_gold_dire'})

        max_gold_by_team_df = pd.merge(max_gold_radiant_df, max_gold_dire_df, left_index=True, right_index=True)
        train_df = pd.merge(train_df, max_gold_by_team_df, left_index=True, right_index=True)
        test_df = pd.merge(test_df, max_gold_by_team_df, left_index=True, right_index=True)

    return train_df, test_df


def add_lh_features(data_dir_path, train_df, test_df,
                    last_lh_by_player=True, last_lh_by_team=True,
                    lh_speed_by_player=True, lh_speed_by_team=True,
                    max_lh_by_player=True, max_lh_by_team=True):
    if last_lh_by_player:
        print('Adding \'last_lh_by_player\'...')
        lh_df = pd.read_csv(os.path.join(data_dir_path, 'lh.csv'), index_col='mid')
        last_lh_by_player_df = lh_df.groupby(lh_df.index).agg(lambda x: x.values[-1])[player_columns].add_prefix(
            'last_lh_')
        train_df = pd.merge(train_df, last_lh_by_player_df, left_index=True, right_index=True)
        test_df = pd.merge(test_df, last_lh_by_player_df, left_index=True, right_index=True)

    if last_lh_by_team:
        print('Adding \'last_lh_by_team\'...')
        if not last_lh_by_player:
            lh_df = pd.read_csv(os.path.join(data_dir_path, 'lh.csv'), index_col='mid')
            last_lh_by_player_df = lh_df.groupby(lh_df.index).agg(lambda x: x.values[-1])[
                player_columns].add_prefix('last_lh_')

        last_lh_radiant_df = last_lh_by_player_df[
            ['last_lh_{}'.format(player) for player in radiant_player_columns]].sum(axis=1).to_frame().rename(
            columns={0: 'last_lh_radiant'})
        last_lh_dire_df = last_lh_by_player_df[
            ['last_lh_{}'.format(player) for player in dire_player_columns]].sum(axis=1).to_frame().rename(
            columns={0: 'last_lh_dire'})

        last_lh_by_team_df = pd.merge(last_lh_radiant_df, last_lh_dire_df, left_index=True, right_index=True)
        train_df = pd.merge(train_df, last_lh_by_team_df, left_index=True, right_index=True)
        test_df = pd.merge(test_df, last_lh_by_team_df, left_index=True, right_index=True)

    if lh_speed_by_player:
        print('Adding \'lh_speed_by_player\'...')
        lh_df = pd.read_csv(os.path.join(data_dir_path, 'lh.csv'), index_col='mid')
        lh_speed_by_player_df = lh_df.groupby(lh_df.index).agg(lambda x: np.diff(x.values).mean())[
            player_columns].add_prefix('lh_speed_')
        train_df = pd.merge(train_df, lh_speed_by_player_df, left_index=True, right_index=True)
        test_df = pd.merge(test_df, lh_speed_by_player_df, left_index=True, right_index=True)

    if lh_speed_by_team:
        print('Adding \'lh_speed_by_team\'...')
        if not lh_speed_by_player:
            lh_df = pd.read_csv(os.path.join(data_dir_path, 'lh.csv'), index_col='mid')
            lh_speed_by_player_df = lh_df.groupby(lh_df.index).agg(lambda x: np.diff(x.values).mean())[
                player_columns].add_prefix('lh_speed_')

        lh_speed_radiant_df = lh_speed_by_player_df[
            ['lh_speed_{}'.format(player) for player in radiant_player_columns]].mean(axis=1).to_frame().rename(
            columns={0: 'lh_speed_radiant'})

        lh_speed_dire_df = lh_speed_by_player_df[
            ['lh_speed_{}'.format(player) for player in dire_player_columns]].mean(axis=1).to_frame().rename(
            columns={0: 'lh_speed_dire'})

        lh_speed_by_team_df = pd.merge(lh_speed_radiant_df, lh_speed_dire_df, left_index=True, right_index=True)
        train_df = pd.merge(train_df, lh_speed_by_team_df, left_index=True, right_index=True)
        test_df = pd.merge(test_df, lh_speed_by_team_df, left_index=True, right_index=True)

    if max_lh_by_player:
        print('Adding \'max_lh_by_player\'...')
        lh_df = pd.read_csv(os.path.join(data_dir_path, 'lh.csv'), index_col='mid')
        max_lh_by_player_df = lh_df.groupby(lh_df.index).agg(lambda x: x.values.max())[player_columns].add_prefix(
            'max_lh_')
        train_df = pd.merge(train_df, max_lh_by_player_df, left_index=True, right_index=True)
        test_df = pd.merge(test_df, max_lh_by_player_df, left_index=True, right_index=True)

    if max_lh_by_team:
        print('Adding \'max_lh_by_team\'...')
        if not max_lh_by_player:
            lh_df = pd.read_csv(os.path.join(data_dir_path, 'lh.csv'), index_col='mid')
            max_lh_by_player_df = lh_df.groupby(lh_df.index).agg(lambda x: x.values.max())[
                player_columns].add_prefix('max_lh_')

        max_lh_radiant_df = max_lh_by_player_df[
            ['max_lh_{}'.format(player) for player in radiant_player_columns]].max(axis=1).to_frame().rename(
            columns={0: 'max_lh_radiant'})
        max_lh_dire_df = max_lh_by_player_df[
            ['max_lh_{}'.format(player) for player in dire_player_columns]].max(axis=1).to_frame().rename(
            columns={0: 'max_lh_dire'})

        max_lh_by_team_df = pd.merge(max_lh_radiant_df, max_lh_dire_df, left_index=True, right_index=True)
        train_df = pd.merge(train_df, max_lh_by_team_df, left_index=True, right_index=True)
        test_df = pd.merge(test_df, max_lh_by_team_df, left_index=True, right_index=True)

    return train_df, test_df


def transform_data(data_dir_path,
                   last_gold_by_player=True, last_gold_by_team=True,
                   gold_speed_by_player=True, gold_speed_by_team=True,
                   max_gold_by_player=True, max_gold_by_team=True,
                   last_lh_by_player=True, last_lh_by_team=True,
                   lh_speed_by_player=True, lh_speed_by_team=True,
                   max_lh_by_player=True, max_lh_by_team=True):
    train_df = pd.read_csv(os.path.join(data_dir_path, 'train.csv'), index_col='mid')
    test_df = pd.read_csv(os.path.join(data_dir_path, 'test.csv'), index_col='mid')

    train_df, test_df = add_gold_features(data_dir_path, train_df, test_df,
                                          last_gold_by_player, last_gold_by_team,
                                          gold_speed_by_player, gold_speed_by_team,
                                          max_gold_by_player, max_gold_by_team)

    train_df, test_df = add_lh_features(data_dir_path, train_df, test_df,
                                        last_lh_by_player, last_lh_by_team,
                                        lh_speed_by_player, lh_speed_by_team,
                                        max_lh_by_player, max_lh_by_team)

    return train_df, test_df