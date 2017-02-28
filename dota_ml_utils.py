import os
from datetime import datetime

import numpy as np
import pandas as pd


def transform_data(data_dir_path,
                   last_gold_by_player=True,
                   last_gold_by_team=True,
                   gold_speed_by_player=True,
                   gold_speed_by_team=True):
    train_df = pd.read_csv(os.path.join(data_dir_path, 'train.csv'), index_col='mid')
    test_df = pd.read_csv(os.path.join(data_dir_path, 'test.csv'), index_col='mid')

    player_columns = ['player_{}'.format(i) for i in range(10)]
    radiant_player_columns = player_columns[:5]
    dire_player_columns = player_columns[5:]

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


    return train_df, test_df


def make_submission(test_df, model, path):
    predictions = model.predict_proba(test_df)[:, 1]
    submission = pd.DataFrame({'mid': test_df.index,
                               'radiant_won': predictions})
    submission.to_csv(path, index=False)


def generate_csv_path(root, title, params={}, score=None, add_time=True):
    filename = title
    filename += str(params) if params != {} else ''
    filename += '[score={:.4}]'.format(score) if score is not None else ''
    filename += '[{}]'.format(datetime.now().strftime('%d-%m-%Y %H:%M:%S') if add_time else '')
    filename += '.csv'
    return os.path.join(root, filename)
