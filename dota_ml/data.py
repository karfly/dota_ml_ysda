import os

import numpy as np
import pandas as pd


data_url = 'https://dl.dropboxusercontent.com/u/67618204/dota_ml/data.tar.gz'
data_dir_path = 'data/'

player_columns = ['player_{}'.format(i) for i in range(10)]
radiant_player_columns = player_columns[:5]
dire_player_columns = player_columns[5:]


def _merge_df_by_index(df_left, df_right):
    return pd.merge(df_left, df_right, left_index=True, right_index=True)


def add_temporal_features(train_df, test_df, feature):
    """
    Adds temporal features to dataset
    :param feature: can be 'gold', 'lh', 'xp'
    :return:
    """
    assert feature in ['gold', 'lh', 'xp'], '{} - unknown feature'.format(feature)

    filename = '{}.csv'.format(feature)
    df = pd.read_csv(os.path.join(data_dir_path, filename), index_col='mid')

    # Last features
    print('Adding \'last_{}_by_player\'...'.format(feature))
    last_feature_by_player_df = df.groupby(df.index).agg(lambda x: x.values[-1])[player_columns].add_prefix(
        'last_{}_'.format(feature))
    train_df = _merge_df_by_index(train_df, last_feature_by_player_df)
    test_df = _merge_df_by_index(test_df, last_feature_by_player_df)

    print('Adding \'last_{}_by_team\'...'.format(feature))
    last_feature_radiant_df = last_feature_by_player_df[
        ['last_{}_{}'.format(feature, player) for player in radiant_player_columns]].sum(axis=1).to_frame().rename(
        columns={0: 'last_{}_radiant'.format(feature)})
    last_feature_dire_df = last_feature_by_player_df[
        ['last_{}_{}'.format(feature, player) for player in dire_player_columns]].sum(axis=1).to_frame().rename(
        columns={0: 'last_{}_dire'.format(feature)})
    last_feature_by_team_df = _merge_df_by_index(last_feature_radiant_df, last_feature_dire_df)
    train_df = _merge_df_by_index(train_df, last_feature_by_team_df)
    test_df = _merge_df_by_index(test_df, last_feature_by_team_df)

    # Gold speed features
    print('Adding \'{}_speed_by_player\'...'.format(feature))
    feature_speed_by_player_df = df.groupby(df.index).agg(lambda x: np.diff(x.values).mean())[
        player_columns].add_prefix('{}_speed_'.format(feature))
    train_df = _merge_df_by_index(train_df, feature_speed_by_player_df)
    test_df = _merge_df_by_index(test_df, feature_speed_by_player_df)

    print('Adding \'{}_speed_by_team\'...'.format(feature))
    feature_speed_radiant_df = feature_speed_by_player_df[
        ['{}_speed_{}'.format(feature, player) for player in radiant_player_columns]].mean(axis=1).to_frame().rename(
        columns={0: '{}_speed_radiant'.format(feature)})
    feature_speed_dire_df = feature_speed_by_player_df[
        ['{}_speed_{}'.format(feature, player) for player in dire_player_columns]].mean(axis=1).to_frame().rename(
        columns={0: '{}_speed_dire'.format(feature)})
    feature_speed_by_team_df = _merge_df_by_index(feature_speed_radiant_df, feature_speed_dire_df)
    train_df = _merge_df_by_index(train_df, feature_speed_by_team_df)
    test_df = _merge_df_by_index(test_df, feature_speed_by_team_df)

    # Max gold features
    print('Adding \'max_{}_by_player\'...'.format(feature))
    max_feature_by_player_df = df.groupby(df.index).agg(lambda x: x.values.max())[player_columns].add_prefix(
        'max_{}_'.format(feature))
    train_df = _merge_df_by_index(train_df, max_feature_by_player_df)
    test_df = _merge_df_by_index(test_df, max_feature_by_player_df)

    print('Adding \'max_{}_by_team\'...'.format(feature))
    max_feature_radiant_df = max_feature_by_player_df[
        ['max_{}_{}'.format(feature, player) for player in radiant_player_columns]].max(axis=1).to_frame().rename(
        columns={0: 'max_{}_radiant'.format(feature)})
    max_feature_dire_df = max_feature_by_player_df[
        ['max_{}_{}'.format(feature, player) for player in dire_player_columns]].max(axis=1).to_frame().rename(
        columns={0: 'max_{}_dire'.format(feature)})
    max_feature_by_team_df = _merge_df_by_index(max_feature_radiant_df, max_feature_dire_df)
    train_df = _merge_df_by_index(train_df, max_feature_by_team_df)
    test_df = _merge_df_by_index(test_df, max_feature_by_team_df)

    return train_df, test_df


def transform_data(gold_features=False, lh_features=False, xp_features=False):
    train_df = pd.read_csv(os.path.join(data_dir_path, 'train.csv'), index_col='mid')
    test_df = pd.read_csv(os.path.join(data_dir_path, 'test.csv'), index_col='mid')

    if gold_features:
        train_df, test_df = add_temporal_features(train_df, test_df, 'gold')

    if lh_features:
        train_df, test_df = add_temporal_features(train_df, test_df, 'lh')

    if xp_features:
        train_df, test_df = add_temporal_features(train_df, test_df, 'xp')

    return train_df, test_df
