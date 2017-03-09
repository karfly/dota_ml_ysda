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


def add_hero_features(train_df, test_df,
                      add_heroes_by_player=False,
                      add_heroes_by_team=False):
    print('Adding one-hot encoding hero features...')
    heroes_df = pd.read_csv(os.path.join(data_dir_path, 'heroes.csv'), index_col='mid', dtype='object')
    heroes_df_ohe = pd.get_dummies(heroes_df.add_suffix('_is_hero'))

    team_heroes_df = pd.DataFrame(index=heroes_df.index)
    n_heroes = len(heroes_df['player_0'].unique())
    for hero_id in range(n_heroes):
        team_heroes_df['radiant_has_hero_{}'.format(hero_id)] = sum(
            [heroes_df_ohe['{}_is_hero_{}'.format(player_id, hero_id)] for player_id in radiant_player_columns])
        team_heroes_df['dire_has_hero_{}'.format(hero_id)] = sum(
            [heroes_df_ohe['{}_is_hero_{}'.format(player_id, hero_id)] for player_id in dire_player_columns])

    # Merging
    if add_heroes_by_player:
        train_df = _merge_df_by_index(train_df, heroes_df_ohe)
        test_df = _merge_df_by_index(test_df, heroes_df_ohe)

    if add_heroes_by_team:
        train_df = _merge_df_by_index(train_df, team_heroes_df)
        test_df = _merge_df_by_index(test_df, team_heroes_df)

    return train_df, test_df


def transform_data(gold_features=False,
                   lh_features=False,
                   xp_features=False,
                   add_heroes_by_player=False,
                   add_heroes_by_team=False):
    train_df = pd.read_csv(os.path.join(data_dir_path, 'train.csv'), index_col='mid')
    test_df = pd.read_csv(os.path.join(data_dir_path, 'test.csv'), index_col='mid')

    if gold_features:
        train_df, test_df = add_temporal_features(train_df, test_df, 'gold')

    if lh_features:
        train_df, test_df = add_temporal_features(train_df, test_df, 'lh')

    if xp_features:
        train_df, test_df = add_temporal_features(train_df, test_df, 'xp')

    if add_heroes_by_player or add_heroes_by_team:
        train_df, test_df = add_hero_features(train_df, test_df,
                                              add_heroes_by_player=add_heroes_by_player,
                                              add_heroes_by_team=add_heroes_by_team)

    return train_df, test_df
