import os

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler

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
                      add_heroes_by_team=False,
                      add_vector_heroes=False,
                      add_bigram_heroes=False):
    heroes_df = pd.read_csv(os.path.join(data_dir_path, 'heroes.csv'), index_col='mid', dtype='object')
    heroes_df_ohe = pd.get_dummies(heroes_df.add_suffix('_is_hero'))

    team_heroes_df = pd.DataFrame(index=heroes_df.index)
    n_heroes = len(heroes_df['player_0'].unique())
    for hero_id in range(n_heroes):
        team_heroes_df['radiant_has_hero_{}'.format(hero_id)] = sum(
            [heroes_df_ohe['{}_is_hero_{}'.format(player_id, hero_id)] for player_id in radiant_player_columns])
        team_heroes_df['dire_has_hero_{}'.format(hero_id)] = sum(
            [heroes_df_ohe['{}_is_hero_{}'.format(player_id, hero_id)] for player_id in dire_player_columns])

    vector_heroes_df = pd.DataFrame(index=heroes_df.index, dtype='int')
    for hero_id in range(n_heroes):
        vector_heroes_df['hero_{}'.format(hero_id)] = team_heroes_df['radiant_has_hero_{}'.format(hero_id)].astype(
            'int') - \
                                                      team_heroes_df['dire_has_hero_{}'.format(hero_id)].astype(int)

    bigram_heroes_df = pd.DataFrame(index=heroes_df.index, dtype='int')
    for hero_i in range(n_heroes):
        for hero_j in range(hero_i + 1, n_heroes):
            bigram_heroes_df['hero_{}_and_hero_{}'.format(hero_i, hero_j)] = (
                (team_heroes_df['radiant_has_hero_{}'.format(hero_i)] == 1) & \
                (team_heroes_df['radiant_has_hero_{}'.format(hero_j)] == 1)).astype('int')

            bigram_heroes_df['hero_{}_and_hero_{}'.format(hero_i, hero_j)] = (
                (team_heroes_df['dire_has_hero_{}'.format(hero_i)] == 1) & \
                (team_heroes_df['dire_has_hero_{}'.format(hero_j)] == 1)).astype('int')

    # Merging
    if add_heroes_by_player:
        print('Adding one-hot encoding hero features by player...')
        train_df = _merge_df_by_index(train_df, heroes_df_ohe)
        test_df = _merge_df_by_index(test_df, heroes_df_ohe)

    if add_heroes_by_team:
        print('Adding one-hot encoding hero features by team...')
        train_df = _merge_df_by_index(train_df, team_heroes_df)
        test_df = _merge_df_by_index(test_df, team_heroes_df)

    if add_vector_heroes:
        print('Adding vector heroes...')
        train_df = _merge_df_by_index(train_df, vector_heroes_df)
        test_df = _merge_df_by_index(test_df, vector_heroes_df)

    if add_bigram_heroes:
        print('Adding bigram heroes...')
        train_df = _merge_df_by_index(train_df, bigram_heroes_df)
        test_df = _merge_df_by_index(test_df, bigram_heroes_df)

    return train_df, test_df


def add_events_features(train_df, test_df):
    print('Adding \'events_features\'...')
    events_df = pd.read_csv(os.path.join(data_dir_path, 'events.csv'), index_col='mid', dtype='object')
    radiant_events_df_ohe = pd.get_dummies(
        events_df[events_df['from_team'] == 'radiant'][['event_type']].add_prefix('radiant_'))
    radiant_events_df_ohe = radiant_events_df_ohe.groupby(radiant_events_df_ohe.index).aggregate(sum)

    dire_events_df_ohe = pd.get_dummies(events_df[events_df['from_team'] == 'dire'][['event_type']].add_prefix('dire_'))
    dire_events_df_ohe = dire_events_df_ohe.groupby(dire_events_df_ohe.index).aggregate(sum)

    events_joined_df_ohe = radiant_events_df_ohe.join(dire_events_df_ohe, how='outer').fillna(0).astype('int')

    train_df = pd.concat([train_df, events_joined_df_ohe], axis=1, join_axes=[train_df.index]).fillna(-999)
    test_df = pd.concat([test_df, events_joined_df_ohe], axis=1, join_axes=[test_df.index]).fillna(-999)

    return train_df, test_df


def add_items_features(train_df, test_df,
                       add_items_by_player=False,
                       add_items_by_team=False,
                       add_vector_items=False,
                       add_bigram_items=False):
    items_df = pd.read_csv('data/items.csv', index_col='mid').fillna(0)

    players_dfs = []
    for player_id in range(10):
        players_dfs.append(items_df[items_df['player'] == player_id].drop('player', axis=1).add_prefix(
            'player_{}_has_'.format(player_id)))
    items_by_player_df = pd.concat(players_dfs, axis=1)

    items_by_team_df = pd.DataFrame(index=items_by_player_df.index)
    n_items = items_df.shape[1] - 1
    for item_id in range(n_items):
        items_by_team_df['radiant_has_item_{}'.format(item_id)] = sum(
            [items_by_player_df['{}_has_item_{}'.format(player_id, item_id)] for player_id in radiant_player_columns])
        items_by_team_df['dire_has_item_{}'.format(item_id)] = sum(
            [items_by_player_df['{}_has_item_{}'.format(player_id, item_id)] for player_id in dire_player_columns])

    vector_items_df = pd.DataFrame(index=items_by_player_df.index)
    for item_id in range(n_items):
        vector_items_df['item_{}'.format(item_id)] = items_by_team_df['radiant_has_item_{}'.format(item_id)] - \
                                                     items_by_team_df['dire_has_item_{}'.format(item_id)]

    bigram_items_df = pd.DataFrame(index=items_df.index, dtype='int')
    for item_i in range(n_items):
        for item_j in range(item_i + 1, n_items):
            bigram_items_df['item_{}_and_item_{}'.format(item_i, item_j)] = (
                (items_by_team_df['radiant_has_item_{}'.format(item_i)] == 1) & \
                (items_by_team_df['radiant_has_item_{}'.format(item_j)] == 1)).astype('int')

            bigram_items_df['item_{}_and_item_{}'.format(item_i, item_j)] = (
                (items_by_team_df['dire_has_item_{}'.format(item_i)] == 1) & \
                (items_by_team_df['dire_has_item_{}'.format(item_j)] == 1)).astype('int')

    # Merging
    if add_items_by_player:
        print('Adding items features by player...')
        train_df = _merge_df_by_index(train_df, items_by_player_df)
        test_df = _merge_df_by_index(test_df, items_by_player_df)

    if add_items_by_team:
        print('Adding items features by team...')
        train_df = _merge_df_by_index(train_df, items_by_team_df)
        test_df = _merge_df_by_index(test_df, items_by_team_df)

    if add_vector_items:
        print('Adding vector items features...')
        train_df = _merge_df_by_index(train_df, vector_items_df)
        test_df = _merge_df_by_index(test_df, vector_items_df)

    if add_bigram_items:
        print('Adding bigram items features...')
        train_df = _merge_df_by_index(train_df, bigram_items_df)
        test_df = _merge_df_by_index(test_df, bigram_items_df)

    return train_df.fillna(0), test_df.fillna(0)


def transform_data(scale=False,
                   gold_features=False,
                   lh_features=False,
                   xp_features=False,
                   heroes_by_player=False, heroes_by_team=False, vector_heroes=False, bigram_heroes=False,
                   events_features=False,
                   items_by_player=False, items_by_team=False, vector_items=False, bigram_items=False):
    train_df = pd.read_csv(os.path.join(data_dir_path, 'train.csv'), index_col='mid')
    test_df = pd.read_csv(os.path.join(data_dir_path, 'test.csv'), index_col='mid')

    features_to_scale = []

    # Numerical features
    if gold_features:
        train_df, test_df = add_temporal_features(train_df, test_df, 'gold')
        features_to_scale.extend([])

    if lh_features:
        train_df, test_df = add_temporal_features(train_df, test_df, 'lh')
        features_to_scale.extend([])

    if xp_features:
        train_df, test_df = add_temporal_features(train_df, test_df, 'xp')
        features_to_scale.extend([])

    if scale:
        print('Scaling...')
        X_train, y_train = train_df.drop('radiant_won', axis=1), train_df['radiant_won']
        X_test = test_df

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        train_df = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
        train_df['radiant_won'] = y_train
        test_df = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)

    # Categorical and binary features
    if heroes_by_player or heroes_by_team or vector_heroes or bigram_heroes:
        train_df, test_df = add_hero_features(train_df, test_df,
                                              add_heroes_by_player=heroes_by_player,
                                              add_heroes_by_team=heroes_by_team,
                                              add_vector_heroes=vector_heroes,
                                              add_bigram_heroes=bigram_heroes)

    if events_features:
        train_df, test_df = add_events_features(train_df, test_df)

    if items_by_player or items_by_team or vector_items or bigram_items:
        train_df, test_df = add_items_features(train_df, test_df,
                                               add_items_by_player=items_by_player,
                                               add_items_by_team=items_by_team,
                                               add_vector_items=vector_items,
                                               add_bigram_items=bigram_items)




    return train_df, test_df
