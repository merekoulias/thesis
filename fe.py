import pandas as pd
import numpy as np

import gc

# featuretools for automated feature engineering
import featuretools as ft

import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')

from data_analysis import obj_encoding, data_correlations, get_features_distribution, correlation_scores, features_scores, score_feature_selection

from pathlib import Path

def data_folder():
    path = Path('engineered_features')
    path.mkdir(parents=True, exist_ok=True)
    return path

def read_data():
    # Read in the full datasets
    app_train = pd.read_csv('data/application_train.csv')
    app_test = pd.read_csv('data/application_test.csv')
    bureau = pd.read_csv('data/bureau.csv')
    bureau_balance = pd.read_csv('data/bureau_balance.csv')
    cash = pd.read_csv('data/POS_CASH_balance.csv')
    credit = pd.read_csv('data/credit_card_balance.csv')
    previous = pd.read_csv('data/previous_application.csv')
    installments = pd.read_csv('data/installments_payments.csv')

    return app_train, app_test, bureau, bureau_balance, cash, credit, previous, installments

def read_data_subset():
    # Read in the datasets and limit to the first 1000 rows (sorted by SK_ID_CURR) 
    # This allows us to actually see the results in a reasonable amount of time! 
    app_train = pd.read_csv('data/application_train.csv').sort_values('SK_ID_CURR').reset_index(drop = True).loc[:1000, :]
    app_test = pd.read_csv('data/application_test.csv').sort_values('SK_ID_CURR').reset_index(drop = True).loc[:1000, :]
    bureau = pd.read_csv('data/bureau.csv').sort_values(['SK_ID_CURR', 'SK_ID_BUREAU']).reset_index(drop = True).loc[:1000, :]
    bureau_balance = pd.read_csv('data/bureau_balance.csv').sort_values('SK_ID_BUREAU').reset_index(drop = True).loc[:1000, :]
    cash = pd.read_csv('data/POS_CASH_balance.csv').sort_values(['SK_ID_CURR', 'SK_ID_PREV']).reset_index(drop = True).loc[:1000, :]
    credit = pd.read_csv('data/credit_card_balance.csv').sort_values(['SK_ID_CURR', 'SK_ID_PREV']).reset_index(drop = True).loc[:1000, :]
    previous = pd.read_csv('data/previous_application.csv').sort_values(['SK_ID_CURR', 'SK_ID_PREV']).reset_index(drop = True).loc[:1000, :]
    installments = pd.read_csv('data/installments_payments.csv').sort_values(['SK_ID_CURR', 'SK_ID_PREV']).reset_index(drop = True).loc[:1000, :]
    
    return app_train, app_test, bureau, bureau_balance, cash, credit, previous, installments

def create_entity_set(app, bureau, bureau_balance, cash, credit, previous, installments):
    es = ft.EntitySet(id = 'clients')

    # Entities with a unique index
    es.add_dataframe(dataframe=app, dataframe_name='app', index='SK_ID_CURR')

    es.add_dataframe(dataframe=bureau, dataframe_name='bureau', index='SK_ID_BUREAU')

    es.add_dataframe(dataframe=previous, dataframe_name='previous', index='SK_ID_PREV')

    # Entities that do not have a unique index
    es.add_dataframe(dataframe=bureau_balance, dataframe_name='bureau_balance', make_index=True, index='bureaubalance_index')

    es.add_dataframe(dataframe=cash, dataframe_name='cash', make_index=True, index='cash_index')

    es.add_dataframe(dataframe=installments, dataframe_name='installments', make_index=True, index='installments_index')

    es.add_dataframe(dataframe=credit, dataframe_name='credit', make_index=True, index='credit_index')
    
    # Relationship between app and bureau
    r_app_bureau = ft.Relationship(es, 'app', 'SK_ID_CURR', 'bureau', 'SK_ID_CURR')

    # Relationship between bureau and bureau balance
    r_bureau_balance = ft.Relationship(es, 'bureau', 'SK_ID_BUREAU', 'bureau_balance', 'SK_ID_BUREAU')

    # Relationship between current app and previous apps
    r_app_previous = ft.Relationship(es, 'app', 'SK_ID_CURR', 'previous', 'SK_ID_CURR')

    # Relationships between previous apps and cash, installments, and credit
    r_previous_cash = ft.Relationship(es, 'previous', 'SK_ID_PREV', 'cash', 'SK_ID_PREV')
    r_previous_installments = ft.Relationship(es, 'previous', 'SK_ID_PREV', 'installments', 'SK_ID_PREV')
    r_previous_credit = ft.Relationship(es, 'previous', 'SK_ID_PREV', 'credit', 'SK_ID_PREV')

    # Add in the defined relationships (mutates es in-place)
    es.add_relationships([r_app_bureau, r_bureau_balance, r_app_previous,
                        r_previous_cash, r_previous_installments, r_previous_credit])
    # Print out the EntitySet
    print(es)

    return es

def dfs(es, path):
    # Default primitives from featuretools
    default_agg_primitives =  ['sum', 'count', 'min', 'max', 'mean', 'mode']
    default_trans_primitives =  ["day", "year", "month", "weekday", "haversine", "numwords", "characters"]

    # DFS for application features using a max depth of 2
    feature_matrix, feature_names = ft.dfs(entityset = es, target_dataframe_name = 'app',
                        # trans_primitives = default_trans_primitives,
                        agg_primitives=default_agg_primitives, 
                        max_depth = 2, features_only=False, verbose = True)
                        
    # Reset the index to make SK_ID_CURR a column again                                      
    feature_matrix = feature_matrix.reset_index()

    print('Saving features...')
    feature_matrix.to_csv(path/'feature_matrix.csv', index = False)

    return feature_matrix, feature_names

def main():
    # print('Reading in data...')
    # app_train, app_test, bureau, bureau_balance, cash, credit, previous, installments = read_data()
    app_train, app_test, bureau, bureau_balance, cash, credit, previous, installments = read_data_subset()
    print('DONE')
    
    # Add identifying column
    app_train['set'] = 'train'
    app_test['set'] = 'test'
    app_test["TARGET"] = np.nan

    # Append the dataframes
    app = pd.concat([app_train, app_test], ignore_index = True)

    print('Creating EntitySet...')
    es = create_entity_set(app, bureau, bureau_balance, cash, credit, previous, installments)
    print('DONE')

    print('Clearing up memory...')
    gc.enable()
    # Clear up memory
    del app, bureau, bureau_balance, cash, credit, installments, previous
    gc.collect()
    print('DONE')

    path = data_folder()

    print('Deep Feature Synthesis in Progress')
    eng_data, feature_names = dfs(es, path)
    print('DONE')

    for c in eng_data.select_dtypes(include=['category']).columns:
        eng_data[c] = eng_data[c].astype('object')

    # TODO: If the code runs without errors, just for debugging purposes use the saved feature matrix instead of recomputing it
    # path = data_folder()
    # eng_data = pd.read_csv(path/'feature_matrix.csv')

    # Separate out the train and test sets
    train_data = eng_data[eng_data['set'] == 'train']
    test_data = eng_data[eng_data['set'] == 'test']

    enc_train_data, categorical_features = obj_encoding(train_data)
    enc_test_data, _ = obj_encoding(test_data)

    # Align dataframes on the columns
    train_data, test_data = enc_train_data.align(enc_test_data, join = 'inner', axis = 1)
    test_data = test_data.drop(columns = ['TARGET'])

    print('Final Training Shape: ', train_data.shape)
    print('Final Testing Shape: ', test_data.shape)

    print('Clearing up memory...')
    gc.enable()
    # Clear up memory
    del es, eng_data, enc_train_data, enc_test_data
    gc.collect()
    print('DONE')

    # Compute correlations on training data only
    data_correlations(train_data, path)

    # Generate feature importances and null importances using train_data
    actual_imp_df, null_imp_df = get_features_distribution(train_data, categorical_features, path)

    features_scores(actual_imp_df, null_imp_df, path)
    corr_scores = correlation_scores(actual_imp_df, null_imp_df, path)

    percentage_scores = [0, 10, 20, 30 , 40, 50 ,60 , 70, 80 , 90, 95, 99]
    for threshold in percentage_scores:
        split_feats = [_f for _f, _score, _ in corr_scores if _score >= threshold]
        split_cat_feats = [_f for _f, _score, _ in corr_scores if (_score >= threshold) & (_f in categorical_features)]
        gain_feats = [_f for _f, _, _score in corr_scores if _score >= threshold]
        gain_cat_feats = [_f for _f, _, _score in corr_scores if (_score >= threshold) & (_f in categorical_features)]

        print('Results for threshold %3d' % threshold)
        split_results = score_feature_selection(df=train_data, train_features=split_feats, cat_cols=split_cat_feats, target=train_data['TARGET'])
        print('\t SPLIT : %.6f +/- %.6f' % (split_results[0], split_results[1]))
        gain_results = score_feature_selection(df=train_data, train_features=gain_feats, cat_cols=gain_cat_feats, target=train_data['TARGET'])
        print('\t GAIN  : %.6f +/- %.6f' % (gain_results[0], gain_results[1]))


if __name__ == "__main__":
    main()