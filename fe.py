import pandas as pd
import numpy as np

import gc

# featuretools for automated feature engineering
import featuretools as ft
from featuretools.primitives import AggregationPrimitive
from woodwork.column_schema import ColumnSchema
from woodwork.logical_types import Categorical, Double
from woodwork.table_accessor import PandasTableAccessor
from woodwork.column_accessor import PandasColumnAccessor


def _patch_woodwork_accessor_cache():
    """pandas 3+ creates a new accessor on each .ww access; woodwork stores state on the accessor."""
    def _cached_accessor(accessor_cls, cache_key):
        class _CachedAccessor:
            def __get__(self, obj, cls):
                if obj is None:
                    return self
                if cache_key not in obj.__dict__:
                    obj.__dict__[cache_key] = accessor_cls(obj)
                return obj.__dict__[cache_key]

        return _CachedAccessor()

    pd.DataFrame.ww = _cached_accessor(PandasTableAccessor, '_woodwork_df_accessor_cache')
    pd.Series.ww = _cached_accessor(PandasColumnAccessor, '_woodwork_series_accessor_cache')


_patch_woodwork_accessor_cache()
from datetime import datetime, timedelta

from collections import Counter

import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')

from data_analysis import obj_encoding, data_correlations, get_features_distribution, percentile_importances, log_ratio_importances, score_feature_selection, model

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
    app_train = pd.read_csv('data/application_train.csv').sort_values('SK_ID_CURR').reset_index(drop = True).loc[:1000, :]
    app_test = pd.read_csv('data/application_test.csv').sort_values('SK_ID_CURR').reset_index(drop = True).loc[:1000, :]
    bureau = pd.read_csv('data/bureau.csv').sort_values(['SK_ID_CURR', 'SK_ID_BUREAU']).reset_index(drop = True).loc[:1000, :]
    bureau_balance = pd.read_csv('data/bureau_balance.csv').sort_values('SK_ID_BUREAU').reset_index(drop = True).loc[:1000, :]
    cash = pd.read_csv('data/POS_CASH_balance.csv').sort_values(['SK_ID_CURR', 'SK_ID_PREV']).reset_index(drop = True).loc[:1000, :]
    credit = pd.read_csv('data/credit_card_balance.csv').sort_values(['SK_ID_CURR', 'SK_ID_PREV']).reset_index(drop = True).loc[:1000, :]
    previous = pd.read_csv('data/previous_application.csv').sort_values(['SK_ID_CURR', 'SK_ID_PREV']).reset_index(drop = True).loc[:1000, :]
    installments = pd.read_csv('data/installments_payments.csv').sort_values(['SK_ID_CURR', 'SK_ID_PREV']).reset_index(drop = True).loc[:1000, :]
    
    return app_train, app_test, bureau, bureau_balance, cash, credit, previous, installments

def change_var_types(df):
    types = {}

    # Iterate through the columns and record the Boolean columns
    for col in df:
        # If column is a number with only two values, encode it as a Boolean
        if (df[col].dtype != 'object') and (len(df[col].unique()) <= 2):
            types[col] = 'Boolean'

    # print(f'Number of boolean variables: {len(types)}')

    return df

def create_entity_set(app_train, app_test, bureau, bureau_balance, cash, credit, previous, installments):
    es = ft.EntitySet(id = 'clients')

    # Entities with a unique index
    es.add_dataframe(dataframe=app_train, dataframe_name='app_train', index='SK_ID_CURR')

    es.add_dataframe(dataframe=app_test, dataframe_name='app_test', index='SK_ID_CURR')

    es.add_dataframe(dataframe=bureau, dataframe_name='bureau', index='SK_ID_BUREAU')

    es.add_dataframe(dataframe=previous, dataframe_name='previous', index='SK_ID_PREV')

    # Entities that do not have a unique index
    es.add_dataframe(dataframe=bureau_balance, dataframe_name='bureau_balance', make_index=True, index='bureaubalance_index')

    es.add_dataframe(dataframe=cash, dataframe_name='cash', make_index=True, index='cash_index')

    es.add_dataframe(dataframe=installments, dataframe_name='installments', make_index=True, index='installments_index')

    es.add_dataframe(dataframe=credit, dataframe_name='credit', make_index=True, index='credit_index')
    
    # Relationship between app_train and bureau
    r_app_train_bureau = ft.Relationship(es, 'app_train', 'SK_ID_CURR', 'bureau', 'SK_ID_CURR')

    # Relationship between app_test and bureau
    r_app_test_bureau = ft.Relationship(es, 'app_test', 'SK_ID_CURR', 'bureau', 'SK_ID_CURR')
    
    # Relationship between bureau and bureau balance
    r_bureau_balance = ft.Relationship(es, 'bureau', 'SK_ID_BUREAU', 'bureau_balance', 'SK_ID_BUREAU')

    # Relationship between current app and previous apps
    r_app_train_previous = ft.Relationship(es, 'app_train', 'SK_ID_CURR', 'previous', 'SK_ID_CURR')

    # Test Relationship between current app and previous apps
    r_app_test_previous = ft.Relationship(es, 'app_test', 'SK_ID_CURR', 'previous', 'SK_ID_CURR')

    # Relationships between previous apps and cash, installments, and credit
    r_previous_cash = ft.Relationship(es, 'previous', 'SK_ID_PREV', 'cash', 'SK_ID_PREV')
    r_previous_installments = ft.Relationship(es, 'previous', 'SK_ID_PREV', 'installments', 'SK_ID_PREV')
    r_previous_credit = ft.Relationship(es, 'previous', 'SK_ID_PREV', 'credit', 'SK_ID_PREV')

    # Add in the defined relationships (mutates es in-place)
    es.add_relationships([r_app_train_bureau, r_app_test_bureau, r_bureau_balance, r_app_train_previous, 
                        r_app_test_previous, r_previous_cash, r_previous_installments, r_previous_credit])
    
    # Print out the EntitySet
    print(es)

    return es

def normalized_mode_count(x):
    """
    Return the fraction of total observations that 
    are the most common observation. For example, 
    in an array of ['A', 'A', 'A', 'B', 'B'], the 
    function will return 0.6."""
    
    if x.mode().shape[0] == 0:
        return np.nan
            
    # Count occurence of each value
    counts = dict(Counter(x.values))
    # Find the mode
    mode = x.mode().iloc[0]
    # Divide the occurences of mode by the total occurrences
    return counts[mode] / np.sum(list(counts.values()))

# Function from https://codereview.stackexchange.com/a/15095
def longest_repetition(x):
    """
    Returns the item with most consecutive occurrences in `x`. 
    If there are multiple items with the same number of conseqcutive occurrences,
    it will return the first one. If `x` is empty, returns None. 
    """
    
    x = x.dropna()
    
    if x.shape[0] < 1:
        return None
    
    # Set the longest element
    longest_element = current_element = None
    longest_repeats = current_repeats = 0
    
    # Iterate through the iterable
    for element in x:
        if current_element == element:
            current_repeats += 1
        else:
            current_element = element
            current_repeats = 1
        if current_repeats > longest_repeats:
            longest_repeats = current_repeats
            longest_element = current_element
            
    return longest_element

# Define custom aggregation primitive: NormalizedModeCount
class NormalizedModeCount(AggregationPrimitive):
    """Returns the fraction of total observations that are the most common observation."""
    name = 'normalized_mode_count'
    input_types = [ColumnSchema(logical_type=Categorical)]
    return_type = ColumnSchema(logical_type=Double)
    
    def get_function(self):
        return normalized_mode_count
    
# Define custom aggregation primitive: LongestSeq
class LongestSeq(AggregationPrimitive):
    """Returns the item with most consecutive occurrences."""
    name = 'longest_seq'
    input_types = [ColumnSchema(logical_type=Categorical)]
    return_type = ColumnSchema(logical_type=Categorical)
    
    def get_function(self):
        return longest_repetition

def dfs(es, path):
    # Reduced primitives to manage memory - only the most essential ones
    # Removed: mode, trend, LongestSeq, NormalizedModeCount (memory-intensive)
    selected_agg_primitives =  ['count', 'mean', 'sum', 'min', 'max']
    selected_trans_primitives =  []  # Empty to reduce memory usage

    # Seed feature representing Late Payment - Late payment means the payment date is greater (closer to 0) than the due date
    installments_df = es['installments']
    due_date = installments_df.ww['DAYS_INSTALMENT']
    paid_date = installments_df.ww['DAYS_ENTRY_PAYMENT']
    late_payment = ft.Feature(due_date) < ft.Feature(paid_date)
    late_payment = late_payment.rename("late_payment")

    # Seed feature representing whether the loan is past due
    bureau_balance_df = es['bureau_balance']
    status = bureau_balance_df.ww['STATUS']
    past_due = ft.Feature(status).isin(['1', '2', '3', '4', '5'])
    past_due = past_due.rename("past_due")
    
    # DFS for application train features using a max depth of 1
    # Removed where_primitives to significantly reduce feature explosion
    train_feature_matrix, train_feature_names = ft.dfs(entityset = es, target_dataframe_name = 'app_train',
                        trans_primitives = [],
                        agg_primitives=selected_agg_primitives, 
                        where_primitives = [],
                        seed_features = [late_payment, past_due],
                        max_depth = 1, features_only=False, verbose = True)

    # DFS for application test features using a max depth of 1
    test_feature_matrix, test_feature_names = ft.dfs(entityset = es, target_dataframe_name = 'app_test',
                        trans_primitives = [],
                        agg_primitives=selected_agg_primitives, 
                        where_primitives = [],
                        seed_features = [late_payment, past_due],
                        max_depth = 1, features_only=False, verbose = True)
                        
    # Reset the index to make SK_ID_CURR a column again                                      
    train_feature_matrix = train_feature_matrix.reset_index()
    test_feature_matrix = test_feature_matrix.reset_index()

    print('Saving features...')
    train_feature_matrix.to_csv(path/'train_feature_matrix.csv', index = False)
    test_feature_matrix.to_csv(path/'test_feature_matrix.csv', index = False)

    return train_feature_matrix, train_feature_names, test_feature_matrix, test_feature_names

def feature_selection(train_data_enc, test_data_enc, percentile_scores, ratio_scores, path):
    # Set thresholds
    IMPORTANCE_THRESHOLD = 0.0
    CORRELATOION_THRESHOLD = 0.80

    # Drop features with negative importance gain
    low_importance_feats = ratio_scores[ratio_scores['gain_score'] < IMPORTANCE_THRESHOLD]['feature'].tolist()
    # Filter to only include features that exist in train_data_enc
    low_importance_feats = [f for f in low_importance_feats if f in train_data_enc.columns]
    print(f'Found {len(low_importance_feats)} features with negative importance gain')
    train_data_enc.drop(columns=low_importance_feats, inplace=True)
    test_data_enc.drop(columns=[f for f in low_importance_feats if f in test_data_enc.columns], inplace=True)
    # Drop from percentile_scores (indexed by feature name)
    percentile_scores.drop(index=[f for f in low_importance_feats if f in percentile_scores.index], inplace=True)

    # Drop constant features with low importance
    constant_feats = []
    for col in train_data_enc.columns:
        if col != 'TARGET' and train_data_enc[col].nunique() <= 1:
            constant_feats.append(col)

    print(f'Found {len(constant_feats)} constant features')

    # Drop constant features from both train and test sets
    train_data_enc.drop(columns=constant_feats, inplace=True)
    test_data_enc.drop(columns=constant_feats, inplace=True)
    # Drop from percentile_scores (indexed by feature name)
    percentile_scores.drop(index=[f for f in constant_feats if f in percentile_scores.index], inplace=True)
    
    # Drop features with more than 80% missing values
    high_missing_feats = []
    for col in train_data_enc.columns:
        if col != 'TARGET' and train_data_enc[col].isnull().sum() / train_data_enc.shape[0] > 0.8:
            high_missing_feats.append(col)
    print(f'Found {len(high_missing_feats)} features with more than 80% missing values')

    # Filter to only include features that exist
    high_missing_feats = [f for f in high_missing_feats if f in train_data_enc.columns]
    train_data_enc.drop(columns=high_missing_feats, inplace=True)
    test_data_enc.drop(columns=[f for f in high_missing_feats if f in test_data_enc.columns], inplace=True)
    # Drop from percentile_scores (indexed by feature name)
    percentile_scores.drop(index=[f for f in high_missing_feats if f in percentile_scores.index], inplace=True)

    # Compute correlations and get correlation pairs
    high_pairs, low_pairs, _ = data_correlations(train_data_enc, path, CORRELATOION_THRESHOLD)
    
    # Drop highly correlated features (keep the one with higher gain_score)
    # Combine high positive and high negative correlation pairs
    corr_pairs = pd.concat([high_pairs, low_pairs], ignore_index=True)
    
    # Build a lookup dict for gain_scores from ratio_scores DataFrame
    gain_score_dict = dict(zip(ratio_scores['feature'], ratio_scores['gain_score']))
    
    high_corr_feats = set()
    for _, row in corr_pairs.iterrows():
        feat1, feat2 = row['feature_1'], row['feature_2']
        
        # Skip if either feature is TARGET or already marked for drop
        if feat1 == 'TARGET' or feat2 == 'TARGET':
            continue
        if feat1 in high_corr_feats or feat2 in high_corr_feats:
            continue
            
        # Get gain scores (default to -inf if not found)
        gain1 = gain_score_dict.get(feat1, float('-inf'))
        gain2 = gain_score_dict.get(feat2, float('-inf'))
        
        # Drop the feature with lower gain_score
        if gain1 < gain2:
            high_corr_feats.add(feat1)
        else:
            high_corr_feats.add(feat2)
    
    high_corr_feats = list(high_corr_feats)
    print(f'Found {len(high_corr_feats)} highly correlated features to drop (|corr| > 0.6)')
    
    # Filter to only include features that exist
    high_corr_feats = [f for f in high_corr_feats if f in train_data_enc.columns]
    train_data_enc.drop(columns=high_corr_feats, inplace=True)
    test_data_enc.drop(columns=[f for f in high_corr_feats if f in test_data_enc.columns], inplace=True)
    # Drop from percentile_scores (indexed by feature name)
    percentile_scores.drop(index=[f for f in high_corr_feats if f in percentile_scores.index], inplace=True)

    return train_data_enc, test_data_enc, percentile_scores


def main():
    # ==================================================== FEATURE ENGINEERING ====================================================
    print('Reading in data...')
    app_train, app_test, bureau, bureau_balance, cash, credit, previous, installments = read_data()
    # app_train, app_test, bureau, bureau_balance, cash, credit, previous, installments = read_data_subset()
    print('DONE')
    
    # Add identifying column
    # app_train['set'] = 'train'
    # app_test['set'] = 'test'
    # app_test["TARGET"] = np.nan

    # Append the dataframes
    # app = pd.concat([app_train, app_test], ignore_index = True)
    
    print('Changing variable types...')
    for df in [app_train, app_test, bureau, bureau_balance, cash, credit, previous, installments]:
        df = change_var_types(df)

    print('Creating EntitySet...')
    es = create_entity_set(app_train, app_test, bureau, bureau_balance, cash, credit, previous, installments)
    print('DONE')

    print('Clearing up memory...')
    gc.enable()
    # Clear up memory
    del app_train, app_test, bureau, bureau_balance, cash, credit, installments, previous
    gc.collect()
    print('DONE')   
    
    path = data_folder()

    print('Deep Feature Synthesis in Progress')
    train_data_eng, train_feature_names, test_data_eng, test_feature_names = dfs(es, path)
    print('DONE')
    
    # ==================================================== FEATURE SELECTION ====================================================
    # TODO: If the code runs without errors, just for debugging purposes use the saved feature matrix instead of recomputing it
    # path = data_folder()
    # train_data_eng = pd.read_csv(path/'train_feature_matrix.csv')
    # test_data_eng = pd.read_csv(path/'test_feature_matrix.csv')

    for c in train_data_eng.select_dtypes(include=['category']).columns:
        train_data_eng[c] = train_data_eng[c].astype('object')

    for c in test_data_eng.select_dtypes(include=['category']).columns:
        test_data_eng[c] = test_data_eng[c].astype('object')

    train_data_enc, train_categorical_features = obj_encoding(train_data_eng)
    test_data_enc, test_categorical_features = obj_encoding(test_data_eng)

    # Align dataframes on the columns (excluding TARGET which only exists in train)
    if(train_data_enc.shape[1]-1 != test_data_enc.shape[1]):
        # Save TARGET before alignment
        target = train_data_enc['TARGET'].copy()
        train_data_enc = train_data_enc.drop(columns=['TARGET'])
        
        # Align on common columns
        train_data_enc, test_data_enc = train_data_enc.align(test_data_enc, join = 'inner', axis = 1)
        
        # Restore TARGET to train data
        train_data_enc['TARGET'] = target

    print('Training Shape after feature engineering and encoding: ', train_data_enc.shape)
    print('Testing Shape after feature engineering and encoding: ', test_data_enc.shape)

    # Align categorical features lists
    train_categorical_features = [f for f in train_categorical_features if f in train_data_enc.columns]
    test_categorical_features = [f for f in test_categorical_features if f in test_data_enc.columns]
    
    # Use common categorical features
    categorical_features = list(set(train_categorical_features) & set(test_categorical_features))
    # print(f'Number of aligned categorical features: {len(categorical_features)}')

    print('Clearing up memory...')
    gc.enable()
    # Clear up memory
    # del es, train_data_eng, test_data_eng, train_categorical_features, test_categorical_features
    del train_data_eng, test_data_eng, train_categorical_features, test_categorical_features
    gc.collect()
    print('DONE')

    # Generate feature importances and null importances using train_data
    actual_imp_df, null_imp_df = get_features_distribution(train_data_enc, categorical_features, path)

    ratio_scores = log_ratio_importances(actual_imp_df, null_imp_df, path)
    percentile_scores = percentile_importances(actual_imp_df, null_imp_df, path)
    
    # Convert percentile_scores from list of tuples to DataFrame if needed
    if isinstance(percentile_scores, list):
        percentile_scores = pd.DataFrame(percentile_scores, columns=['feature', 'split_score', 'gain_score'])
        percentile_scores = percentile_scores.set_index('feature')
    
    train_data, test_data, percentage_importances = feature_selection(train_data_enc, test_data_enc, percentile_scores, ratio_scores, path)

    print(f'Final training shape after feature selection: {train_data.shape}')
    print(f'Final testing shape after feature selection: {test_data.shape}')

    percentage_scores = [0, 10, 20, 30 , 40, 50 ,60 , 70, 80 , 90, 95, 99]
    for threshold in percentage_scores:
        # Iterate over DataFrame rows instead of list of tuples
        gain_feats = [_f for _f, row in percentage_importances.iterrows() if row['gain_score'] >= threshold]
        gain_cat_feats = [_f for _f in gain_feats if _f in categorical_features]

        print('Results for threshold %3d' % threshold)
        # split_results = score_feature_selection(df=train_data_enc, train_features=split_feats, cat_cols=split_cat_feats, target=train_data_enc['TARGET'])
        # print('\t SPLIT : %.6f +/- %.6f' % (split_results[0], split_results[1]))
        gain_results = score_feature_selection(df=train_data, train_features=gain_feats, cat_cols=gain_cat_feats, target=train_data['TARGET'], output_folder=path, threshold=threshold)
        print('\t GAIN  : %.6f +/- %.6f' % (gain_results[0], gain_results[1]))

    # ========================================================= MODEL =========================================================
    # model(train_data, test_data, categorical_features, path, n_folds=5)


if __name__ == "__main__":
    main()