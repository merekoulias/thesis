# numpy and pandas for data manipulation
import numpy as np
import pandas as pd

# sklearn preprocessing for dealing with categorical variables
from sklearn.preprocessing import LabelEncoder

# File system manangement
import os

# Suppress warnings 
import warnings
warnings.filterwarnings('ignore')

# matplotlib and seaborn for plotting
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

# plotly offline iplot for interactive plots
from plotly.offline import iplot

# Matplotlib ticker for better axis formatting
import matplotlib.ticker as mtick

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
import time
from lightgbm import LGBMClassifier
import lightgbm as lgb
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import psutil

# import gc
# gc.enable()

# Globals used by worker processes (set by initializer)
_WORKER_DF = None
_WORKER_CAT_COLS = None

def _init_worker(df, cat_cols):
    """Initializer for worker processes. Stores a reference to the dataframe and categorical
    columns in module-level variables to avoid re-pickling them for each task."""
    global _WORKER_DF, _WORKER_CAT_COLS
    _WORKER_DF = df
    _WORKER_CAT_COLS = cat_cols

def _worker_run(seed):
    """Worker function executed in each process. Uses the global _WORKER_DF and _WORKER_CAT_COLS
    set by _init_worker. Returns the feature importances DataFrame for a shuffled target.
    The seed is used for reproducibility."""
    try:
        imp = features_importance(_WORKER_DF, shuffle=True, cat_cols=_WORKER_CAT_COLS, seed=seed)
        return imp
    except Exception as e:
        # Return the exception to be handled by the parent
        return e

def load_and_preview_data():
    # Training data
    try:
        print(f"Loading data ('application_train.csv') from data folder...")
        train_data = pd.read_csv('data/application_train.csv')
        print(f"Data loaded successfully")
        print(80*'=')
        print('Training data shape: ', train_data.shape)
        print(80*'=')
        print("First 5 rows of training data:")
        head_df = train_data.head()
        print(head_df)
        print("Saving file's head to exploratory_data_analysis/application_train_head.csv")
        head_df.to_csv('exploratory_data_analysis/application_train_head.csv', index=False)
        print(80*'=')
    
    except FileNotFoundError:
        print(f"application_train.csv not found")

    # Testing data features
    try:
        print(f"Loading test data ('application_test.csv') from data folder...")
        test_data = pd.read_csv('data/application_test.csv')
        print(f"Data loaded successfully")
        print(80*'=')
        print('Testing data shape: ', test_data.shape)
        print("First 5 rows of test data:")
        print(test_data.head())
        print(80*'=')
    
    except FileNotFoundError:
        print(f"application_test.csv not found")

    # Column Types
    print("====================================COLUMN TYPES====================================")
    print("Number of each column type in training data:")
    print(train_data.dtypes.value_counts())
    print(80*'=')
    print("Number of unique entries (classes) in each of the categorial (object) columns:")
    print(train_data.select_dtypes('object').apply(pd.Series.nunique, axis = 0))

    # Target distribution
    print("================EXAMINING THE DISTRIBUTION OF THE TARGET COLUMN================")
    print(train_data['TARGET'].value_counts())
    print(80*'=')

    # Target distribution plot
    print("Plotting TARGET histogram to see class distribution")
    target_hist = train_data['TARGET'].astype(int).plot.hist()
    target_hist.set_xlabel("TARGET")
    target_hist.set_ylabel("Number of Applicants")
    print("Saving histogram to exploratory_data_analysis/target_histogram.png")
    plt.savefig('exploratory_data_analysis/target_histogram.png', bbox_inches='tight', dpi=300)
    plt.close()
    print(80*'=')

    # Gender distribution
    print("================EXAMINING THE DISTRIBUTION OF THE GENDER COLUMN================")
    print(train_data['CODE_GENDER'].value_counts())
    print(80*'=')

    # NAME_EDUCATION_TYPE donut chart
    print("================EXAMINING THE DISTRIBUTION OF APPLICANT'S EDUCATION LEVEL (NAME_EDUCATION_TYPE COLUMN)================")
    name_education_type = train_data["NAME_EDUCATION_TYPE"].value_counts()
    plt.figure(figsize=(6,6))
    wedges, texts, autotexts = plt.pie(name_education_type.values, 
            labels=name_education_type.index, 
            startangle=90, 
            counterclock=False,
            wedgeprops=dict(width=0.4, edgecolor='w'),
            autopct='%1.1f%%',           
            pctdistance=0.75)
    plt.title('Types of Education')
    plt.legend(wedges, name_education_type.index, title='Education Level',
           loc='center left', bbox_to_anchor=(1.0, 0.5))
    plt.close()
    plt.savefig('exploratory_data_analysis/education_level_donut.png', bbox_inches='tight', dpi=300)
    print(f"Saved static donut to 'exploratory_data_analysis/education_level_donut.png'")
    print(80*'=')

    # NAME_EDUCATION_TYPE distribution plot grouped by TARGET
    print("================EXAMINING THE DISTRIBUTION OF APPLICANT'S EDUCATION LEVEL (NAME_EDUCATION_TYPE COLUMN) GROUPED BY TARGET================")
    plt.figure(figsize=(12,5))
    sns.countplot(data=train_data, x='NAME_EDUCATION_TYPE', hue='TARGET')
    plt.title("Distribution of Education Level by Target")
    plt.savefig('exploratory_data_analysis/education_level_by_target.png', bbox_inches='tight', dpi=300)
    plt.close()
    print(80*'=')

    # AMT_CREDIT distribution plot
    print("================EXAMINING THE DISTRIBUTION OF THE AMT_CREDIT COLUMN================")
    plt.figure(figsize=(12,5))
    plt.title("Distribution of AMT_CREDIT")
    sns.distplot(train_data["AMT_CREDIT"])
    plt.savefig('exploratory_data_analysis/amt_credit_distribution.png', bbox_inches='tight', dpi=300)
    plt.close()
    print(80*'=')

    # AMT_TOTAL_INCOME distribution plot
    print("================EXAMINING THE DISTRIBUTION OF THE AMT_TOTAL_INCOME COLUMN================")
    plt.figure(figsize=(12,5))
    plt.title("Distribution of AMT_INCOME_TOTAL")
    sns.distplot(train_data["AMT_INCOME_TOTAL"].dropna())
    plt.savefig('exploratory_data_analysis/amt_total_income_distribution.png', bbox_inches='tight', dpi=300)
    plt.close()
    print(80*'=')

    # AMT_GOODS_PRICE distribution plot
    print("================EXAMINING THE DISTRIBUTION OF THE AMT_GOODS_PRICE COLUMN================")
    plt.figure(figsize=(12,5))
    plt.title("Distribution of AMT_GOODS_PRICE")
    sns.distplot(train_data["AMT_GOODS_PRICE"].dropna())
    plt.savefig('exploratory_data_analysis/amt_goods_price_distribution.png', bbox_inches='tight', dpi=300)
    plt.close()
    print(80*'=')

    # NAME_CONTRACT_TYPE donut chart
    print("================EXAMINING THE DISTRIBUTION OF LOAN TYPE (NAME_CONTRACT_TYPE COLUMN)================")
    name_contract_type = train_data["NAME_CONTRACT_TYPE"].value_counts()
    plt.figure(figsize=(6,6))
    wedges, texts, autotexts = plt.pie(name_contract_type.values, 
            labels=name_contract_type.index, 
            startangle=90, 
            counterclock=False,
            wedgeprops=dict(width=0.4, edgecolor='w'),
            autopct='%1.1f%%',           
            pctdistance=0.75)
    plt.title('Types of loan')
    plt.legend(wedges, name_contract_type.index, title='Loan Type',
           loc='center left', bbox_to_anchor=(1.0, 0.5))
    plt.close()
    plt.savefig('exploratory_data_analysis/loan_type_donut.png', bbox_inches='tight', dpi=300)
    print(f"Saved static donut to 'exploratory_data_analysis/loan_type_donut.png'")
    print(80*'=')

    # NAME_INCOME_TYPE donut chart
    print("================EXAMINING THE DISTRIBUTION OF APPLICANT'S INCOME SOURCE (NAME_INCOME_TYPE COLUMN)================")
    name_income_type = train_data["NAME_INCOME_TYPE"].value_counts()
    plt.figure(figsize=(6,6))
    wedges, texts, autotexts = plt.pie(name_income_type.values, 
            labels=name_income_type.index, 
            startangle=90, 
            counterclock=False,
            wedgeprops=dict(width=0.4, edgecolor='w'),
            autopct='%1.1f%%',           
            pctdistance=0.75)
    plt.title('Applicant\'s Income Source')
    plt.legend(wedges, name_income_type.index, title='Income Source',
           loc='center left', bbox_to_anchor=(1.0, 0.5))
    plt.close()
    plt.savefig('exploratory_data_analysis/income_source_donut.png', bbox_inches='tight', dpi=300)
    print(f"Saved static donut to 'exploratory_data_analysis/income_source_donut.png'")
    print(80*'=')
    
    # NAME_INCOME_TYPE distribution grouped by TARGET
    print("================EXAMINING THE DISTRIBUTION OF APPLICANT'S INCOME SOURCE (NAME_INCOME_TYPE COLUMN) GROUPED BY TARGET================")
    plt.figure(figsize=(12, 6))
    sns.countplot(data=train_data, x="NAME_INCOME_TYPE", hue="TARGET")
    plt.title("Distribution of Applicant's Income Source by Target")
    plt.xlabel("Income Source")
    plt.ylabel("Count")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('exploratory_data_analysis/income_source_by_target.png', bbox_inches='tight', dpi=300)
    plt.close()
    print(80*'=')

    # OCCUPATION_TYPE distribution plot
    print("================EXAMINING THE DISTRIBUTION OF APPLICANT'S OCCUPATION (OCCUPATION_TYPE COLUMN)================")
    # counts-only bar chart (dropna by default)
    occupation_type = train_data["OCCUPATION_TYPE"].value_counts()
    plt.figure(figsize=(10, 6))
    bars = plt.bar(occupation_type.index, occupation_type.values, color='green', edgecolor='k')
    plt.title("Occupation of Applicants who applied for loan")
    plt.xlabel("Occupation")
    plt.ylabel("Count")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    # annotate counts above bars
    for b in bars:
        h = b.get_height()
        plt.text(b.get_x() + b.get_width() / 2, h + max(1, occupation_type.values.max()*0.01), f'{int(h)}',
                ha='center', va='bottom', fontsize=9)

    plt.savefig('exploratory_data_analysis/occupation_type_bar_chart.png', dpi=300, bbox_inches='tight')
    print(f"Saved static bar chart to 'exploratory_data_analysis/occupation_type_bar_chart.png'")
    plt.close()

    # OCCUPATION_TYPE distribution plot grouped by TARGET
    print("================EXAMINING THE DISTRIBUTION OF APPLICANT'S OCCUPATION (OCCUPATION_TYPE COLUMN) GROUPED BY TARGET================")
    plt.figure(figsize=(12, 6))
    sns.countplot(data=train_data, x="OCCUPATION_TYPE", hue="TARGET")
    plt.title("Distribution of Applicant's Occupation by Target")
    plt.xlabel("Occupation")
    plt.ylabel("Count")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('exploratory_data_analysis/occupation_type_by_target.png', bbox_inches='tight', dpi=300)
    plt.close()
    print(80*'=')

    # ORGANIZATION_TYPE distribution plot
    print("================EXAMINING THE DISTRIBUTION OF THE ORGANIZATION THE APPLICANT'S WORKS (ORGANIZATION_TYPE COLUMN)================")
    organization_type = train_data["ORGANIZATION_TYPE"].value_counts(dropna=False)
    total = int(organization_type.sum())

    labels = organization_type.index.astype(str)
    counts = organization_type.values

    fig, ax = plt.subplots(figsize=(12, 10))
    y_pos = np.arange(len(labels))

    # horizontal bars (easier to read long labels)
    bars = ax.barh(y_pos, counts, color='green', edgecolor='k', height=0.5)

    # invert so largest is on top
    ax.invert_yaxis()

    # label formatting
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel('Count', fontsize=12)
    ax.set_title("Organization of Applicants who applied for loan", fontsize=14)

    # thousands separator on x-axis
    ax.xaxis.set_major_formatter(mtick.FuncFormatter(lambda x, p: format(int(x), ',')))

    # annotate counts and percentages just outside the bars
    max_count = counts.max() if len(counts) else 1
    pad = max_count * 0.01
    for i, v in enumerate(counts):
        pct = v / total * 100.0 if total else 0.0
        ax.text(v + pad, i, f'{v:,}  ({pct:.1f}%)', va='center', fontsize=8)

    plt.tight_layout()
    plt.savefig('exploratory_data_analysis/organization_type_bar_chart.png', dpi=300, bbox_inches='tight')
    print(f"Saved static bar chart to 'exploratory_data_analysis/organization_type_bar_chart.png'")
    plt.close()

    # ORGANIZATION_TYPE distribution plot grouped by TARGET
    print("================EXAMINING THE DISTRIBUTION OF THE ORGANIZATION THE APPLICANT'S WORKS (ORGANIZATION_TYPE COLUMN) GROUPED BY TARGET================")
    plt.figure(figsize=(12, 6))
    sns.countplot(data=train_data, x="ORGANIZATION_TYPE", hue="TARGET")
    plt.title("Distribution of Organization Type by Target")
    plt.xlabel("Organization Type")
    plt.ylabel("Count")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('exploratory_data_analysis/organization_type_by_target.png', bbox_inches='tight', dpi=300)
    plt.close()
    print(80*'=')

    return train_data, test_data

def missing_values(df):
        # Total missing values
        mis_val = df.isnull().sum()
        
        # Percentage of missing values
        mis_val_percent = 100 * df.isnull().sum() / len(df)
        
        # Make a table with the results
        mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
        
        # Rename the columns
        mis_val_table_ren_columns = mis_val_table.rename(
        columns = {0 : 'Missing Values', 1 : '% of Total Values'})
        
        # Sort the table by percentage of missing descending
        mis_val_table_ren_columns = mis_val_table_ren_columns[
            mis_val_table_ren_columns.iloc[:,1] != 0].sort_values(
        '% of Total Values', ascending=False).round(1)
        
        # Print some summary information
        print ("There are " + str(mis_val_table_ren_columns.shape[0]) + " columns that have missing values.")
        print(80*'=')

        # Save the missing values table to a CSV file
        print("Saving missing values table to 'exploratory_data_analysis/missing_values_table.csv'")
        mis_val_table_ren_columns.to_csv('exploratory_data_analysis/missing_values_table.csv')

        print("First 20 rows of missing values table:")
        print(mis_val_table_ren_columns.head(20))
        print(80*'=')
        return 

def obj_encoding(data):
    categorical_features = [f for f in data.columns if data[f].dtype == 'object']

    for f_ in categorical_features:
        data[f_], _ = pd.factorize(data[f_])
        # Set feature type as categorical
        data[f_] = data[f_].astype('category')

    return data, categorical_features

def get_features_distribution(train_data, categorical_features, seed=None):
    # ================================ ACTUAL IMPORTANCE DISTRIBUTION (without shuffling) ================================
    np.random.seed(123) # Seed the unexpected randomness of this world

    actual_imp_df = features_importance(train_data, shuffle=False, cat_cols=categorical_features)

    print("Saving actual feature importances to 'exploratory_data_analysis/actual_feature_importances.csv'")
    actual_imp_df.to_csv('exploratory_data_analysis/actual_feature_importances.csv', index=False)
    
    print("First 5 rows of actual (non-shuffled) feature importances:")
    print(actual_imp_df.head())
    
    # ================================ NULL IMPORTANCE DISTRIBUTION (without shuffling) ================================
    null_imp_df = pd.DataFrame()
    nb_runs = 80
    msg = ''

    start = time.time()

    # Number of workers defaults to number of CPUs - 1 (but at least 1)
    cpu_count = multiprocessing.cpu_count()
    # Estimate physical cores (not logical CPUs)
    try:
        physical_cores = psutil.cpu_count(logical=False)
        if physical_cores is None:
            physical_cores = cpu_count // 2  # Fallback: assume 2 threads per core
    except ImportError:
        physical_cores = cpu_count // 2  # Fallback if psutil not available

    # If you know threads_per_worker (e.g. LightGBM num_threads=1), use that
    threads_per_worker = 1
    nb_workers = min(cpu_count, physical_cores)
    print(f"Using {nb_workers} workers for parallel execution (cpu_count={cpu_count}, physical_cores={physical_cores}, threads_per_worker={threads_per_worker})")
    print("Using %d workers for parallel execution" % nb_workers)

    try:
        # Use ProcessPoolExecutor with initializer to avoid sending the entire dataframe for every task
        seeds = [None] * nb_runs  # you can replace with explicit seeds like list(range(nb_runs)) if desired
        with ProcessPoolExecutor(max_workers=nb_workers, initializer=_init_worker, initargs=(train_data, categorical_features)) as exe:
            futures = {exe.submit(_worker_run, seeds[i]): i for i in range(nb_runs)}

            for fut in as_completed(futures):
                i = futures[fut]
                res = fut.result()
                if isinstance(res, Exception):
                    # Fall back to serial execution if a worker raised
                    raise res

                imp_df = res
                imp_df['run'] = i + 1
                null_imp_df = pd.concat([null_imp_df, imp_df], axis=0)

                # Progress message
                for l in range(len(msg)):
                    print('\b', end='', flush=True)
                spent = (time.time() - start) / 60
                msg = 'Done with %4d of %4d (Spent %5.1f min)' % (len(null_imp_df['run'].unique()), nb_runs, spent)
                print(msg, end='', flush=True)

    except Exception as e:
        # If parallel execution fails for any reason, fall back to the original serial loop
        print('\nParallel execution failed (%s). Falling back to serial execution.' % repr(e))
        null_imp_df = pd.DataFrame()
        start = time.time()
        for i in range(nb_runs):
            imp_df = features_importance(train_data, shuffle=True, cat_cols=categorical_features, seed=None)
            imp_df['run'] = i + 1
            null_imp_df = pd.concat([null_imp_df, imp_df], axis=0)

            # Erase previous message
            for l in range(len(msg)):
                print('\b', end='', flush=True)

            # Display current run and time used
            spent = (time.time() - start) / 60
            msg = 'Done with %4d of %4d (Spent %5.1f min)' % (i + 1, nb_runs, spent)
            print(msg, end='', flush=True)
    
    print("\n")
    print("Saving null feature importances distribution to 'exploratory_data_analysis/null_importances_distribution.csv'")
    imp_df.to_csv('exploratory_data_analysis/null_importances_distribution.csv', index=False)

    print("First 5 rows of shuffled feature importances:")
    print(imp_df.head())

    return actual_imp_df, null_imp_df

def features_importance(data, shuffle, cat_cols, seed=None):
    """
    trains a LightGBM model (configured as Random Forest) on the given dataset and returns a DataFrame containing:
        -Each feature's gain-based importance
        -Each feature's split importance
        -A single AUC score on the data (not a CV score). Depending on the shuffle flag, 
        it either uses the real target (observed importance) or a randomly permuted target. 
        Repeating it many times with shuffle=True builds the null distributions.
    """

    train_features = [f for f in data if f not in ['TARGET', 'SK_ID_CURR']]

    y = data['TARGET'].copy()
    if shuffle:
        y = data['TARGET'].copy().sample(frac=1.0)
    
    # Pass categorical features to the Dataset object
    dtrain = lgb.Dataset(data[train_features], y, free_raw_data=False, categorical_feature=cat_cols)
    lgb_params = {
        'objective': 'binary',
        'verbosity': -1,
        'boosting_type': 'rf', # Fit LightGBM in RF mode beacause it's quicker than sklearn RandomForest
        'subsample': 0.623,
        'colsample_bytree': 0.7,
        'num_leaves': 127,
        'max_depth': 8,
        'seed': seed,
        'bagging_freq': 1,
        'n_jobs': 1
    }
    
    # Fit the model
    clf = lgb.train(params=lgb_params, train_set=dtrain, num_boost_round=200)

    # Get feature importances
    imp_df = pd.DataFrame()
    imp_df["feature"] = list(train_features)
    imp_df["importance_gain"] = clf.feature_importance(importance_type='gain')
    imp_df["importance_split"] = clf.feature_importance(importance_type='split')
    imp_df['trn_score'] = roc_auc_score(y, clf.predict(data[train_features]))
    
    return imp_df

def display_distributions(actual_imp_df, null_imp_df, feature):
    plt.figure(figsize=(13, 6))
    gs = gridspec.GridSpec(1, 2)
    # Plot Split importances
    ax = plt.subplot(gs[0, 0])
    a = ax.hist(null_imp_df.loc[null_imp_df['feature'] == feature, 'importance_split'].values, label='Null importances')
    ax.vlines(x=actual_imp_df.loc[actual_imp_df['feature'] == feature, 'importance_split'].mean(), 
               ymin=0, ymax=np.max(a[0]), color='r',linewidth=10, label='Real Target')
    ax.legend()
    ax.set_title('Split Importance of %s' % feature.upper(), fontweight='bold')
    plt.xlabel('Null Importance (split) Distribution for %s ' % feature.upper())
    # Plot Gain importances
    ax = plt.subplot(gs[0, 1])
    a = ax.hist(null_imp_df.loc[null_imp_df['feature'] == feature, 'importance_gain'].values, label='Null importances')
    ax.vlines(x=actual_imp_df.loc[actual_imp_df['feature'] == feature, 'importance_gain'].mean(), 
               ymin=0, ymax=np.max(a[0]), color='r',linewidth=10, label='Real Target')
    ax.legend()
    ax.set_title('Gain Importance of %s' % feature.upper(), fontweight='bold')
    plt.xlabel('Null Importance (gain) Distribution for %s ' % feature.upper())
    
    return 

def features_scores(actual_imp_df, null_imp_df):
    feature_scores = []

    for _f in actual_imp_df['feature'].unique():
        f_null_imps_gain = null_imp_df.loc[null_imp_df['feature'] == _f, 'importance_gain'].values
        f_act_imps_gain = actual_imp_df.loc[actual_imp_df['feature'] == _f, 'importance_gain'].mean()

        gain_score = np.log(1e-10 + f_act_imps_gain / (1 + np.percentile(f_null_imps_gain, 75)))  # Avoid didvide by zero

        f_null_imps_split = null_imp_df.loc[null_imp_df['feature'] == _f, 'importance_split'].values
        f_act_imps_split = actual_imp_df.loc[actual_imp_df['feature'] == _f, 'importance_split'].mean()

        split_score = np.log(1e-10 + f_act_imps_split / (1 + np.percentile(f_null_imps_split, 75)))  # Avoid didvide by zero

        feature_scores.append((_f, split_score, gain_score))

    scores_df = pd.DataFrame(feature_scores, columns=['feature', 'split_score', 'gain_score'])
    print("Saving feature scores to 'exploratory_data_analysis/feature_scores.csv'")
    scores_df.to_csv('exploratory_data_analysis/feature_scores.csv', index=False)

    plt.figure(figsize=(16, 16))
    gs = gridspec.GridSpec(1, 2)
    # Plot Split importances
    ax = plt.subplot(gs[0, 0])
    sns.barplot(x='split_score', y='feature', data=scores_df.sort_values('split_score', ascending=False).iloc[0:70], ax=ax)
    ax.set_title('Feature scores wrt split importances', fontweight='bold', fontsize=14)
    # Plot Gain importances
    ax = plt.subplot(gs[0, 1])
    sns.barplot(x='gain_score', y='feature', data=scores_df.sort_values('gain_score', ascending=False).iloc[0:70], ax=ax)
    ax.set_title('Feature scores wrt gain importances', fontweight='bold', fontsize=14)
    plt.tight_layout()
    print("Saving feature scores plot to 'exploratory_data_analysis/feature_scores.png'")
    plt.savefig('exploratory_data_analysis/feature_scores.png', bbox_inches='tight', dpi=300)
    plt.close()

    return scores_df

def correlation_scores(actual_imp_df, null_imp_df):
    correlation_scores = []

    for _f in actual_imp_df['feature'].unique():
        f_null_imps = null_imp_df.loc[null_imp_df['feature'] == _f, 'importance_gain'].values
        f_act_imps = actual_imp_df.loc[actual_imp_df['feature'] == _f, 'importance_gain'].values
        
        gain_score = 100 * (f_null_imps < np.percentile(f_act_imps, 25)).sum() / f_null_imps.size

        f_null_imps = null_imp_df.loc[null_imp_df['feature'] == _f, 'importance_split'].values
        f_act_imps = actual_imp_df.loc[actual_imp_df['feature'] == _f, 'importance_split'].values

        split_score = 100 * (f_null_imps < np.percentile(f_act_imps, 25)).sum() / f_null_imps.size

        correlation_scores.append((_f, split_score, gain_score))

    corr_scores_df = pd.DataFrame(correlation_scores, columns=['feature', 'split_score', 'gain_score'])

    fig = plt.figure(figsize=(16, 16))
    gs = gridspec.GridSpec(1, 2)
    # Plot Split importances
    ax = plt.subplot(gs[0, 0])
    sns.barplot(x='split_score', y='feature', data=corr_scores_df.sort_values('split_score', ascending=False).iloc[0:70], ax=ax)
    ax.set_title('Feature scores wrt split importances', fontweight='bold', fontsize=14)
    # Plot Gain importances
    ax = plt.subplot(gs[0, 1])
    sns.barplot(x='gain_score', y='feature', data=corr_scores_df.sort_values('gain_score', ascending=False).iloc[0:70], ax=ax)
    ax.set_title('Feature scores wrt gain importances', fontweight='bold', fontsize=14)
    plt.tight_layout()
    plt.suptitle("Features' split and gain scores", fontweight='bold', fontsize=16)
    fig.subplots_adjust(top=0.93)
    print("Saving feature scores plot to 'exploratory_data_analysis/correlation_scores.png'")
    plt.savefig('exploratory_data_analysis/correlation_scores.png', bbox_inches='tight', dpi=300)
    plt.close()

    return correlation_scores

def score_feature_selection(df, train_features, cat_cols, target):
    dtrain = lgb.Dataset(df[train_features], target, free_raw_data=False, categorical_feature=cat_cols)

    lgb_params = {
        'objective': 'binary',
        'verbosity': -1,
        'boosting_type': 'gbdt',
        'learning_rate': .1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'num_leaves': 31,
        'max_depth': -1,
        'seed': 13,
        'n_jobs': 4,
        'min_split_gain': .00001,
        'reg_alpha': .00001,
        'reg_lambda': .00001,
        'metric': 'auc'
    }

    # Use callbacks for early stopping
    callbacks = [lgb.early_stopping(stopping_rounds=50, verbose=False)]

    hist = lgb.cv(
        params=lgb_params,
        train_set=dtrain,
        num_boost_round=2000,
        nfold=5,
        stratified=True,
        shuffle=True,
        seed=17,
        callbacks=callbacks
    )
    
    return hist['valid auc-mean'][-1], hist['valid auc-stdv'][-1]
    

def objects_encoding(train_data, test_data):
    """
    for any categorical variable (dtype == object) with 2 unique categories, 
    we will use label encoding, 
    and for any categorical variable with more than 2 unique categories, 
    we will use one-hot encoding.
    """
    # Drop rows where CODE_GENDER == 'XNA' - only 4 rows in training data 
    for df_name, df in (('train_data', train_data), ('test_data', test_data)):
        if 'CODE_GENDER' in df.columns:
            xna_mask = df['CODE_GENDER'] == 'XNA'
            n_xna = int(xna_mask.sum())
            if n_xna:
                df.drop(df[xna_mask].index, inplace=True)
                print(f"Removed {n_xna} rows from {df_name} where CODE_GENDER == 'XNA'")
    
    # Create a label encoder object
    le = LabelEncoder()
    le_count = 0

    encoded_cols = []

    # Iterate through the columns
    for col in train_data:
        if train_data[col].dtype == 'object':
            # If 2 or fewer unique categories (ignore NaNs)
            if train_data[col].nunique() <= 2:
                # Train on the training data
                le.fit(train_data[col])
                # Label encoding - Transform both training and testing data
                train_data[col] = le.transform(train_data[col])
                test_data[col] = le.transform(test_data[col])

                # Keep track of how many columns were label encoded
                le_count += 1
                encoded_cols.append(col)

    # Display which object columns will be label encoded            
    print('%d columns were label encoded.' % le_count)
    if encoded_cols:
        print('Encoded columns:', encoded_cols)
    print(80*'=')

    # Display which object columns will be one-hot encoded 
    one_hot_cols = [c for c in train_data.columns if train_data[c].dtype == 'object' and train_data[c].nunique() > 2]
    print(f"{len(one_hot_cols)} columns were one-hot encoded.")
    if one_hot_cols:
        print('One-hot columns:', one_hot_cols)
    print(80*'=')

    # one-hot encoding of categorical variables
    train_data = pd.get_dummies(train_data)
    test_data = pd.get_dummies(test_data)

    # Check if training and test data are aligned
    if(train_data.shape[1] != test_data.shape[1]+1):
        train_labels = train_data['TARGET']

        # Align the training and testing data, keep only columns present in both dataframes
        train_data, test_data = train_data.align(test_data, join = 'inner', axis = 1)

        # Add the target back in
        train_data['TARGET'] = train_labels

        print("Training Features' columns after one-hot encoding: ", train_data.shape[1])
        print("Testing Features' columns after one-hot encoding: ", test_data.shape[1])

    else:
        print("Training Features' columns after one-hot encoding: ", train_data.shape[1])
        print("Testing Features' columns after one-hot encoding: ", test_data.shape[1])

    return train_data, test_data

def description_analysis(train_data):
    # Compute describe() on numeric-only data and save
    numeric_df = train_data.select_dtypes(include=[np.number])
    desc_df = numeric_df.describe().round(6)
    print("First 5 rows of data description analysis (numeric columns):")
    print(desc_df.head())
    print(80*'=')
    print("Saving data description analysis to 'exploratory_data_analysis/application_train_desc.csv'")
    desc_df.to_csv('exploratory_data_analysis/application_train_desc.csv')
    print(80*'=')

    # Prepare analysis file and header
    analysis_file = 'exploratory_data_analysis/desc_analysis.csv'
    cols = ['statistic'] + list(desc_df.columns)
    if not os.path.exists('exploratory_data_analysis'):
        os.makedirs('exploratory_data_analysis')
    if not os.path.exists(analysis_file):
        pd.DataFrame(columns=cols).to_csv(analysis_file, index=False)

    # Convert describe output to a dataframe with 'statistic' column and a lookup index
    desc_analysis_df = desc_df.reset_index().rename(columns={'index': 'statistic'})
    stats_idx = desc_analysis_df.set_index('statistic')

    # Metric rows (consistent snake_case names)
    metric_rows = {
        'mean_eq_min_max': {'statistic': 'mean_eq_min_max'},
        'mean_median_gap': {'statistic': 'mean_median_gap'},
        'large_std': {'statistic': 'large_std'},
        'extreme_max': {'statistic': 'extreme_max'},
        'long_tail': {'statistic': 'long_tail'},
    }

    # Compute metrics per numeric column
    for col in desc_df.columns:
        try:
            mean = float(stats_idx.at['mean', col])
            std = float(stats_idx.at['std', col])
            min_v = float(stats_idx.at['min', col])
            max_v = float(stats_idx.at['max', col])
            q25 = float(stats_idx.at['25%', col])
            q50 = float(stats_idx.at['50%', col])
            q75 = float(stats_idx.at['75%', col])
        except KeyError:
            # missing statistic row/column; fill blanks
            for mr in metric_rows.values():
                mr[col] = ''
            continue
        except Exception:
            for mr in metric_rows.values():
                mr[col] = ''
            continue

        # Robust metric calculations with guards for small/zero denominators
        # Use a small epsilon relative to typical scale to avoid degenerate tests
        scale = max(abs(mean), abs(q50), 1.0)
        eps = max(1e-8, scale * 1e-6)

        # mean equal to min and max -> 1 if true, 0 otherwise
        metric_rows['mean_eq_min_max'][col] = 1 if (mean == min_v == max_v) else 0

        # mean vs median gap -> 1 if gap > std, else 0
        gap = abs(mean - q50)
        metric_rows['mean_median_gap'][col] = 1 if gap > std else 0

        # large std (relative to mean) -> use floor to avoid mean≈0 issues
        denom = max(abs(mean), eps)
        metric_rows['large_std'][col] = 1 if std >= 0.5 * denom else 0

        # extreme max: prefer median-based test, but fallback when median ~ 0
        # Primary test: max > 5 * |median| when median is sufficiently large
        if abs(q50) > eps:
            metric_rows['extreme_max'][col] = 1 if max_v > 5 * abs(q50) else 0
        else:
            # fallback: compare to mean + 3*std OR q75 + 3*IQR whichever is more conservative
            iqr = q75 - q25
            threshold1 = mean + 3 * std
            threshold2 = q75 + 3 * iqr if iqr > eps else threshold1
            metric_rows['extreme_max'][col] = 1 if max_v > max(threshold1, threshold2) else 0

        # long tail check via IQR rule -> encode as integers:
        # 0 = Not long-tailed, 1 = Left long-tailed, 2 = Right long-tailed, 3 = Both tails
        iqr = q75 - q25
        if iqr <= eps:
            # No spread in IQR; use std-based fallback or mark as not long-tailed
            # If std is very small too, treat as not long-tailed
            if std <= eps:
                metric_rows['long_tail'][col] = 0
            else:
                # use deviations from median scaled by std as a proxy
                right_flag = 1 if (max_v - q50) > 3 * std else 0
                left_flag = 1 if (q50 - min_v) > 3 * std else 0
                metric_rows['long_tail'][col] = (2 if right_flag and not left_flag else
                                                  1 if left_flag and not right_flag else
                                                  3 if left_flag and right_flag else 0)
        else:
            # standard IQR-based rule
            right = max_v > q75 + 1.5 * iqr
            left = min_v < q25 - 1.5 * iqr
            if right and left:
                metric_rows['long_tail'][col] = 3
            elif right:
                metric_rows['long_tail'][col] = 2
            elif left:
                metric_rows['long_tail'][col] = 1
            else:
                metric_rows['long_tail'][col] = 0

    # Append metric rows to desc_analysis_df and save
    appended = pd.concat([desc_analysis_df, pd.DataFrame(list(metric_rows.values()))], ignore_index=True)

    # Add legend row describing the numeric codes used
    legend_text = (
        "mean_eq_min_max (mean equals to min and max values) - 1: true, 0: false"
        "mean_median_gap (where gap = |mean - median|) - 1: large gap, 0: small gap"
        "large_std (if 1/2*mean > std) -  1: true, 0: false"
        "extreme_max (if max > 5*median) - 1: true, 0: false; "
        "long_tail - 0 = Not long-tailed, 1 = Left long-tailed, 2 = Right long-tailed, 3 = Both tails"
        "A = max > 75% + 1.5*iqr, B = min < q25 - 1.5*iqr, if A & B are true, then long-tailed on both sides"
        "If only A or B is true, then long-tailed on that side"
    )

    # Create a legend row with the description in the 'statistic' column and blanks for features
    legend_row = {'statistic': legend_text}
    for c in desc_analysis_df.columns[1:]:
        legend_row[c] = ''

    appended = pd.concat([appended, pd.DataFrame([legend_row])], ignore_index=True)
    appended.to_csv(analysis_file, index=False)
    print(f'Analysis file created: {analysis_file}')

    return desc_df

def anomalies_handling(train_data, test_data):
    """
    From data's description i found anomaly in DAYS_EMPLOYED column
    """

    # Display statistics for DAYS_EMPLOYED
    print(train_data['DAYS_EMPLOYED'].describe())
    print("The maximum value (besides being positive) is about 1000 years")
    # Plotting columns histogram
    train_data['DAYS_EMPLOYED'].plot.hist(title = 'Days Employment Histogram')
    plt.xlabel('Days Employment')

    #  Subset the anomalous clients
    anom = train_data[train_data['DAYS_EMPLOYED'] == 365243]
    non_anom = train_data[train_data['DAYS_EMPLOYED'] != 365243]
    print('The non-anomalies default on %0.2f%% of loans' % (100 * non_anom['TARGET'].mean()))
    print('The anomalies default on %0.2f%% of loans' % (100 * anom['TARGET'].mean()))
    print('There are %d anomalous days of employment' % len(anom))
    print("The anomalies have a lower rate of default.")

    # Create an anomalous flag column
    train_data['DAYS_EMPLOYED_ANOM'] = train_data["DAYS_EMPLOYED"] == 365243

    # Replace the anomalous values with nan
    train_data['DAYS_EMPLOYED'].replace({365243: np.nan}, inplace = True)

    train_data['DAYS_EMPLOYED'].plot.hist(title = 'Days Employment Histogram')
    plt.xlabel('Days Employment')
    plt.close()

    # Fill in the existing column with np.nan in the testing data
    test_data['DAYS_EMPLOYED_ANOM'] = test_data["DAYS_EMPLOYED"] == 365243
    test_data["DAYS_EMPLOYED"].replace({365243: np.nan}, inplace = True)

    print('There are %d anomalies in the test data out of %d entries' % (test_data["DAYS_EMPLOYED_ANOM"].sum(), len(test_data)))

    return 

def data_correlations(train_data, desc_df):
    """
    Interpretations of the absolute value of the correlation coefficent are:
    .00-.19 “very weak”
    .20-.39 “weak”
    .40-.59 “moderate”
    .60-.79 “strong”
    .80-1.0 “very strong”
    """
    print(80*'=')
    print(f"Working with {len(train_data.columns)} numeric columns (including TARGET)")
    print(80*'=')

    # Calculate full correlation matrix
    corr_matrix = train_data.corr()

    # Extract unique pairwise correlations (upper triangle) and sort
    upper_tri = np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    corr_pairs = corr_matrix.where(upper_tri).stack().sort_values()

    # Save full correlation matrix as CSV
    # corr_matrix.to_csv(os.path.join("exploratory_data_analysis", "full_correlation_matrix.csv"))

    # Store all pairwise correlations to CSV
    pairs_df = corr_pairs.reset_index()
    pairs_df.columns = ['feature_1', 'feature_2', 'correlation']

    # Find most positive correlation pairs (correlation > 0.60)
    threshold = 0.60
    high_pairs = pairs_df[pairs_df['correlation'] > threshold].copy()
    high_pairs = high_pairs.sort_values(by='correlation', key=lambda s: s.abs(), ascending=False)
    high_pairs.to_csv(os.path.join('exploratory_data_analysis', 'most_positive_correlation_pairs.csv'), index=False)

    # Find most negative correlation pairs (correlation < -0.60)
    low_pairs = pairs_df[pairs_df['correlation'] < -threshold].copy()
    low_pairs = low_pairs.sort_values(by='correlation', key=lambda s: s.abs(), ascending=False)
    low_pairs.to_csv(os.path.join('exploratory_data_analysis', 'most_negative_correlation_pairs.csv'), index=False)

    # Display summary
    print(80*'=')
    print(f"Saved most_positive_correlation_pairs and most_negative_correlation_pairs |corr|>{threshold}).")
    print('Most Positive Correlations:\n', high_pairs.head(15))
    print('\nMost Negative Correlations:\n', low_pairs.head(15))
    print(80*'=')

    # Find correlations with the target and sort
    target_corr = train_data.corr()['TARGET'].sort_values()

    # Save full correlation matrix as CSV
    print(80*'=')
    print("Saving correlation to TARGET matrix to 'exploratory_data_analysis")
    target_corr.to_csv(os.path.join("exploratory_data_analysis", "target_correlation_matrix.csv"))
    print(80*'=')
    
    # Display correlations to target
    print(80*'=')
    print('Most Positive Correlations:\n', target_corr.tail(15))
    print('\nMost Negative Correlations:\n', target_corr.head(15))
    print(80*'=')

    # Correlation heatmap of statistics
    # Accept either a describe() DataFrame (stats in the index) or an analysis DataFrame
    if 'statistic' not in desc_df.columns:
        stats_df = desc_df.reset_index().rename(columns={'index': 'statistic'})
    else:
        stats_df = desc_df.copy()

    # Set 'statistic' as the index for selection
    stats_df = stats_df.set_index('statistic')

    # Select only the common statistic rows that exist
    wanted = ['mean', 'std', 'min', '25%', '50%', '75%', 'max']
    existing = [r for r in wanted if r in stats_df.index]
    if not existing:
        print('No descriptive statistic rows found to compute statistic correlations.')
        correlation_between_stats = pd.DataFrame()
    else:
        selected = stats_df.loc[existing]
        correlation_between_stats = selected.T.corr()

    # Save statistics correlation csv
    correlation_between_stats.to_csv(os.path.join("exploratory_data_analysis", "statistics_correlation_matrix.csv"))
    # Create heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(correlation_between_stats, 
                annot=True,
                cmap='coolwarm',
                fmt='.2f',
                square=True,
                linewidths=0.5)

    plt.title(f'Correlation Heatmap of Statistics')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)

    # Save the plot
    output_path = f'exploratory_data_analysis/statistics_correlation_heatmap.png'
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()
    
    return 

def corr_exam(train_data):
    """
    Examing the variables with the with the strongest 
    possitve correlation (DAYS_BIRTH) 
    and negative correlations (EXT_SOURCE_1, EXT_SOURCE_2, and EXT_SOURCE_3)
    with the TARGET
    """

    # Find the correlation of the positive days since birth and target
    train_data['DAYS_BIRTH'] = train_data['DAYS_BIRTH'].abs()
    print(train_data['DAYS_BIRTH'].corr(train_data['TARGET']))

    # Set the style of plots
    plt.style.use('fivethirtyeight')

    # Plot the distribution of ages in years
    plt.figure(figsize=(8, 5))
    plt.hist(train_data['DAYS_BIRTH'] / 365, edgecolor='k', bins=25)
    plt.title('Age of Client'); plt.xlabel('Age (years)'); plt.ylabel('Count')
    plt.savefig('exploratory_data_analysis/age_distribution.png', bbox_inches='tight', dpi=300)
    plt.close()

    # Prepare age dataframe once, create YEARS_BINNED, then group
    age_data = train_data[['TARGET', 'DAYS_BIRTH']].copy()
    age_data['YEARS_BIRTH'] = age_data['DAYS_BIRTH'] / 365
    age_data['YEARS_BINNED'] = pd.cut(age_data['YEARS_BIRTH'], bins=np.linspace(20, 70, num=11))

    # Group by the bin and calculate averages
    age_groups = age_data.groupby('YEARS_BINNED').mean()

    # Plot average failure to repay loans by age bracket
    plt.figure(figsize=(8, 8))
    plt.bar(age_groups.index.astype(str), 100 * age_groups['TARGET'])
    plt.xticks(rotation=75); plt.xlabel('Age Group (years)'); plt.ylabel('Failure to Repay (%)')
    plt.title('Failure to Repay by Age Group')
    plt.tight_layout()
    plt.savefig('exploratory_data_analysis/failure_by_age_group.png', bbox_inches='tight', dpi=300)
    plt.close()

    # Extract the EXT_SOURCE variables and show correlations
    ext_data = train_data[['TARGET', 'EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'DAYS_BIRTH']]
    ext_data_corrs = ext_data.corr()

    # Display the correlations of the EXT_SOURCE features with the target and with each other
    plt.figure(figsize = (8, 6))

    # Heatmap of correlations
    sns.heatmap(ext_data_corrs, cmap = plt.cm.RdYlBu_r, vmin = -0.25, annot = True, vmax = 0.6)
    plt.title('Correlation Heatmap')
    plt.savefig('exploratory_data_analysis/ext_source_correlation_heatmap.png', bbox_inches='tight', dpi=300)  
    plt.close()
    
    # Display the distribution of each of these features colored by the value of the target
    plt.figure(figsize = (10, 12))

    # iterate through the sources
    for i, source in enumerate(['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']):
        
        # create a new subplot for each source
        plt.subplot(3, 1, i + 1)
        # plot repaid loans
        sns.kdeplot(train_data.loc[train_data['TARGET'] == 0, source], label = 'target == 0')
        # plot loans that were not repaid
        sns.kdeplot(train_data.loc[train_data['TARGET'] == 1, source], label = 'target == 1')
        
        # Label the plots
        plt.title('Distribution of %s by Target Value' % source)
        plt.xlabel('%s' % source); plt.ylabel('Density');
        
    plt.tight_layout(h_pad = 2.5)
    plt.savefig('exploratory_data_analysis/ext_source_distribution_by_target.png', bbox_inches='tight', dpi=300)
    plt.close()

    return 


def main():
    # List files available
    # print(os.listdir("data/"))

    # ===============================================EDA===============================================
    print(80*'=')
    print("========================DATA ANALYSIS========================")
    print(80*'=')
    train_data, test_data = load_and_preview_data() # Load and preview data

    # print("============================EXAMINING MISSING VALUES=============================")
    # missing_values(train_data)

    # print("============================LABEL ENCODING AND ONE-HOT ENCODING=============================")
    # train_data_en, test_data_en = objects_encoding(train_data, test_data)
    train_data, categorical_features = obj_encoding(train_data)
    test_data, _ = obj_encoding(test_data)

    # ==========================================FEATURE IMPORTANCE==========================================
    actual_imp_df, null_imp_df = get_features_distribution(train_data, categorical_features)
    display_distributions(actual_imp_df, null_imp_df, feature='LIVINGAPARTMENTS_AVG')
    display_distributions(actual_imp_df, null_imp_df, feature='CODE_GENDER')

    features_scores(actual_imp_df, null_imp_df)
    corr_scores = correlation_scores(actual_imp_df, null_imp_df)

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

    print("============================DATA DESCRIPTION AND ANALYSIS=============================")
    desc_df = description_analysis(train_data)

    print("=====================================HANDLING ANOMALIES======================================")
    anomalies_handling(train_data, test_data)

    print("=====================================CORRELATIONS======================================")
    data_correlations(train_data, desc_df)

    print("=====================================EXAMING CORRELATIONS======================================")
    corr_exam(train_data)

    # ==========================================FEATURE ENGINEERING==========================================


if __name__ == "__main__":
    main()