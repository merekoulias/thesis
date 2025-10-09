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
import seaborn as sns

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
        print("Saving file's head to data_analysis/application_train_head.csv")
        head_df.to_csv('data_analysis/application_train_head.csv', index=False)
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

    print("================EXAMINING THE DISTRIBUTION OF THE TARGET COLUMN================")
    print(train_data['TARGET'].value_counts())
    print(80*'=')

    # Examining the distribution of the protected attribute
    print("================EXAMINING THE DISTRIBUTION OF THE GENDER COLUMN================")
    print(train_data['CODE_GENDER'].value_counts())
    print(80*'=')

    print("Plotting TARGET histogram to see class distribution")
    target_hist = train_data['TARGET'].astype(int).plot.hist()
    target_hist.set_xlabel("TARGET")
    target_hist.set_ylabel("Number of Applicants")
    print("Saving histogram to data_analysis/target_histogram.png")
    plt.savefig('data_analysis/target_histogram.png', bbox_inches='tight', dpi=300)
    plt.show()
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
        print("Saving missing values table to 'data_analysis/missing_values_table.csv'")
        mis_val_table_ren_columns.to_csv('data_analysis/missing_values_table.csv')

        print("First 20 rows of missing values table:")
        print(mis_val_table_ren_columns.head(20))
        print(80*'=')
        return 

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
    print("Saving data description analysis to 'data_analysis/application_train_desc.csv'")
    desc_df.to_csv('data_analysis/application_train_desc.csv')
    print(80*'=')

    # Prepare analysis file and header
    analysis_file = 'data_analysis/desc_analysis.csv'
    cols = ['statistic'] + list(desc_df.columns)
    if not os.path.exists('data_analysis'):
        os.makedirs('data_analysis')
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
    # corr_matrix.to_csv(os.path.join("data_analysis", "full_correlation_matrix.csv"))

    # Store all pairwise correlations to CSV
    pairs_df = corr_pairs.reset_index()
    pairs_df.columns = ['feature_1', 'feature_2', 'correlation']

    # Find most positive correlation pairs (correlation > 0.60)
    threshold = 0.60
    high_pairs = pairs_df[pairs_df['correlation'] > threshold].copy()
    high_pairs = high_pairs.sort_values(by='correlation', key=lambda s: s.abs(), ascending=False)
    high_pairs.to_csv(os.path.join('data_analysis', 'most_positive_correlation_pairs.csv'), index=False)

    # Find most negative correlation pairs (correlation < -0.60)
    low_pairs = pairs_df[pairs_df['correlation'] < -threshold].copy()
    low_pairs = low_pairs.sort_values(by='correlation', key=lambda s: s.abs(), ascending=False)
    low_pairs.to_csv(os.path.join('data_analysis', 'most_negative_correlation_pairs.csv'), index=False)

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
    print("Saving correlation to TARGET matrix to 'data_analysis")
    target_corr.to_csv(os.path.join("data_analysis", "target_correlation_matrix.csv"))
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
    correlation_between_stats.to_csv(os.path.join("data_analysis", "statistics_correlation_matrix.csv"))
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
    output_path = f'data_analysis/statistics_correlation_heatmap.png'
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.show()
    
    return 

def main():
    # List files available
    # print(os.listdir("data/"))

    print(80*'=')
    print("========================DATA ANALYSIS========================")
    print(80*'=')
    train_data, test_data = load_and_preview_data() # Load and preview data

    print("============================EXAMINING MISSING VALUES=============================")
    missing_values(train_data)

    print("============================LABEL ENCODING AND ONE-HOT ENCODING=============================")
    train_data_en, test_data_en = objects_encoding(train_data, test_data)

    print("============================DATA DESCRIPTION AND ANALYSIS=============================")
    desc_df = description_analysis(train_data)

    print("=====================================CORRELATIONS======================================")
    data_correlations(train_data_en, desc_df)


if __name__ == "__main__":
    main()