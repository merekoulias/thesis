# from copilot 
# for testing a different feature importance approach

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.inspection import permutation_importance

def fit_lgb_get_importances(X, y,
                           categorical_feature=None,
                           test_size=0.2,
                           random_state=42,
                           lgb_params=None,
                           num_boost_round=10000,
                           early_stopping_rounds=50,
                           compute_permutation=False,
                           perm_n_repeats=10,
                           perm_random_state=42,
                           verbose_eval=False):
    """
    Train a LightGBM classifier and return feature importances (gain and split).
    Optionally compute permutation importances on a validation split.
    
    Parameters
    ----------
    X : pd.DataFrame
        Feature dataframe (columns = feature names).
    y : pd.Series or np.array
        Binary target (0/1).
    categorical_feature : list or None
        List of column names that are categorical (or indices). Pass None to infer.
    test_size : float
        Fraction of data to keep for validation/permutation.
    lgb_params : dict or None
        LightGBM parameters. If None, sensible defaults are used.
    compute_permutation : bool
        If True, compute permutation importance on the validation set.
    perm_n_repeats : int
        Number of repeats for permutation importance.
    Returns
    -------
    importances_df : pd.DataFrame
        DataFrame with columns ['feature', 'importance_gain', 'importance_split', 'perm_importance_mean' (optional)]
    model : trained lgb.Booster or LGBMClassifier
        Trained model object (sklearn API LGBMClassifier).
    val_score : float
        ROC AUC on validation set (if computed).
    """
    if lgb_params is None:
        lgb_params = {
            'objective': 'binary',
            'boosting_type': 'gbdt',
            'learning_rate': 0.05,
            'num_leaves': 31,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 0.0,
            'reg_lambda': 0.0,
            'verbose': -1,
            'n_jobs': -1,
            'seed': random_state
        }

    # train / validation split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    # LightGBM dataset
    lgb_train = lgb.Dataset(X_train, y_train, categorical_feature=categorical_feature, free_raw_data=False)
    lgb_valid = lgb.Dataset(X_val, y_val, reference=lgb_train, categorical_feature=categorical_feature, free_raw_data=False)
    
    # train with early stopping (using native API to get booster easily)
    booster = lgb.train(
        params=lgb_params,
        train_set=lgb_train,
        num_boost_round=num_boost_round,
        valid_sets=[lgb_valid],
        early_stopping_rounds=early_stopping_rounds,
        verbose_eval=verbose_eval
    )
    
    # validation score
    y_pred_val = booster.predict(X_val, num_iteration=booster.best_iteration)
    val_score = roc_auc_score(y_val, y_pred_val)

    # feature importances from booster (importance_type: 'gain' and 'split')
    feature_names = booster.feature_name()
    gain_importances = booster.feature_importance(importance_type='gain')
    split_importances = booster.feature_importance(importance_type='split')

    importances_df = pd.DataFrame({
        'feature': feature_names,
        'importance_gain': gain_importances,
        'importance_split': split_importances
    }).sort_values('importance_gain', ascending=False).reset_index(drop=True)

    # Optionally compute permutation importances on the validation set (model-agnostic)
    if compute_permutation:
        # Use the booster predict method; sklearn's permutation_importance expects an estimator with predict_proba or predict
        # We'll wrap a tiny callable
        def booster_predict_proba(X_):
            return booster.predict(X_, num_iteration=booster.best_iteration)

        # permutation_importance expects an estimator; we can pass a dummy and use scoring with predictions function via scoring param is not available,
        # so use custom approach: compute baseline and permute columns manually (simple, but clear)
        perm_means = []
        rng = np.random.RandomState(perm_random_state)
        baseline = roc_auc_score(y_val, y_pred_val)
        for col in feature_names:
            drops = []
            for _ in range(perm_n_repeats):
                X_val_perm = X_val.copy()
                X_val_perm[col] = X_val_perm[col].sample(frac=1.0, random_state=rng.randint(0, 2**31-1)).reset_index(drop=True)
                perm_pred = booster.predict(X_val_perm, num_iteration=booster.best_iteration)
                perm_score = roc_auc_score(y_val, perm_pred)
                drops.append(baseline - perm_score)
            perm_means.append(np.mean(drops))
        importances_df['perm_importance_mean'] = perm_means
        importances_df = importances_df.sort_values('perm_importance_mean', ascending=False).reset_index(drop=True)

    return importances_df, booster, val_score

# Example usage:
if __name__ == "__main__":
    # simple example: generate random data or load your dataset
    from sklearn.datasets import make_classification
    Xs, ys = make_classification(n_samples=2000, n_features=20, n_informative=5, random_state=0)
    X = pd.DataFrame(Xs, columns=[f'f{i}' for i in range(Xs.shape[1])])
    y = pd.Series(ys)

    imp_df, model, auc = fit_lgb_get_importances(X, y, compute_permutation=True, perm_n_repeats=5, verbose_eval=False)
    print(f"Validation AUC: {auc:.6f}")
    print(imp_df.head(30))