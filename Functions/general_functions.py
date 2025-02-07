import pandas as pd
import numpy as np
import ast
import json
import sklearn

from sklearn.model_selection import train_test_split, KFold, GroupKFold, GridSearchCV, StratifiedGroupKFold
from sklearn.metrics import accuracy_score, f1_score, make_scorer
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
# from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB


import warnings
from sklearn.exceptions import ConvergenceWarning

from extract_features import *

# Suppress ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

def process_time_series(data, window_size, overlap, pref, feature_dicts, time_series_dicts):
    """
    Process a time series into overlapping windows and call function to retireve features of these windows

    Parameters:
    - data (list): Time series of interest that will be added to the feature_dicts and time_series_dicts
    - window_size (int): Window size in number of samples
    - overlap (float) : Overlap ratio (0 till 1)
    - pref (str): Prefix used in dataframe for distinction between features from different time series
    - feature_dicts (dict): Current features and their associated labels
    - time_series_dicts (dict): Current time series and their associated labels 
    
    Returns:
    - features (dict): Updated dict containing features and their associated labels;
        added features for data 
    - time_series (dict): Updated dict containing time series and their associated labels 
    """

    step_size = int(window_size * (1 - overlap))
    total_windows = max(0, (len(data) - window_size) // step_size + 1)

    if len(data) < window_size:
        print(f"length of data is {len(data)}, where the length of the window is {window_size}")
        raise ValueError(f"Yo man, that window is too big")

    for step in range(0, total_windows):
        window = data[step*step_size : step*step_size + window_size]
        features = extract_features(window, prefix=pref)
        feature_dicts[step].update(features)
        ts = {}

        ts[f'{pref}ts'] = list(window)
        # ts = {f"{pref}ts": json.dumps(list(window))}

        time_series_dicts[step].update(ts)

    return feature_dicts, time_series_dicts


def cl_grid():
    param_grids = {
    'LogisticRegression': {
        'C': [0.1, 1, 10, 50, 100, 150, 200],
        'solver': ['lbfgs'],
        'penalty': ['l2'],
        'multi_class': ['multinomial'],
        'max_iter': [50, 100, 150, 200, 300, 400]
    },
    'SVC': {
        'C': [0.1, 1, 10, 100, 200],
        'max_iter': [500, 1000, 2000], 
        'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
        'gamma': ['scale', 'auto']
    },
    'RandomForestClassifier': {
        'n_estimators': [10, 50, 100, 150],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'class_weight': ['balanced', 'balanced_subsample', None]
    },
    'KNeighborsClassifier': {
        'n_neighbors': [3, 5, 7, 9],
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan']
    },
    'DecisionTreeClassifier': {
        'criterion': ['gini', 'entropy'],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10]
    },
    # 'MLPClassifier': {
    # 'hidden_layer_sizes': [(50,), (100,), (100, 50)],
    # 'activation': ['relu', 'tanh'],
    # 'learning_rate': ['constant', 'adaptive'],
    # 'max_iter': [200, 500]
    # },
    'GaussianNB': {}
    }

    classifiers = {
        'LogisticRegression': LogisticRegression(random_state=42),
        'SVC': SVC(random_state=42),
        'RandomForestClassifier': RandomForestClassifier(random_state=42),
        'KNeighborsClassifier': KNeighborsClassifier(),
        'DecisionTreeClassifier': DecisionTreeClassifier(random_state=42),
        # 'MLPClassifier': MLPClassifier(random_state=42),
        'GaussianNB': GaussianNB(),
    }
    return param_grids, classifiers

def simple_models(data_df, k, split_type='random'):
    """
    Train and evaluate multiple classifiers using different splitting strategies.

    Parameters:
    - data_df (pd.dataframe): the input features dataset with labels
    - k (int): number of splits for cross-validation
    - split_type: str, the splitting strategy ('random' or 'group_kfold' or 'stratified_group_kfold')
    """
    [param_grids, classifiers] = cl_grid()
    f1_macro_scorer = make_scorer(f1_score, average='macro')

    # Initialize results
    all_results = []

    # Unique groups (bearings)
    unique_bearings = data_df['bearing_id'].unique()
    
    # Loop through classifiers
    for model_name, model in classifiers.items():
        grid_search_results = []
        param_grid = param_grids[model_name]
        print(f"Training {model_name} using {split_type} split")

        if split_type == 'random':
            # Random splitting
            for fold in range(k):
                train_df, test_df = train_test_split(data_df, test_size=1/k, random_state=42+fold, stratify=data_df['label'])
                
                X_train, y_train = train_df.drop(columns=['label', 'bearing_id']), train_df['label']
                X_test, y_test = test_df.drop(columns=['label', 'bearing_id']), test_df['label']

                # Scale features
                scaler = StandardScaler()
                X_train = scaler.fit_transform(X_train)
                X_test = scaler.transform(X_test)

                grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='accuracy', verbose=0)
                grid_search.fit(X_train, y_train)
                best_model = grid_search.best_estimator_
                y_pred = best_model.predict(X_test)

                grid_search_results.append({
                    'fold': fold+1,
                    'best_params': grid_search.best_params_,
                    'test_accuracy': accuracy_score(y_test, y_pred),
                    'f1_score': f1_score(y_test, y_pred, average='macro')
                })

        elif split_type == 'stratified_group_kfold':
            # Stratified Group K-Fold
            sgkf = StratifiedGroupKFold(n_splits=k)
            for fold, (train_idx, test_idx) in enumerate(sgkf.split(data_df, data_df['label'], groups=data_df['bearing_id']), start=1):
                train_df, test_df = data_df.iloc[train_idx], data_df.iloc[test_idx]

                X_train, y_train = train_df.drop(columns=['label', 'bearing_id']), train_df['label']
                X_test, y_test = test_df.drop(columns=['label', 'bearing_id']), test_df['label']

                # shuffle data
                X_train, y_train = sklearn.utils.shuffle(X_train, y_train)
                X_test, y_test = sklearn.utils.shuffle(X_test, y_test) 

                # Scale features
                scaler = StandardScaler()
                X_train = scaler.fit_transform(X_train)
                X_test = scaler.transform(X_test)

                grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='accuracy', verbose=0)
                grid_search.fit(X_train, y_train)
                best_model = grid_search.best_estimator_
                y_pred = best_model.predict(X_test)

                grid_search_results.append({
                    'fold': fold,
                    'best_params': grid_search.best_params_,
                    'test_accuracy': accuracy_score(y_test, y_pred),
                    'f1_score': f1_score(y_test, y_pred, average='macro')
                })

        elif  split_type == 'group_kfold':# Default Group K-Fold
            kf = GroupKFold(n_splits=k)
            for fold, (train_idx, test_idx) in enumerate(kf.split(data_df, groups=data_df['bearing_id']), start=1):
                train_df, test_df = data_df.iloc[train_idx], data_df.iloc[test_idx]

                X_train, y_train = train_df.drop(columns=['label', 'bearing_id']), train_df['label']
                X_test, y_test = test_df.drop(columns=['label', 'bearing_id']), test_df['label']

                # shuffle data
                X_train, y_train = sklearn.utils.shuffle(X_train, y_train)
                X_test, y_test = sklearn.utils.shuffle(X_test, y_test) 
                # Scale features
                scaler = StandardScaler()
                X_train = scaler.fit_transform(X_train)
                X_test = scaler.transform(X_test)

                grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='accuracy', verbose=0)
                grid_search.fit(X_train, y_train)
                best_model = grid_search.best_estimator_
                y_pred = best_model.predict(X_test)

                grid_search_results.append({
                    'fold': fold,
                    'best_params': grid_search.best_params_,
                    'test_accuracy': accuracy_score(y_test, y_pred),
                    'f1_score': f1_score(y_test, y_pred, average='macro')
                })

        elif split_type == 'balanced_device':
            # Balanced device-based split
            device_class_dist = data_df.groupby('bearing_id')['label'].value_counts(normalize=True).unstack(fill_value=0)
            sorted_devices = device_class_dist.index.tolist()

            train_devices, test_devices = [], []
            train_classes = pd.Series(0, index=device_class_dist.columns)
            test_classes = pd.Series(0, index=device_class_dist.columns)

            for device in sorted_devices:
                if (train_classes + device_class_dist.loc[device]).max() < (test_classes + device_class_dist.loc[device]).max():
                    train_devices.append(device)
                    train_classes += device_class_dist.loc[device]
                else:
                    test_devices.append(device)
                    test_classes += device_class_dist.loc[device]

            train_df = data_df[data_df['bearing_id'].isin(train_devices)]
            test_df = data_df[data_df['bearing_id'].isin(test_devices)]

            X_train, y_train = train_df.drop(columns=['label', 'bearing_id']), train_df['label']
            X_test, y_test = test_df.drop(columns=['label', 'bearing_id']), test_df['label']

            # Scale features
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

            grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='accuracy', verbose=0)
            grid_search.fit(X_train, y_train)
            best_model = grid_search.best_estimator_
            y_pred = best_model.predict(X_test)

            grid_search_results.append({
                'fold': 1,  # Single split for this method
                'best_params': grid_search.best_params_,
                'test_accuracy': accuracy_score(y_test, y_pred),
                'f1_score': f1_score(y_test, y_pred, average='macro')
            })
        else:
            raise SyntaxError("Invalid 'split_type' parameter is used")

        # Compile results for this model
        model_results_df = pd.DataFrame(grid_search_results)
        model_results_df['model'] = model_name
        all_results.append(model_results_df)

    # Combine results from all models
    final_results = pd.concat(all_results, ignore_index=True)
    return final_results
  

# def split_data(df, target_column, target_series='all', split_type='random', test_size=0.2, group_column=None, n_splits=3):
#     """
#     Split data using different techniques.

#     Parameters:
#     - df (pd.DataFrame): The input dataframe.
#     - target_column (str): The name of the target column.
#     - split_type (str): The splitting strategy ('random', 'group_kfold', 'stratified_group_kfold').
#     - test_size (float): The proportion of the dataset to include in the test split (for random split).
#     - group_column (str): Column name to define groups (required for group-based splits).
#     - n_splits (int): Number of splits for K-Fold.

#     Returns:
#     - X_train, X_test, y_train, y_test: Training and testing sets for features and targets.
#     """
#     print('wtf is happening')

#     # if len(target_series) == 0:
#     if target_series == 'all':
#         print('wtf is happening')
#         X = df.iloc[:, :-2]
#     else:
#         X = df[target_series]
#     y = df[target_column]

    
#     if isinstance(X.iloc[0], np.ndarray):
#         X = np.vstack(X)
#     else:
#         X = X.values.reshape(-1, 1)

#     if split_type == 'random':
#         X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
#     elif split_type == 'group_kfold':
#         if group_column is None:
#             raise ValueError("group_column must be provided for group_kfold splitting.")
#         gkf = GroupKFold(n_splits=n_splits)
#         groups = df[group_column]
#         train_idx, test_idx = next(gkf.split(X, y, groups=groups))
#         X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
#         y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

#     elif split_type == 'stratified_group_kfold':
#         if group_column is None:
#             raise ValueError("group_column must be provided for stratified_group_kfold splitting.")
#         sgkf = StratifiedGroupKFold(n_splits=n_splits)
#         groups = df[group_column]
#         train_idx, test_idx = next(sgkf.split(X, y, groups=groups))
#         X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
#         y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

#     else:
#         raise ValueError("Invalid split_type. Use 'random', 'group_kfold', or 'stratified_group_kfold'.")
    
#     # Scale features
#     scaler = StandardScaler()
#     X_train = scaler.fit_transform(X_train)
#     X_test = scaler.transform(X_test)
    
#     return X_train, X_test, y_train, y_test


def split_data(df, target_column, target_series='all', split_type='random', test_size=0.2, group_column=None, n_splits=3):
    """
    Split data using different techniques.

    Parameters:
    - df (pd.DataFrame): The input dataframe.
    - target_column (str): The name of the target column.
    - split_type (str): The splitting strategy ('random', 'group_kfold', 'stratified_group_kfold').
    - test_size (float): The proportion of the dataset to include in the test split (for random split).
    - group_column (str): Column name to define groups (required for group-based splits).
    - n_splits (int): Number of splits for K-Fold.

    Returns:
    - X_train, X_test, y_train, y_test: Training and testing sets for features and targets.
    """

    # if len(target_series) == 0:
    if target_series == 'all':
        X = df.iloc[:, :-2]
    else:
        X = df[target_series]
    y = df[target_column]

    if split_type == 'random':
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    elif split_type == 'group_kfold':
        if group_column is None:
            raise ValueError("group_column must be provided for group_kfold splitting.")
        gkf = GroupKFold(n_splits=n_splits)
        groups = df[group_column]
        train_idx, test_idx = next(gkf.split(X, y, groups=groups))
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        # shuffle data
        X_train, y_train = sklearn.utils.shuffle(X_train, y_train)
        X_test, y_test = sklearn.utils.shuffle(X_test, y_test) 

    elif split_type == 'stratified_group_kfold':
        if group_column is None:
            raise ValueError("group_column must be provided for stratified_group_kfold splitting.")
        sgkf = StratifiedGroupKFold(n_splits=n_splits)
        groups = df[group_column]
        train_idx, test_idx = next(sgkf.split(X, y, groups=groups))
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        # shuffle data
        X_train, y_train = sklearn.utils.shuffle(X_train, y_train)
        X_test, y_test = sklearn.utils.shuffle(X_test, y_test) 

    else:
        raise ValueError("Invalid split_type. Use 'random', 'group_kfold', or 'stratified_group_kfold'.")
    
    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    return X_train, X_test, y_train, y_test