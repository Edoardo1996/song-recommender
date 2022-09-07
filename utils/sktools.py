"""
Module for applying and scoring a ML model for a regression or classification problem.
It defines a pipeline where outliers, encoders, scalers and ML model can be tuned.
"""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import classification_report, mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, OrdinalEncoder


def remove_outliers(data: pd.DataFrame, skip_columns: list[str],
                    threshold: float = 1.5, verbose: bool = False) -> pd.DataFrame:
    """
    Removes outliers from the dataset.

    Parameters:
    ----------
    data: pandas.DataFrame
        Datasert
    skip_columns: list[str]
        Columns for which outliers are not removed
    threshold: float, default=1.5
        Threshold for removal of outliers
    verbose: bool, default=False,
        if True print details of removal of outliers

    Returns:
    -------
    data: pandas.DataFrame
        Dataset without outliers
    """
    initial_size = len(data)
    for col in data.select_dtypes(np.number).columns:
        if col not in skip_columns:
            upper = np.percentile(data[col], 75)
            lower = np.percentile(data[col], 25)
            iqr = upper - lower
            upper_limit = upper + threshold * iqr
            lower_limit = lower - threshold * iqr
            data = data[(data[col] > lower_limit) & (data[col] < upper_limit)]
            assert not data.empty, 'Threshold too high for col: ' + col
    if verbose:
        print('Outliers removal has removed {} rows ({} % of initial size)'.format(
            initial_size-len(data), round((1-len(data)/initial_size)*100, 2)
        ))
    return data


def split_data(data: pd.DataFrame, target: str, test_size: float = 0.3,
               random_state: int = 42):
    """
    Split the dataset into random train and test subset

    Parameters:
    ----------
    data: pandas.DataFrame
        Dataset
    target: str
        Column that denotes the target for predictions
    test_size: float, default=0.3
        Size of the test subset
    random_state: int, default=42
        Controls the shuffling applied to the data before the split.

    Returns:
    -------
    X_train: np.array
        Data of training features
    X_test: np.array
        Data of test features
    y_train: np.array)
        Data of train target
    y_test: np.array
        Data of test target
    """
    X = data.drop([target], axis=1)
    y = data[target]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    return X_train, X_test, y_train, y_test


def scale_data(X_train: np.array, X_test: np.array, scaler_class) -> None:
    """
    Fit chosen scaler class from scikit-learn on training data and transforms
    both training and test data.
    TODO: reduce lines

    Parameters:
    ----------
    X_train: np.array
        Data of training features
    X_test: np.array
        Data of test features
    scaler_class: Callable
        Class of scikit-learn for scaling the data
    """
    if not isinstance(scaler_class, FunctionTransformer):
        scaler = scaler_class.fit(X_train.select_dtypes(np.number))
    else:
        scaler = scaler_class

    X_train_scaled = scaler.transform(X_train.select_dtypes(np.number))
    X_train[X_train.select_dtypes(np.number).columns] = X_train_scaled
    X_test_scaled = scaler.transform(X_test.select_dtypes(np.number))
    X_test[X_test.select_dtypes(np.number).columns] = X_test_scaled

    all_finite = np.all(np.isfinite(X_train.select_dtypes(np.number)).values)
    if not all_finite:
        X_test.replace([np.inf, -np.inf], np.nan, inplace=True)
        X_train.replace([np.inf, -np.inf], np.nan, inplace=True)
        X_train.dropna(inplace=True)
        X_test.dropna(inplace=True)


def uniform_after_scaling(X_train, y_train, X_test, y_test):
    """
    Uniform target and features after a row-dropping scaling.
    """
    for X, y in zip([X_train, X_test], [y_train, y_test]):
        index_to_drop = list(set(y.index) - set(X.index)) + \
            list(set(X.index) - set(y.index))
        y.drop(index_to_drop, inplace=True)
        assert len(X) == len(y), 'Features and target are not uniform'


def encode_data(X_train: np.array, X_test: np.array, encoders: list,
                cols_to_encode: list[str]):
    """
    Fit chosen encoding class from scikit-learn on training data and transforms
    both training and test data.
    TODO: refactor

    Parameters:
    ----------
    X_train: np.array
        Data of training features
    X_test: np.array
        Data of test features
    encoders: list
        List of encoders to apply to corresponding cols_to_encode
    cols_to_encode: list[str]
        List of columns in which apply the encoding

    Returns:
    X_train: np.array
        Data of encoded training features
    X_test: np.array
        Data of encoded test features
    """
    for encoder, cols in zip(encoders, cols_to_encode):  # loop on chosen encoders and columns
        if (isinstance(encoder, OrdinalEncoder)) and bool(list(cols)):
            for col in cols:
                encoder_model = encoder.fit(X_train[col].values.reshape(-1, 1))
                X_train[col] = encoder_model.transform(
                    X_train[col].values.reshape(-1, 1))
                X_test[col] = encoder_model.transform(
                    X_test[col].values.reshape(-1, 1))
        elif isinstance(encoder, OneHotEncoder) and bool(list(cols)):
            encoder_model = encoder.fit(X_train[cols])
            X_train_onehot_encoded = pd.DataFrame(
                encoder_model.transform(X_train[cols]).toarray())
            X_train_onehot_encoded = X_train_onehot_encoded.set_index(
                X_train.index)
            X_train = X_train.drop(X_train[cols], axis=1).join(
                X_train_onehot_encoded)
            X_test_onehot_encoded = pd.DataFrame(
                encoder_model.transform(X_test[cols]).toarray())
            X_test_onehot_encoded = X_test_onehot_encoded.set_index(
                X_test.index)
            X_test = X_test.drop(X_test[cols], axis=1).join(
                X_test_onehot_encoded)
        elif bool(list(cols)):
            # Columns are present but no encoder was recognized
            raise ValueError("Encoder not recognized, please use another")

    return X_train, X_test


def apply_model(X_train: np.array, X_test: np.array, y_train: np.array, model,
                return_formula: bool, k: int == None) -> np.array:
    """
    Apply a ML model to a scaled and encoded dataset.
    # TODO: check istance of model class. For now is assumed that if k != None.
    a KNClassifier or KNRegressor is used.

    Parameters:
    ----------
    X_train: np.array
        Data of training features
    X_test: np.array
        Data of test features
    y_train: np.array)
        Data of train target
    model: Callable
        ML model (classifier or regressor)
    return_forumula: bool
        If True, returns the mathematical coefficients and intercept of a linear
        model. Fails if model is not linear
    k: int, default=None
        Number of neighbors to use for a KNClassifier or KNRegressor
    """
    if k:
        # optimization have to be implemented
        model = eval(str(model).strip('()')+'(n_neighbors={})'.format(k))
    model.fit(X_train, y_train)

    if return_formula:
        print('Coefficients:')
        print(model.coef_, end='\n\n')
        print('Intercept:')
        print(model.intercept_, end='\n\n')

    return model.predict(X_test.dropna())


def save_results(path, results, append=True, variable=None):
    """
    Save results and metrics in a file for further manipulation
    Data is in format: r2, mae, mse
    TODO: refactor
    """
    if not os.path.isdir(path.split('/')[0]):
        os.makedirs(path.split('/')[0])
    if append:
        f = open(path, 'a')
    else:
        f = open(path, 'w')
    f.write(str(results).replace('(', '').replace(')', ''))
    if variable:
        f.write(' ' + str(variable) + '\n')
    else:
        f.write(' ' + str(variable) + '\n')
    f.close()

def score_regression_model(data: pd.DataFrame, 
                           target: str, 
                           model,
                           return_formula: bool = False,
                           cols_to_encode: list[str] = None,
                           scaler=None,
                           encoders=None,
                           cols_to_drop: list[str] = [],
                           test_size: float = 0.3, 
                           random_state: int = 42,
                           outsiders_thresh: float = None,
                           skip_outsiders_cols: list[str] = [],
                           k = None):
    """
    Scores a Regression Model, it assumes data is already cleaned

    Parameters:
    ----------
    data: pandas.DataFrame: 
        Dataset for the model
    target: str
        Name of target column
    model: Callable
        ML model for regression
    outsiders_thresh: float
        Threshold for the removal of outliers
    return_forumula: bool
        If True, returns the mathematical coefficients and intercept of a linear model
    cols_to_encode: list[str]
        List of columns in which apply the encoding
    scaler: Callable
        Class of scikit-learn for scaling the data
    encoders: list[Callable]
        List of encoders to apply to corresponding cols_to_encode
    cols_to_drop: list[str]
        List of columns' names to be dropped from the dataset
    test_size: float, default=0.3
            Size of the test subset
    random_state: int, default=42
        Controls the shuffling applied to the data before the split.
    outsiders_thresh: float
        Threshold for the removal of outliers
    skip_outsiders_cols: list[str]
        Columns for which outliers are not removed
    
    Returns:
    -------
    predictions: numpy.array
        Predicted target values
    r2: float
        r2 score of the method
    mae: float
        Mean absolute error 
    mse: float
        Mean squared error
    """
    data = data.drop(cols_to_drop, axis=1)
    if outsiders_thresh:
        data = remove_outliers(data,
                             threshold=outsiders_thresh,
                             skip_columns=skip_outsiders_cols + [target])
    X_train, X_test, y_train, y_test = split_data(
        data, target, test_size, random_state)
    if scaler:
        scale_data(X_train, X_test, scaler)
    if encoders:
        X_train, X_test = encode_data(
            X_train, X_test, encoders, cols_to_encode)
    predictions = apply_model(X_train, X_test, y_train, model, return_formula, k)

    return (predictions,
            r2_score(y_test, predictions),
            mean_absolute_error(y_test, predictions),
            mean_squared_error(y_test, predictions, squared=False))


def score_classification_model(data: pd.DataFrame, 
                               target: str, 
                               model,
                               return_formula: bool = False,
                               cols_to_encode: list[str] = None,
                               scaler=None,
                               encoders=None,
                               cols_to_drop: list[str] = [],
                               test_size: float = 0.3, 
                               random_state: int = 42,
                               outsiders_thresh: float = None,
                               skip_outsiders_cols: list[str] = [],
                               k_range=None,
                               show_opt_plot=False,
                               balance_dataset=False,
                               balancer=None, 
                               ):
    """
    Scores a Classification Model, it assumes data is already cleaned

    Parameters:
    ----------
    data: pandas.DataFrame: 
        Dataset for the model
    target: str
        Name of target column
    model: Callable
        ML model for regression
    outsiders_thresh: float
        Threshold for the removal of outliers
    return_forumula: bool
        If True, returns the mathematical coefficients and intercept of a linear model
    cols_to_encode: list[str]
        List of columns in which apply the encoding
    scaler: Callable
        Class of scikit-learn for scaling the data
    encoders: list[Callable]
        List of encoders to apply to corresponding cols_to_encode
    cols_to_drop: list[str]
        List of columns' names to be dropped from the dataset
    test_size: float, default=0.3
            Size of the test subset
    random_state: int, default=42
        Controls the shuffling applied to the data before the split.
    outsiders_thresh: float
        Threshold for the removal of outliers
    skip_outsiders_cols: list[str]
        Columns for which outliers are not removed
    k_range: TODO: refactor (accept str or number)
        optional k if classifier is KN Clasisifier. if 'optimal', k is optimized
    show_opt_plot: bool, default=False
        If True and a KN classifier is used, plot the optimization algorithm
    balance_dataset: bool, default=True
        If True, dataset is balanced through defined balancer
    balancer: Callable
        Balance class from module imblearn

    Returns:
    -------
    y_test: numpy.array
        True target values
    predictions: numpy.array
        Predicted target values
    classification_report: str
        Report of the metrics for evaluation of the classification model
    """
    data = data.drop(cols_to_drop, axis=1)
    if outsiders_thresh:
        data = remove_outliers(data,
                             threshold=outsiders_thresh,
                             skip_columns=skip_outsiders_cols + [target])

    X_train, X_test, y_train, y_test = split_data(
        data, target, test_size, random_state)

    if scaler:
        scale_data(X_train, X_test, scaler)
        if len(X_train) != len(y_train):
            # Scaling has dropped some rows (log-transform). Uniform with target
            uniform_after_scaling(X_train, y_train, X_test, y_test)

    if encoders:
        X_train, X_test = encode_data(
            X_train, X_test, encoders, cols_to_encode)

    # balancing
    if balance_dataset:
        X_train, y_train = balancer.fit_resample(X_train, y_train)

    if k_range:  # TODO: when k is not a range?
        opt_k = knn_class_optimization(X_train, y_train, X_test, y_test,
                                       k_range, show_opt_plot)
        predictions = apply_model(X_train, X_test, y_train, model,
                                  return_formula, opt_k[0])
    else:
        predictions = apply_model(X_train, X_test, y_train, model,
                                  return_formula, None)  # TODO: to optimize

    return y_test, predictions, classification_report(y_test, predictions)


def knn_regr_optimization(X_train: np.array, y_train: np.array,
                          X_test: np.array, y_test: np.array,
                          metric,
                          k: list,
                          show_plot: bool):
    """
    Find a optimal k for the KNNregression algorithm
    TODO: refactor
    Parameters:
    X_train: np.array
        Data of training features
    X_test: np.array
        Data of test features
    y_train: np.array
        Data of train target
    y_test: np.array
        Data of test target
    metric: Callable
        Chosen metric for which k is studied 
    k: list
        List of chosen k for optimization
    show_plot: bool
        If True, plot the optimization algorithm
    """
    # Plot error rates
    rate = []  # list of metric calculations
    for i in range(1, max(k)):
        knn = KNeighborsRegressor(n_neighbors=i)
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)
        rate.append(metric(y_test, y_pred))

    if show_plot:
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, max(k)), rate, color='blue', linestyle='dashed',
                 marker='o', markerfacecolor='red', markersize=10)
        plt.title(metric.__name__ + ' vs. K Value')
        plt.xlabel('K')
        plt.ylabel(metric.__name__)

def knn_class_optimization(X_train: np.array, y_train: np.array,
                          X_test: np.array, y_test: np.array,
                          k: list,
                          show_plot: bool):
    """
    Find a optimal k for the KNN Classification algorithm
    TODO: refactor
    Parameters:
    X_train: np.array
        Data of training features
    X_test: np.array
        Data of test features
    y_train: np.array
        Data of train target
    y_test: np.array
        Data of test target
    metric: Callable
        Chosen metric for which k is studied 
    k: list
        List of chosen k for optimization
    show_plot: bool
        If True, plot the optimization algorithm
    
    Returns:
    -------
    k_opt: float
        Optimal k for the algorithm
    min(error_rate): float
        Minimum error rate for k_opt
    """
    # Plot error rates
    error_rate = []
    for i in range(1, max(k)):
        knn = KNeighborsClassifier(n_neighbors=i)
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)
        error_rate.append(np.mean(y_pred != y_test))

    if show_plot:
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, max(k)), error_rate, color='blue', linestyle='dashed',
                 marker='o', markerfacecolor='red', markersize=10)
        plt.title('Error Rate vs. K Value')
        plt.xlabel('K')
        plt.ylabel('Error Rate')
        print("Minimum error:-", min(error_rate),
              "at K =", error_rate.index(min(error_rate)))

    k_opt = error_rate.index(min(error_rate))
    return k_opt, min(error_rate)


def boxcox_transform(df):
    """
    Apply a boxcox transformation
    """
    numeric_cols = df.select_dtypes(np.number).columns
    _ci = {column: None for column in numeric_cols}
    for column in numeric_cols:
        df[column] = np.where(df[column] <= 0, np.NAN, df[column])
        df[column] = df[column].fillna(df[column].mean())
        transformed_data, ci = stats.boxcox(df[column])
        df[column] = transformed_data
        _ci[column] = [ci]
    return df, _ci


def log_transform(x):
    """
    Apply a logarithmic trandormation.
    Returns a NaN if x <= 0
    """
    if np.isfinite(np.log(x)):
        return np.log(x)
    else:
        return np.NAN