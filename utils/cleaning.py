"""
Module for loading and cleaning the dataset
"""
import os
import sys
import time

import pandas as pd

from . import config
from sklearn.impute import KNNImputer
from typing import Union

def load_data(filepath: str, index_col: Union[int, str] = None) -> pd.DataFrame:
    """
    Load an excel or a .csv file into a pandas.DataFrame

    Parameters: 
    -----------
    filepath: str
        Relative path to the data
    index_col: int, default=None
        Column index

    Returns
    -------
    pandas.DataFrame
        Data structured in a DataFrame
    """
    if filepath.endswith('.xlsx'):
        return pd.read_excel(filepath)
    else:
        return pd.read_csv(filepath, index_col=index_col)


def drop_nan(data: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
    """
    Drops all NaN in the dataset

    Parameters:
    ----------
    data: pandas.DataFrameç
        Dataset where NaN are present
    verbose: bool, default = False
        Parameter for verbose option.
        If True, it will print the numbers of row dropped

    Returns:
    -------
    data: pandas.DataFrame
        Dataset with dropped NaNs.
    """
    orig_size = len(data)
    data = data.dropna()
    if verbose:
        print('Dropped {} rows, {:.2f}% of original rows'. format(
            orig_size-len(data), (orig_size-len(data))/len(data)*100
        ))
    return data


def format_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    Format the dataset. Changes columns' names in python-style snake_case and 
    allows to change index column if proposed column is unique.

    Parameters:
    ----------
    data: pandas.DataFrame
        Dataset

    Returns:
    ----------
    data: pandas.DataFrame
        Dataset formatted
    """
    # Format columns' names
    data.columns = data.columns.str.replace(' ', '_').str.lower()
    data.columns = data.columns.str.replace('#_', '')
    # Change index if unique
    if data[config.INDEX_DF].is_unique:
        return data.set_index(config.INDEX_DF)


def dump_value_counts(path: str, data: pd.DataFrame, verbose: str = False):
    """
    Export value counts for chosen dataset in a .txt file for 
    further readability.

    Parameters:
    ----------
    path: str
        Relative path where values are stored
    data: pandas.DataFrame
        Dataset
    verbose: bool, default = False
        Parameter for verbose option.

    Returns:
    -------
    data: pandas.DataFrame
        Dataset with dropped NaNs.
    """
    if not os.path.isdir(path):
        os.mkdir(path)
    timestamp = time.time()

    orig_stdout = sys.stdout
    with open(f"{path}/value_counts-{timestamp}.txt", 'w', encoding='utf-8') as f:
        sys.stdout = f
        for col in data.columns:
            print(col)
            print(data[col].value_counts(), end='\n'*3)
        sys.stdout = orig_stdout

    if verbose:
        print('Value counts information has been printed to {}'.format(
            path
        ))
    f.close()

def impute_knn(data, target, n_features=10):
    # Find at least n num features for correlation
    data = data.copy()
    features = data.corr()[target].sort_values(ascending=False).head(n_features).index.tolist()
    # init imputer
    imputer = KNNImputer()
    imputed_data = imputer.fit_transform(data[features])
    # return imputed feature
    return imputed_data[target]