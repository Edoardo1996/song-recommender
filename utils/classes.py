"""
Custom classes for ML model Pipeline definition
"""

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer


class NaNHandler(BaseEstimator, TransformerMixin):
    def __init__(self) -> None:
        super().__init__()
    def fit(self, X, y=None):
        return self # nothing to do here
    def transform(self, X):
        # drop all nans
        return X.dropna()