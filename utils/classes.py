"""
Custom classes for ML model Pipeline definition
"""

from sklearn.base import BaseEstimator, TransformerMixin
from itertools import compress

class NaNHandler(BaseEstimator, TransformerMixin):
    def __init__(self) -> None:
        super().__init__()
    def fit(self, X, y=None):
        return self # nothing to do here
    def transform(self, X):
        # drop all nans
        return X.dropna()

class ColumnDropperTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, columns:list[str]):
        self.columns = columns
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X.drop(self.columns, axis=1)

def genre_transorm(sub_genre, all_genres):
    filt = [genre in sub_genre for genre in all_genres]
    if sub_genre == 'unknown' or sum(filt) == 0:
        return 'other'
    elif sum(filt) == 1:
        # genre found and unique
        return list(compress(sub_genres, filt))[0]
    else:
        # return mutliple genres (for now)
        return ' '.join(list(compress(sub_genres, filt)))

class GenreModifier(BaseEstimator, TransformerMixin):
    def __init__(self, sub_genres) -> None:
        self.sub_genres = sub_genres
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        X.apply
    