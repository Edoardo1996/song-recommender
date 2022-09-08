"""
Module handling the ML model
"""
from classes import NaNHandler, ColumnDropperTransformer, GenreModifier, DataConverter
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import pandas as pd
import numpy as np

def build_pipeline(data: pd.DataFrame):
    # Variables
    cols_to_convert = ['danceability', 'energy', 'key', 'loudness', 'mode', 
                       'speechiness', 'instrumentalness', 'liveness', 'valence',
                       'tempo', 'duration_ms', 'time_signature']
    cols_to_drop = ['playlist_id', 'artist', 'album', 'track_name']
    music_genres = ['rock', 'pop', 'metal', 'punk', 'folk', 'hip hop', 'edm', 'classical']
    num_attribs = data.select_dtypes(np.number).columns
    cat_attribs = ['genres']

    # Define column transormers
    nan_handler = NaNHandler()
    converter = DataConverter(cols_to_convert)
    cols_dropper = ColumnDropperTransformer(cols_to_drop)
    genre_modifier = GenreModifier(music_genres)

    cols_transformation = Pipeline([
        ('nan_handling', nan_handler),
        ('cols_dropping', cols_dropper),
        ('genre_modifier', genre_modifier),
    ])



    scale_encode = ColumnTransformer([
        ('num', StandardScaler(), num_attribs),
        ('cat', OneHotEncoder(), cat_attribs)
    ])

    full_pipeline = Pipeline([
        ('cols_transformation', cols_transformation),
        ('scale_encode', scale_encode)
    ])

    df1 = cols_transformation.fit_transform(df).genres.value_counts().__len__()
    df2 = scale_encode.fit_transform(df1)


