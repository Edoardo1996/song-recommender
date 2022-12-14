"""
Module handling the ML model
"""
from utils.classes import NaNHandler, ColumnDropperTransformer, GenreModifier, DataConverter
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.cluster import KMeans
from yellowbrick.cluster.elbow import kelbow_visualizer
from yellowbrick.cluster import intercluster_distance
import pandas as pd

def build_pipeline(data: pd.DataFrame) -> Pipeline:
    # Variables, TODO: generealize process
    cols_to_convert = ['danceability', 'energy', 'key', 'loudness', 'mode', 
                       'speechiness', 'instrumentalness', 'liveness', 'valence',
                       'tempo', 'duration_ms', 'time_signature']
    cols_to_drop = ['playlist_id', 'artist', 'album', 'track_name']
    music_genres = ['rock', 'pop', 'metal', 'punk', 'folk', 'hip hop', 'edm', 'classical']
    num_attribs = ['danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness',
       'instrumentalness', 'liveness', 'valence', 'tempo', 'duration_ms',
       'time_signature']
    cat_attribs = ['genres']

    # Define column transormers
    nan_handler = NaNHandler()
    converter = DataConverter(cols_to_convert)
    cols_dropper = ColumnDropperTransformer(cols_to_drop)
    genre_modifier = GenreModifier(music_genres)

    cols_transformation = Pipeline([
        ('nan_handling', nan_handler),
        ('cols_dropping', cols_dropper),
        ('converter', converter),
        ('genre_modifier', genre_modifier)
    ])

    # Define scaling and encoding
    scale_encode = ColumnTransformer([
        ('num', StandardScaler(), num_attribs),
        ('cat', OneHotEncoder(), cat_attribs)
    ])

    # Wrap up in full pipeline
    full_pipeline = Pipeline([
        ('cols_transformation', cols_transformation),
        ('scale_encode', scale_encode)
    ])

    return full_pipeline

def visualize_kmeans(data, krange=(2, 10)):
    """
    Generates some visualizations in order to find optimal k for
    a KMEANS algotithm
    """
    # kelbow_visualizer(KMeans(random_state=42),
    #                   data,
    #                   k=(2,10),
    #                   timings=False)

    intercluster_distance(KMeans(n_clusters=60, random_state=42),
                          data,
                          embedding='mds', 
                          random_state=42)


