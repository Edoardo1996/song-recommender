"""
Module for the creation of the songs' database, quierying the data from spotify 
API
"""
import warnings
import spotipy
import pandas as pd
from spotipy.oauth2 import SpotifyClientCredentials
from tqdm import tqdm

class SpotifyClient:
    """
    Simple class that accepts a CLIENT_ID and CLIENT_SECRET,
    they can be specified in constructor or set up manually"""

    def __init__(self, *args):
        if len(args) == 0:
            warnings.warn("No arguments passed to constructor, please add them \
                or set them manuallt", Warning)
        elif len(args) == 1:
            warnings.warn("Only one argument in constructor, \
                                value set to client_id", Warning)
            self.client_id = args[0]
        elif len(args) == 2:
            self.client_id = args[0]
            self.client_secret = args[1]
        else:
            raise Exception("Constructor has more than 2 arguments")

    def set_client_id(self, client_id):
        self.client_id = client_id

    def set_client_secret(self, client_secret):
        self.client_secret = client_secret

    def authorize(self, verbose=0):
        if not hasattr(self, "client_id"):
            raise Exception("client_id has not been set")
        elif not hasattr(self, "client_secret"):
            raise Exception("client_id has not been set")
        else:
            clients_credentials_manager = SpotifyClientCredentials(
                client_id=self.client_id,
                client_secret=self.client_secret
            )
            try:
                access = spotipy.Spotify(
                    client_credentials_manager=clients_credentials_manager)
                if verbose > 0:
                    print("Authorization Successful")
                return access
            except:
                raise Exception("Authorization Failes")


def retrieve_playlists(sp: spotipy.Spotify, creator: str, offset: int = None) -> list[str]:
    """
    Retrieve playlists ids created by a certain user
    """
    def generator():
        while True:
            yield

    playlists_ids = list()
    playlists = sp.user_playlists(creator)
    for _ in (pbar := tqdm(generator())):
        pbar.set_description("Extracting playlists ids")
        for i, playlist in enumerate(playlists['items']):
            playlists_ids.append(playlist['id'])
        if playlists['next']:
            playlists = sp.next(playlists)
        else:
            break

    return playlists_ids

def analyze_playlist(sp: spotipy.Spotify, creator, playlist_id) -> pd.DataFrame:
    
    # Create empty dataframe
    playlist_features_list = ["artist","album","track_name",  "track_id","danceability","energy","key","loudness","mode", "speechiness","instrumentalness","liveness","valence","tempo", "duration_ms","time_signature"]
    
    playlist_df = pd.DataFrame(columns = playlist_features_list)
    
    # Loop through every track in the playlist, extract features and append the features to the playlist dLf
    
    playlist = sp.user_playlist_tracks(creator, playlist_id)["items"]
    for track in playlist:
        # Create empty dict
        playlist_features = {}
        # Get metadata
        playlist_features["artist"] = track["track"]["album"]["artists"][0]["name"]
        playlist_features["album"] = track["track"]["album"]["name"]
        playlist_features["track_name"] = track["track"]["name"]
        playlist_features["track_id"] = track["track"]["id"]
        try:
            playlist_features["genres"] = sp.artist(track["track"]["album"]["artists"][0]["id"])["genres"][0] # TODO: add genre handling
        except:
            playlist_features["genres"] = 'unknown'
        
        # Get audio features
        audio_features = sp.audio_features(playlist_features["track_id"])[0]
        for feature in playlist_features_list[4:]:
            playlist_features[feature] = audio_features[feature]
        
        # Concat the dfs
        track_df = pd.DataFrame(playlist_features, index = [0])
        playlist_df = pd.concat([playlist_df, track_df], ignore_index = True)
        
    return playlist_df
