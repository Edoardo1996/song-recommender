"""
Module for the creation of the songs' database, quierying the data from spotify 
API
"""
import warnings
import spotipy
import pandas as pd
from spotipy.oauth2 import SpotifyClientCredentials
from tqdm import tqdm
import itertools

class SpotifyClient:
    """
    Simple class that accepts a CLIENT_ID and CLIENT_SECRET,
    they can be specified in constructor or set up manually
    """

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


def retrieve_playlists(sp: spotipy.Spotify, creator: str, offset: int = None) -> pd.DataFrame:
    """
    Retrieve playlists ids created by a certain user
    """
    def generator():
        while True:
            yield

    playlists_ids = list()
    playlists_uris = list()
    playlists_names = list()
    playlists = sp.user_playlists(creator)
    for _ in (pbar := tqdm(generator())):
        pbar.set_description("Extracting playlists ids")
        for i, playlist in enumerate(playlists['items']):
            playlists_ids.append(playlist['id'])
            playlists_uris.append(playlist['uri'])
            playlists_names.append(playlist['name'])
        if playlists['next']:
            playlists = sp.next(playlists)
        else:
            break
    
    # export in a pd.Dataframe
    playlists_df = pd.DataFrame()
    playlists_df['name'] = playlists_names
    playlists_df['uri'] = playlists_uris
    playlists_df['id'] = playlists_ids
    playlists_df = playlists_df.set_index(keys='id', drop=True)


    return playlists_df

def analyse_playlist(sp: spotipy.Spotify, creator: str, playlist_id: str) -> pd.DataFrame:
    
    # Create empty dataframe
    playlist_features_list = ["artist","album","track_name",  "track_id","danceability","energy","key","loudness","mode", "speechiness","instrumentalness","liveness","valence","tempo", "duration_ms","time_signature"]
    
    playlist_df = pd.DataFrame(columns = playlist_features_list)
    
    # Loop through every track in the playlist, extract features and append the features to the playlist dLf
    
    playlist = sp.user_playlist_tracks(creator, playlist_id)["items"]
    for track in playlist:
        if track["track"]:
            # Create empty dict
            playlist_features = {}
            # Get metadata
            try:
                playlist_features["artist"] = track["track"]["album"]["artists"][0]["name"]
            except:
                playlist_features["artist"] = 'Unknown'
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
                try:
                    playlist_features[feature] = audio_features[feature]
                except:
                    playlist_features[feature] = 'Unknown'
            
            # Concat the dfs
            track_df = pd.DataFrame(playlist_features, index = [0])
            playlist_df = pd.concat([playlist_df, track_df], ignore_index = True)
            
    return playlist_df

def retrieve_artists_id(sp:spotipy.Spotify, songs: pd.DataFrame) -> list:
    """
    Retrieve artists id from a songs database saved as pd.DataFrame
    """
    tracks_ids = songs.index

    # Slice the list in order to bypass Spotify API limits
    sliced_tracks_ids = list(zip(*(iter(tracks_ids),) * 50))

    artist_ids = list()
    for i in (pbar := tqdm(range(len(sliced_tracks_ids)))):
        pbar.set_description("Retrieving artists ids")
        raw_tracks = sp.tracks(sliced_tracks_ids[i])["tracks"]
        raw_artists = [track.get("artists") for track in raw_tracks]
        artist_ids_raw = [[artist.get("id") for artist in raw_artist] for raw_artist in raw_artists]
        artist_ids.extend(itertools.chain(*artist_ids_raw))
    
    return list(set(artist_ids))


def retrieve_top_10_tracks_ids(sp: spotipy.Spotify, artists_ids):
    """
    Retrieves the top 10 songs of an artist or a list
    of artists
    """
    top_ten_tracks_ids = list()
    for artist_id in (pbar:= tqdm(artists_ids)):
        pbar.set_description("Retrieving top 10 songs")
        top_ten_tracks_ids.extend([track.get("id") for track in sp.artist_top_tracks(artist_id)["tracks"]])

    print(f'Retrieved {len(top_ten_tracks_ids)} top 10 songs')
    return list(set(top_ten_tracks_ids))

def enrich_songs(sp: spotipy.Spotify, top_10_tracks_ids_filtered, songs):
    """
    Enrich songs database with top 10 songs for every found artists
    """
        # Create empty dataframe
    playlist_features_list = ["artist","album","track_name",  "track_id","danceability","energy","key","loudness","mode", "speechiness","instrumentalness","liveness","valence","tempo", "duration_ms","time_signature"]
    
    enriching_playlist_df = pd.DataFrame(columns = playlist_features_list)
    
    # Loop through every track in the top_10_tracks, extract features and append the features to the playlist df

    for track_id in (pabr:= tqdm(top_10_tracks_ids_filtered)):
        pabr.set_description("Enriching songs")
        # Create empty dict
        playlist_features = {}
        # Get track info
        track = sp.track(track_id)
        # Get metadata
        playlist_features["artist"] = track["artists"][0]["name"]
        playlist_features["album"] = track["album"]["name"]
        playlist_features["track_name"] = track["name"]
        playlist_features["track_id"] = track["id"]
        try:
            playlist_features["genres"] = sp.artist(track["album"]["artists"][0]["id"])["genres"][0] # TODO: add genre handling
        except:
            playlist_features["genres"] = 'unknown'
        
        # Get audio features
        audio_features = sp.audio_features(track["id"])[0]
        for feature in playlist_features_list[4:]:
            playlist_features[feature] = audio_features[feature]
        
        # Concat the dfs
        track_df = pd.DataFrame(playlist_features, index = [0])
        enriching_playlist_df = pd.concat([enriching_playlist_df, track_df], ignore_index = True)

    # Concat with previous playlist
     # Clean the duplicates track and export 
    enriching_playlist_df = enriching_playlist_df.set_index("track_id")
    enriching_playlist_df = enriching_playlist_df.drop_duplicates()
    enriching_playlist_df = pd.concat([songs, enriching_playlist_df], axis=0)
    enriching_playlist_df = enriching_playlist_df.drop_duplicates()
    return enriching_playlist_df