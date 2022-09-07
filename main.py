from utils.songdb import *
from utils import config, sktools, cleaning
from tqdm import tqdm
import pandas as pd
import os

def data_preparation(db_path):
    """
    Prepares data for ML model
    """
    data = cleaning.load_data(db_path, index_col='track_id')
    


def main():
    # Set up authorization
    auth_manager = SpotifyClientCredentials(
    client_id=config.CLIENT_ID,
    client_secret=config.CLIENT_SECRET
    )
    sp = spotipy.Spotify(auth_manager=auth_manager)
    # client = SpotifyClient(config.CLIENT_ID, config.CLIENT_SECRET)
    # sp = client.authorize()

    # Retrieve all playlists info
    if not os.path.exists(config.PLAYLISTS_IDS_PATH):   
        playlists = retrieve_playlists(sp, "spotify")
        playlists.to_csv(config.PLAYLISTS_IDS_PATH)
    else:
        print("Spotify playlists IDs already retrieved")
        playlists = pd.read_csv(config.PLAYLISTS_IDS_PATH, index_col=0)

    # Get all songs' features from the playlists and append them to csv file
    songs = pd.DataFrame() # Playlist songs
    if os.path.exists(config.DB_PATH):
        all_songs = pd.read_csv(config.DB_PATH, index_col='track_id') # All playlists songs
        retrieved_playlist_ids = all_songs['playlist_id'].unique()
    else:
        # first run init
        all_songs = pd.DataFrame()
        retrieved_playlist_ids = []
    # filter already retrieved playlist
    filtered_playlists_ids = playlists.drop(retrieved_playlist_ids).index
    for id in (pbar := tqdm(filtered_playlists_ids)):
        pbar.set_description("Retrieving song features")
        # Get playlist songs
        songs = analyse_playlist(sp, 'spotify', id)
        songs = prep_for_append(all_songs, songs, config.DB_PATH) # set index and remove duplicates song
        all_songs = pd.concat([all_songs, songs], axis=0)
        if os.path.exists(config.DB_PATH):
            songs.to_csv(config.DB_PATH, mode='a', index=True, header=False)
        else:
            songs.to_csv(config.DB_PATH)
        
    # Retrieve artists ids
    artists_ids = retrieve_artists_id(sp, all_songs)

    # Retrieve top 10 tracks ids for each artists
    top_10_tracks_ids = retrieve_top_10_tracks_ids(sp, artists_ids)
    top_10_tracks_ids_filtered = [song for song in top_10_tracks_ids if song not in list(all_songs.index)] # filter tracks ids

    # Enrich the database with the retrieved top 10 songs
    enriched_songs = enrich_songs(sp, top_10_tracks_ids_filtered, all_songs)
    enriched_songs.to_csv(config.ENRICHED_DB_PATH)

    # Clean retrieved dataset and prepare for ML model

    # Train the model and save it to pickle

if __name__ == '__main__':
    main()