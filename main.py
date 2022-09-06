from operator import index
from utils.songdb import *
import utils.config as config
from tqdm import tqdm
import pandas as pd
import os


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

    # Get all songs' features from the playlists
    songs = pd.DataFrame()
    for id in (pbar := tqdm(playlists.index)):
        pbar.set_description("Retrieving song features")
        songs = pd.concat([songs, analyse_playlist(sp, 'spotify', id)], axis=0, ignore_index=True)

    # Clean the duplicates track and export 
    songs = songs.set_index("track_id")
    songs = songs.drop_duplicates()
    songs.to_csv(config.DB_PATH)

    # Retrieve artists ids
    artists_ids = retrieve_artists_id(sp, songs)

    # Retrieve top 10 tracks ids for each artists
    top_10_tracks_ids = retrieve_top_10_tracks_ids(sp, artists_ids)
    top_10_tracks_ids_filtered = [song for song in top_10_tracks_ids if song not in list(songs.index)] # filter tracks ids

    # Enrich the database with the retrieved top 10 songs
    enriched_songs = enrich_songs(sp, top_10_tracks_ids_filtered, songs)
    enriched_songs.to_csv(config.ENRICHED_DB_PATH)


if __name__ == '__main__':
    main()