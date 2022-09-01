from utils.songdb import *
import utils.config as config
from tqdm import tqdm
import pandas as pd
import os


def main():
    # Set up authorization
    client = SpotifyClient(config.CLIENT_ID, config.CLIENT_SECRET)
    sp = client.authorize()

    # Retrieve all playlists ids
    playlists_ids = retrieve_playlists(sp, "spotify")

    # Get all songs' features from the playlists if database has not been created yet -TODO: handle db if it was already created
    # if os.path.exists(config.DB_PATH):
    #     songs = pd.read_csv(config.DB_PATH)
    # else:
    #     songs = pd.DataFrame()
    #     for id in (pbar := tqdm(playlists_ids[:2])):
    #         pbar.set_description("Retrieving song features")
    #         songs = pd.concat([songs, analyse_playlist(sp, 'spotify', id)], axis=0, ignore_index=True)
    #     # Clean the duplicates track
    #     songs = songs.set_index("track_id")
    #     songs = songs.drop_duplicates()
    #     songs.to_csv("songs.csv")

    # Get all songs' features from the playlists
    songs = pd.DataFrame()
    for id in (pbar := tqdm(playlists_ids[:2])):
        pbar.set_description("Retrieving song features")
        songs = pd.concat([songs, analyse_playlist(sp, 'spotify', id)], axis=0, ignore_index=True)

    # Clean the duplicates track and export 
    songs = songs.set_index("track_id")
    songs = songs.drop_duplicates()
    songs.to_csv(config.DB_PATH)

    # Retrieve artists ids
    artists_ids = retrieve_artists_id(sp, songs)
    print(len(artists_ids))



if __name__ == '__main__':
    main()