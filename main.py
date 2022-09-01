from utils.songdb import SpotifyClient, retrieve_playlists
import utils.config as config
import tqdm
import pandas as pd
import os


def main():
    # Set up authorization
    client = SpotifyClient(config.CLIENT_ID, config.CLIENT_SECRET)
    sp = client.authorize()

    # Retrieve all playlists ids
    playlists_ids = retrieve_playlists(sp, "spotify")

    # Get all songs from the playlists
    # if os.path.exists(config.DB_PATH):
    #     songs = pd.read_csv(config.DB_PATH)
    # else:
    #     songs = pd.DataFrame()
    #     for id in tqdm(playlists_ids[:5]):
    #         songs = pd.concat([songs, analyze_playlist('spotify', id)], axis=0, ignore_index=True)
    #     # Clean the duplicates track
    #     songs = songs.set_index("track_id")
    #     songs = songs.drop_duplicates()
    #     songs.to_csv("songs.csv")

    # Retrieve song features


    



if __name__ == '__main__':
    main()