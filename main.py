from utils.songdb import SpotifyClient, retrieve_playlists
import utils.config as config


def main():
    # Set up authorization
    client = SpotifyClient(config.CLIENT_ID, config.CLIENT_SECRET)
    sp = client.authorize()

    # Retrieve all playlists ids
    playlists_ids = retrieve_playlists(sp, "spotify")

    
    



if __name__ == '__main__':
    main()