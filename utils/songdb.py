"""
Module for the creation of the songs' database, quierying the data from spotify 
API
"""
import warnings
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials


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


def retrieve_playlists(sp: spotipy.Spotify, user: str, offset: int = None) -> list[str]:
    """
    Retrieve playlists ids created by a certain user
    """
    playlists_ids = list()
    playlists = sp.user_playlists(user)
    while playlists:
        for i, playlist in enumerate(playlists['items']):
            playlists_ids.append(playlist['id'])
        if playlists['next']:
            playlists = sp.next(playlists)
        else:
            playlists = None

    return playlists_ids
