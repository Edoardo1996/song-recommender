# song-recommender
### How the songs database is built
First of all, we want to build a database suitable for the most generic customer possible, <br>
because no specification of geography or customer profile has been provided. For this <br>
reason, all official playlists created by the account "spotify" are analysed in the following <br>
way:
1. Get playlists ids of every Spotify playlist now active;
2. Retrieve song data for every track that is in those playlist;
3. Build a first, not-enriched, database of all this songs (ensuring no repetitions)
4. Get all the artists from this database and look for each artist top-10 songs;
5. Enrich the previuos database with additional data of these top-10 tracks;