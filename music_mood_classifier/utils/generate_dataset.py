"""
utils/generate_dataset.py
--------------------------
Generates a realistic synthetic Spotify-like dataset for training.
This is used when the real Kaggle dataset is not available.

Features generated:
- valence: musical positivity (0.0 to 1.0)
- energy: intensity and activity (0.0 to 1.0)
- tempo: beats per minute (60 to 200)
- danceability: how suitable for dancing (0.0 to 1.0)
- loudness: overall loudness in dB (-60 to 0)
- acousticness: acoustic confidence (0.0 to 1.0)
- instrumentalness: no vocals likelihood (0.0 to 1.0)
- speechiness: spoken word presence (0.0 to 1.0)
- liveness: audience presence (0.0 to 1.0)
"""

import numpy as np
import pandas as pd
import os

# Set random seed for reproducibility
np.random.seed(42)


def generate_mood_cluster(n_songs, mood):
    """
    Generate songs with feature distributions typical for each mood.

    Mood definitions:
    - happy:    high valence (0.6-1.0), medium-high energy (0.5-0.9)
    - sad:      low valence (0.0-0.4), low energy (0.1-0.5)
    - energetic: high energy (0.7-1.0), high tempo (130-200 BPM)
    - calm:     low energy (0.0-0.4), high acousticness (0.5-1.0)
    """

    songs = []

    for _ in range(n_songs):
        if mood == "happy":
            song = {
                "valence": np.random.uniform(0.6, 1.0),
                "energy": np.random.uniform(0.5, 0.9),
                "tempo": np.random.uniform(100, 160),
                "danceability": np.random.uniform(0.5, 0.95),
                "loudness": np.random.uniform(-10, -2),
                "acousticness": np.random.uniform(0.0, 0.4),
                "instrumentalness": np.random.uniform(0.0, 0.3),
                "speechiness": np.random.uniform(0.03, 0.2),
                "liveness": np.random.uniform(0.05, 0.4),
            }

        elif mood == "sad":
            song = {
                "valence": np.random.uniform(0.0, 0.4),
                "energy": np.random.uniform(0.1, 0.5),
                "tempo": np.random.uniform(60, 110),
                "danceability": np.random.uniform(0.1, 0.5),
                "loudness": np.random.uniform(-30, -10),
                "acousticness": np.random.uniform(0.3, 0.9),
                "instrumentalness": np.random.uniform(0.0, 0.5),
                "speechiness": np.random.uniform(0.02, 0.15),
                "liveness": np.random.uniform(0.05, 0.3),
            }

        elif mood == "energetic":
            song = {
                "valence": np.random.uniform(0.3, 0.9),
                "energy": np.random.uniform(0.7, 1.0),
                "tempo": np.random.uniform(130, 200),
                "danceability": np.random.uniform(0.5, 0.95),
                "loudness": np.random.uniform(-8, 0),
                "acousticness": np.random.uniform(0.0, 0.2),
                "instrumentalness": np.random.uniform(0.0, 0.4),
                "speechiness": np.random.uniform(0.03, 0.3),
                "liveness": np.random.uniform(0.05, 0.5),
            }

        elif mood == "calm":
            song = {
                "valence": np.random.uniform(0.2, 0.6),
                "energy": np.random.uniform(0.0, 0.4),
                "tempo": np.random.uniform(60, 110),
                "danceability": np.random.uniform(0.1, 0.5),
                "loudness": np.random.uniform(-35, -12),
                "acousticness": np.random.uniform(0.5, 1.0),
                "instrumentalness": np.random.uniform(0.1, 0.8),
                "speechiness": np.random.uniform(0.02, 0.1),
                "liveness": np.random.uniform(0.05, 0.3),
            }

        song["mood"] = mood
        songs.append(song)

    return songs


def generate_song_names(n, mood):
    """Generate realistic-sounding song and artist names."""

    happy_songs = [
        ("Walking on Sunshine", "Katrina"),
        ("Happy", "Pharrell Williams"),
        ("Can't Stop the Feeling", "Justin Timberlake"),
        ("Shake It Off", "Taylor Swift"),
        ("Uptown Funk", "Bruno Mars"),
        ("Good as Hell", "Lizzo"),
        ("Girls Just Want to Have Fun", "Cyndi Lauper"),
        ("Dancing Queen", "ABBA"),
        ("Don't Stop Me Now", "Queen"),
        ("I Gotta Feeling", "Black Eyed Peas"),
        ("Levitating", "Dua Lipa"),
        ("Blinding Lights", "The Weeknd"),
        ("Watermelon Sugar", "Harry Styles"),
        ("Dynamite", "BTS"),
        ("Peaches", "Justin Bieber"),
    ]

    sad_songs = [
        ("Someone Like You", "Adele"),
        ("The Night We Met", "Lord Huron"),
        ("Skinny Love", "Bon Iver"),
        ("Let Her Go", "Passenger"),
        ("Fix You", "Coldplay"),
        ("Liability", "Lorde"),
        ("Hurt", "Johnny Cash"),
        ("The Sound of Silence", "Simon & Garfunkel"),
        ("Mad World", "Gary Jules"),
        ("Tears in Heaven", "Eric Clapton"),
        ("Creep", "Radiohead"),
        ("Nothing Compares 2 U", "Sinead O'Connor"),
        ("All I Want", "Kodaline"),
        ("Everybody Hurts", "R.E.M."),
        ("Black", "Pearl Jam"),
    ]

    energetic_songs = [
        ("Eye of the Tiger", "Survivor"),
        ("Lose Yourself", "Eminem"),
        ("Jump", "Van Halen"),
        ("Sandstorm", "Darude"),
        ("Thunderstruck", "AC/DC"),
        ("Can't Hold Us", "Macklemore"),
        ("Till I Collapse", "Eminem"),
        ("Run the World", "Beyoncé"),
        ("Power", "Kanye West"),
        ("Stronger", "Kanye West"),
        ("Bangarang", "Skrillex"),
        ("Turn Down for What", "DJ Snake"),
        ("Radioactive", "Imagine Dragons"),
        ("Believer", "Imagine Dragons"),
        ("Kickstart My Heart", "Mötley Crüe"),
    ]

    calm_songs = [
        ("Clair de Lune", "Debussy"),
        ("Weightless", "Marconi Union"),
        ("River Flows in You", "Yiruma"),
        ("Holocene", "Bon Iver"),
        ("The Night Cafe", "Van Morrison"),
        ("Bloom", "The Paper Kites"),
        ("Re: Stacks", "Bon Iver"),
        ("First Breath After Coma", "Explosions in the Sky"),
        ("Gymnopédie No.1", "Erik Satie"),
        ("April", "Nils Frahm"),
        ("Experience", "Ludovico Einaudi"),
        ("Comptine d'un autre été", "Yann Tiersen"),
        ("Aqualung", "Jethro Tull"),
        ("The Rain Song", "Led Zeppelin"),
        ("Breathe", "Pink Floyd"),
    ]

    pool = {
        "happy": happy_songs,
        "sad": sad_songs,
        "energetic": energetic_songs,
        "calm": calm_songs,
    }[mood]

    names = []
    artists = []
    for i in range(n):
        # Cycle through real names then add variations
        base_idx = i % len(pool)
        base_name, base_artist = pool[base_idx]
        if i < len(pool):
            names.append(base_name)
            artists.append(base_artist)
        else:
            # Add numbered variations for extra songs
            names.append(f"{base_name} (Version {i // len(pool) + 1})")
            artists.append(base_artist)

    return names, artists


def generate_dataset(n_per_mood=500, save_path="data/spotify_songs.csv"):
    """
    Main function to generate and save the complete dataset.

    Args:
        n_per_mood: Number of songs to generate per mood (default 500)
        save_path: Where to save the CSV file
    """
    print("🎵 Generating Spotify-like music dataset...")

    all_songs = []
    moods = ["happy", "sad", "energetic", "calm"]

    for mood in moods:
        print(f"  ✓ Generating {n_per_mood} {mood} songs...")
        songs = generate_mood_cluster(n_per_mood, mood)
        names, artists = generate_song_names(n_per_mood, mood)

        for i, song in enumerate(songs):
            song["song_name"] = names[i]
            song["artist"] = artists[i]
            song["popularity"] = int(np.random.uniform(10, 100))
            song["language"] = np.random.choice(
                ["English", "English", "English", "Spanish", "French", "Korean"],
                p=[0.6, 0.1, 0.05, 0.1, 0.05, 0.1],
            )

        all_songs.extend(songs)

    # Create DataFrame
    df = pd.DataFrame(all_songs)

    # Reorder columns nicely
    cols = [
        "song_name",
        "artist",
        "valence",
        "energy",
        "tempo",
        "danceability",
        "loudness",
        "acousticness",
        "instrumentalness",
        "speechiness",
        "liveness",
        "popularity",
        "language",
        "mood",
    ]
    df = df[cols]

    # Add some intentional noise and missing values (realistic!)
    noise_mask = np.random.random(len(df)) < 0.02  # 2% missing
    df.loc[noise_mask, "acousticness"] = np.nan

    noise_mask2 = np.random.random(len(df)) < 0.01
    df.loc[noise_mask2, "instrumentalness"] = np.nan

    # Save to CSV
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
    df.to_csv(save_path, index=False)

    print(f"\n✅ Dataset saved to: {save_path}")
    print(f"   Total songs: {len(df)}")
    print(f"   Mood distribution:\n{df['mood'].value_counts().to_string()}")
    print(f"   Missing values: {df.isnull().sum().sum()}")

    return df


if __name__ == "__main__":
    df = generate_dataset()
    print("\nSample rows:")
    print(df.head(3).to_string())
