"""
notebooks/explore.py
---------------------
Interactive exploration script — run section by section to understand the project.
This replaces a Jupyter notebook with a plain Python script that's easy to run.

Run any section independently:
    python notebooks/explore.py

Or copy individual sections into a Jupyter notebook for interactive use.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

from utils.preprocessing import prepare_data, FEATURE_COLS, MOOD_COLORS
from model.knn_model import (
    train_classifier, build_recommender, recommend_songs,
    find_song_by_name, predict_mood
)

print("=" * 60)
print("🎵 MUSIC MOOD CLASSIFIER — INTERACTIVE EXPLORATION")
print("=" * 60)

# ─────────────────────────────────────────────────────────────
# SECTION 1: Load and Inspect Data
# ─────────────────────────────────────────────────────────────
print("\n📂 SECTION 1: Loading Data")
print("-" * 40)

data = prepare_data("data/spotify_songs.csv")
df = data["df"]
X = data["X"]
y = data["y"]
scaler = data["scaler"]

print("\nFirst 5 rows:")
print(df[["song_name", "artist", "valence", "energy", "tempo", "mood"]].head())

print("\nMood distribution:")
print(df["mood"].value_counts())

print("\nFeature statistics:")
print(df[FEATURE_COLS].describe().round(3))

# ─────────────────────────────────────────────────────────────
# SECTION 2: Understanding Feature Differences by Mood
# ─────────────────────────────────────────────────────────────
print("\n📊 SECTION 2: Feature Averages by Mood")
print("-" * 40)

mood_avg = df.groupby("mood")[["valence", "energy", "tempo", "acousticness", "danceability"]].mean()
print(mood_avg.round(3))

print("""
Observations:
  😊 Happy:     High valence (0.8), medium energy (0.7)
  😢 Sad:       Low valence (0.2), low energy (0.3), high acousticness
  ⚡ Energetic: High energy (0.85), high tempo (~165 BPM)
  🌿 Calm:      Low energy (0.2), very high acousticness (0.75)
""")

# ─────────────────────────────────────────────────────────────
# SECTION 3: Train KNN Step by Step
# ─────────────────────────────────────────────────────────────
print("\n🤖 SECTION 3: Training KNN — Step by Step")
print("-" * 40)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Training samples: {len(X_train)}")
print(f"Testing samples:  {len(X_test)}")

# Train KNN
print("\nTraining KNN with K=5...")
knn = KNeighborsClassifier(n_neighbors=5, metric="euclidean", weights="distance")
knn.fit(X_train, y_train)

# Evaluate
y_pred = knn.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Accuracy with K=5: {acc*100:.2f}%")

# ─────────────────────────────────────────────────────────────
# SECTION 4: Predict Mood for a Custom Song
# ─────────────────────────────────────────────────────────────
print("\n🎵 SECTION 4: Predict Mood for Custom Features")
print("-" * 40)

# Define a "workout" song manually
workout_song = {
    "valence": 0.75,        # Positive
    "energy": 0.92,         # Very high energy
    "tempo": 155.0,         # Fast
    "danceability": 0.80,
    "loudness": -4.0,
    "acousticness": 0.02,
    "instrumentalness": 0.10,
    "speechiness": 0.08,
    "liveness": 0.15,
}

features = np.array([[workout_song[f] for f in FEATURE_COLS]])
features_scaled = scaler.transform(features)

mood_pred = knn.predict(features_scaled)[0]
proba = knn.predict_proba(features_scaled)[0]

print(f"Custom song features: energy={workout_song['energy']}, tempo={workout_song['tempo']} BPM")
print(f"Predicted mood: {mood_pred.upper()}")
print("\nProbabilities:")
for cls, p in zip(knn.classes_, proba):
    bar = "█" * int(p * 30)
    print(f"  {cls:10s}: {bar} {p*100:.1f}%")

# ─────────────────────────────────────────────────────────────
# SECTION 5: Playlist Recommendations
# ─────────────────────────────────────────────────────────────
print("\n🎶 SECTION 5: Generating Playlists")
print("-" * 40)

# Load saved recommender
import pickle
with open("model/knn_recommender.pkl", "rb") as f:
    recommender = pickle.load(f)

for mood in ["happy", "energetic", "calm", "sad"]:
    emoji = {"happy": "😊", "sad": "😢", "energetic": "⚡", "calm": "🌿"}[mood]
    print(f"\n{emoji} {mood.upper()} playlist:")

    recs = recommend_songs(mood, df, X, recommender, k=3)
    for i, (_, row) in enumerate(recs.iterrows(), 1):
        print(f"  {i}. {row['song_name']} — {row['artist']}")

# ─────────────────────────────────────────────────────────────
# SECTION 6: Song Search
# ─────────────────────────────────────────────────────────────
print("\n🔍 SECTION 6: Song Search")
print("-" * 40)

query = "Sandstorm"
results = find_song_by_name(query, df)
if len(results) > 0:
    song = results.iloc[0]
    print(f"Found: {song['song_name']} by {song['artist']}")
    print(f"Mood: {song['mood']} | Energy: {song['energy']:.2f} | Tempo: {song['tempo']:.0f} BPM")

    # Get similar songs
    song_idx = results.index[0]
    query_vec = X[song_idx].reshape(1, -1)
    similar = recommend_songs(query_vec, df, X, recommender, k=3)
    print("\nSimilar songs:")
    for _, row in similar.iterrows():
        print(f"  → {row['song_name']} ({row['mood']})")
else:
    print(f"No songs found matching '{query}'")

# ─────────────────────────────────────────────────────────────
# SECTION 7: Quick Visualization
# ─────────────────────────────────────────────────────────────
print("\n📊 SECTION 7: Quick Visualization")
print("-" * 40)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Plot 1: Valence vs Energy colored by mood
ax = axes[0]
for mood in ["happy", "sad", "energetic", "calm"]:
    mask = df["mood"] == mood
    ax.scatter(df[mask]["valence"], df[mask]["energy"],
               c=MOOD_COLORS[mood], label=mood, alpha=0.5, s=20)

ax.set_xlabel("Valence (positivity)")
ax.set_ylabel("Energy")
ax.set_title("Valence vs Energy by Mood")
ax.legend()

# Plot 2: Tempo distribution by mood
ax2 = axes[1]
for mood in ["happy", "sad", "energetic", "calm"]:
    data_mood = df[df["mood"] == mood]["tempo"]
    ax2.hist(data_mood, bins=20, alpha=0.5, label=mood,
             color=MOOD_COLORS[mood])

ax2.set_xlabel("Tempo (BPM)")
ax2.set_ylabel("Count")
ax2.set_title("Tempo Distribution by Mood")
ax2.legend()

plt.tight_layout()
plt.savefig("screenshots/exploration_plots.png", dpi=120, bbox_inches="tight")
print("✅ Saved exploration_plots.png")
plt.close()

print("\n" + "="*60)
print("✅ Exploration complete! Check screenshots/ for plots.")
print("="*60)
