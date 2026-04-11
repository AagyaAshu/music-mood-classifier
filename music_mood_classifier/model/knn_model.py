"""
model/knn_model.py
-------------------
KNN Mood Classifier + Playlist Generator

This module contains two KNN models:

1. KNeighborsClassifier → Predicts mood from song features
2. NearestNeighbors    → Finds K similar songs (playlist generator)

 What is KNN? (Simple Explanation)
──────────────────────────────────────
KNN = K-Nearest Neighbors

Imagine you walk into a room full of people.
To guess your mood, KNN looks at the K people closest to you
(based on similar features like energy, valence, tempo).
Whatever mood most of those K people have → that's your predicted mood!

For songs:
- Each song is a point in multi-dimensional space (9 features)
- KNN finds the K closest songs and votes on the mood
- Distance = Euclidean distance (like a ruler in 9D space)
"""

import numpy as np
import pickle
import os
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# ─────────────────────────────────────────────
# MODEL PATHS
# ─────────────────────────────────────────────
CLASSIFIER_PATH = "model/knn_classifier.pkl"
RECOMMENDER_PATH = "model/knn_recommender.pkl"


def train_classifier(X, y, n_neighbors=5, test_size=0.2, random_state=42):
    """
    Train the KNN mood classifier.

    Args:
        X: Scaled feature matrix (numpy array)
        y: Mood labels array
        n_neighbors: K value for KNN (how many neighbors to consider)
        test_size: Fraction of data for testing (0.2 = 20%)
        random_state: Random seed for reproducibility

    Returns:
        dict with model, predictions, and evaluation metrics
    """
    print("\n" + "=" * 50)
    print(" TRAINING KNN MOOD CLASSIFIER")
    print("=" * 50)
    print(f"   K (neighbors): {n_neighbors}")
    print(f"   Test size: {test_size * 100:.0f}%")

    # ── Step 1: Split data into training and testing sets
    # We train on 80%, test on 20% (unseen data)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    print(f"\n   Training samples: {len(X_train)}")
    print(f"   Testing samples:  {len(X_test)}")

    # ── Step 2: Create and train the KNN classifier
    classifier = KNeighborsClassifier(
        n_neighbors=n_neighbors,
        metric="euclidean",  # Distance formula: sqrt((x1-x2)² + (y1-y2)² + ...)
        weights="distance",  # Closer neighbors have more voting power
        algorithm="auto",  # Automatically chooses fastest algorithm
        n_jobs=-1,  # Use all CPU cores
    )

    print("\n   Training model...")
    classifier.fit(X_train, y_train)
    print("    Training complete!")

    # ── Step 3: Make predictions on test set
    y_pred = classifier.predict(X_test)

    # ── Step 4: Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred, labels=["happy", "sad", "energetic", "calm"])
    report = classification_report(y_test, y_pred, output_dict=True)
    report_str = classification_report(y_test, y_pred)

    print("\n EVALUATION RESULTS:")
    print(f"   Accuracy: {accuracy * 100:.2f}%")
    print(f"\n{report_str}")

    return {
        "model": classifier,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "y_pred": y_pred,
        "accuracy": accuracy,
        "confusion_matrix": cm,
        "classification_report": report,
        "classification_report_str": report_str,
    }


def build_recommender(X, df, n_neighbors=10):
    """
    Build the KNN-based playlist recommender.

    This is different from the classifier!
    Instead of predicting a class, it finds the K most similar songs
    using Euclidean distance in feature space.

    Think of it like: "Find me songs that SOUND similar to this one"

    Args:
        X: Scaled feature matrix
        df: Original DataFrame (to get song names later)
        n_neighbors: Max neighbors to consider

    Returns:
        Fitted NearestNeighbors model
    """
    print("\n" + "=" * 50)
    print(" BUILDING PLAYLIST RECOMMENDER")
    print("=" * 50)

    recommender = NearestNeighbors(
        n_neighbors=n_neighbors + 1,  # +1 because it includes the query song itself
        metric="euclidean",
        algorithm="ball_tree",  # Efficient for medium datasets
        n_jobs=-1,
    )

    recommender.fit(X)
    print(f"    Recommender trained on {len(X)} songs")
    print(f"   Max recommendations: {n_neighbors}")

    return recommender


def predict_mood(features_scaled, classifier):
    """
    Predict mood for a single song or batch of songs.

    Args:
        features_scaled: Scaled feature vector(s)
        classifier: Trained KNN classifier

    Returns:
        Predicted mood(s) and probabilities
    """
    if features_scaled.ndim == 1:
        features_scaled = features_scaled.reshape(1, -1)

    mood = classifier.predict(features_scaled)[0]
    probabilities = classifier.predict_proba(features_scaled)[0]
    classes = classifier.classes_

    prob_dict = {cls: float(prob) for cls, prob in zip(classes, probabilities)}

    return mood, prob_dict


def recommend_songs(
    mood_or_features,
    df,
    X_scaled,
    recommender,
    classifier=None,
    scaler=None,
    k=5,
    mood_filter=None,
    lang_filter=None,
    min_popularity=0,
):
    """
    Generate playlist recommendations.

    Can work in two modes:
    1. Mood-based: Input is a mood string → finds songs of that mood
    2. Song-based: Input is feature vector → finds similar songs

    Args:
        mood_or_features: mood string ("happy") OR feature array
        df: DataFrame with song info
        X_scaled: Scaled feature matrix (all songs)
        recommender: Fitted NearestNeighbors model
        classifier: KNN classifier (needed for mood-based)
        scaler: Fitted scaler
        k: Number of songs to recommend
        mood_filter: Filter results by specific mood
        lang_filter: Filter by language (e.g., "English")
        min_popularity: Minimum popularity score (0-100)

    Returns:
        DataFrame with recommended songs
    """

    if isinstance(mood_or_features, str):
        # ── MOOD-BASED RECOMMENDATION ──
        # 1. Find all songs of the requested mood
        # 2. Pick a random one as "anchor"
        # 3. Find K nearest neighbors to that anchor
        mood = mood_or_features.lower()
        mood_mask = df["mood"] == mood

        if not mood_mask.any():
            raise ValueError(f"No songs found with mood: {mood}")

        # Pick representative song (highest popularity in this mood)
        mood_songs = df[mood_mask]
        anchor_idx = mood_songs["popularity"].idxmax()
        query_features = X_scaled[anchor_idx].reshape(1, -1)

    else:
        # ── FEATURE-BASED RECOMMENDATION ──
        query_features = mood_or_features.reshape(1, -1)
        anchor_idx = None

    # Find K nearest neighbors
    distances, indices = recommender.kneighbors(query_features)

    # Flatten arrays
    distances = distances[0]
    indices = indices[0]

    # Remove the anchor song itself (distance = 0)
    if anchor_idx is not None:
        mask = indices != anchor_idx
        distances = distances[mask]
        indices = indices[mask]

    # Get recommended songs
    recommended = df.iloc[indices].copy()
    recommended["similarity_score"] = 1 / (
        1 + distances[: len(recommended)]
    )  # Convert distance to similarity
    recommended["distance"] = distances[: len(recommended)]

    # ── Apply Filters ──
    if mood_filter:
        recommended = recommended[recommended["mood"] == mood_filter]

    if lang_filter and lang_filter != "All":
        recommended = recommended[recommended["language"] == lang_filter]

    if min_popularity > 0:
        recommended = recommended[recommended["popularity"] >= min_popularity]

    # Return top K results
    recommended = recommended.head(k)

    return recommended


def find_song_by_name(song_name, df):
    """
    Search for a song in the dataset by name (partial match).

    Args:
        song_name: Song name to search for
        df: DataFrame

    Returns:
        Matching rows (DataFrame)
    """
    mask = df["song_name"].str.lower().str.contains(song_name.lower(), na=False)
    return df[mask]


def save_models(
    classifier, recommender, clf_path=CLASSIFIER_PATH, rec_path=RECOMMENDER_PATH
):
    """Save trained models to disk."""
    os.makedirs(os.path.dirname(clf_path), exist_ok=True)

    with open(clf_path, "wb") as f:
        pickle.dump(classifier, f)
    print(f" Classifier saved to {clf_path}")

    with open(rec_path, "wb") as f:
        pickle.dump(recommender, f)
    print(f" Recommender saved to {rec_path}")


def load_models(clf_path=CLASSIFIER_PATH, rec_path=RECOMMENDER_PATH):
    """Load pre-trained models from disk."""
    with open(clf_path, "rb") as f:
        classifier = pickle.load(f)
    print(f" Classifier loaded from {clf_path}")

    with open(rec_path, "rb") as f:
        recommender = pickle.load(f)
    print(f" Recommender loaded from {rec_path}")

    return classifier, recommender


def find_best_k(X_train, y_train, X_test, y_test, k_range=range(3, 21)):
    """
    Find the optimal K value by testing multiple K values.

    Args:
        k_range: Range of K values to test

    Returns:
        (best_k, accuracies_dict)
    """
    print("\n Finding optimal K value...")
    accuracies = {}

    for k in k_range:
        knn = KNeighborsClassifier(
            n_neighbors=k, metric="euclidean", weights="distance"
        )
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        accuracies[k] = acc
        print(f"   K={k:2d}: Accuracy = {acc * 100:.2f}%")

    best_k = max(accuracies, key=accuracies.get)
    print(f"\n Best K = {best_k} (Accuracy: {accuracies[best_k] * 100:.2f}%)")

    return best_k, accuracies
