"""
utils/preprocessing.py
-----------------------
Handles all data loading, cleaning, and preparation steps.

Key Steps:
1. Load the CSV dataset
2. Handle missing values
3. Normalize/scale features using StandardScaler
4. Assign mood labels based on audio feature rules (if not already labeled)
5. Return clean data ready for model training
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import pickle
import os


# ─────────────────────────────────────────────
# Feature columns we use for ML
# ─────────────────────────────────────────────
FEATURE_COLS = [
    "valence",
    "energy",
    "tempo",
    "danceability",
    "loudness",
    "acousticness",
    "instrumentalness",
    "speechiness",
    "liveness",
]

# ─────────────────────────────────────────────
# Mood color palette (for visualizations)
# ─────────────────────────────────────────────
MOOD_COLORS = {
    "happy": "#FFD700",       # Gold
    "sad": "#4169E1",         # Royal Blue
    "energetic": "#FF4500",   # Orange Red
    "calm": "#2E8B57",        # Sea Green
}

MOOD_EMOJIS = {
    "happy": "happy",
    "sad": "sad",
    "energetic": "energetic",
    "calm": "calm",
}


def load_data(filepath="data/spotify_songs.csv"):
    """
    Load the dataset from CSV.

    Args:
        filepath: Path to the CSV file

    Returns:
        pandas DataFrame with raw data
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"Dataset not found at '{filepath}'.\n"
            "Please run: python utils/generate_dataset.py\n"
            "Or download the Kaggle Spotify dataset and place it in /data/"
        )

    df = pd.read_csv(filepath)
    print(f" Loaded {len(df)} songs from '{filepath}'")
    print(f"   Columns: {list(df.columns)}")
    return df


def handle_missing_values(df):
    """
    Clean the dataset by handling missing/null values.

    Strategy:
    - Numeric features → fill with median (robust to outliers)
    - Categorical features → fill with mode (most common value)

    Args:
        df: Raw DataFrame

    Returns:
        Cleaned DataFrame
    """
    missing_before = df.isnull().sum().sum()

    if missing_before == 0:
        print(" No missing values found!")
        return df

    print(f"  Found {missing_before} missing values. Cleaning...")

    # Fill numeric columns with median
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].isnull().any():
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            print(f"   '{col}': filled {df[col].isnull().sum()} nulls with median={median_val:.3f}")

    # Fill categorical columns with mode
    categorical_cols = df.select_dtypes(include=["object"]).columns
    for col in categorical_cols:
        if df[col].isnull().any():
            mode_val = df[col].mode()[0]
            df[col] = df[col].fillna(mode_val)

    missing_after = df.isnull().sum().sum()
    print(f" Missing values after cleaning: {missing_after}")

    return df


def assign_mood_labels(df):
    """
    Assign mood labels to songs based on audio feature rules.
    This is used ONLY if the dataset doesn't already have a 'mood' column.

    Mood Classification Logic:
    ──────────────────────────
    We use a rule-based approach inspired by music psychology research:

    HAPPY:     High valence (positive/happy sound) + medium-high energy
               → Think pop, disco, feel-good music

    SAD:       Low valence (negative/sad sound) + low energy
               → Think ballads, blues, slow songs

    ENERGETIC: High energy + high tempo (fast BPM)
               → Think EDM, rock, workout music

    CALM:      Low energy + high acousticness (acoustic instruments)
               → Think classical, ambient, lo-fi music

    Note: We normalize features first so thresholds are percentile-based.

    Args:
        df: DataFrame with audio features

    Returns:
        DataFrame with 'mood' column added
    """
    if "mood" in df.columns:
        print(" Dataset already has 'mood' labels. Skipping assignment.")
        return df

    print("  Assigning mood labels based on audio features...")

    # Use median as threshold (50th percentile)
    val_med = df["valence"].median()
    eng_med = df["energy"].median()
    tempo_med = df["tempo"].median()
    acoust_med = df["acousticness"].median()

    def classify_mood(row):
        high_val = row["valence"] >= val_med
        high_eng = row["energy"] >= eng_med
        high_tempo = row["tempo"] >= tempo_med
        high_acoust = row["acousticness"] >= acoust_med

        # Priority order matters here!
        if high_eng and high_tempo:
            return "energetic"
        elif high_val and high_eng:
            return "happy"
        elif not high_val and not high_eng:
            return "sad"
        elif not high_eng and high_acoust:
            return "calm"
        elif high_val:
            return "happy"
        else:
            return "calm"

    df["mood"] = df.apply(classify_mood, axis=1)

    print(f" Mood distribution:\n{df['mood'].value_counts().to_string()}")
    return df


def normalize_features(df, scaler=None, fit=True):
    """
    Normalize features using StandardScaler.

    Why normalize?
    → KNN uses Euclidean distance, so features on different scales
      (e.g., tempo in 60-200 vs valence in 0-1) would bias results.
    → After scaling, all features have mean=0 and std=1.

    Args:
        df: DataFrame with feature columns
        scaler: Pre-fitted scaler (used during inference/prediction)
        fit: If True, fit a new scaler; if False, use provided scaler

    Returns:
        (scaled_features_array, scaler)
    """
    features = df[FEATURE_COLS].values

    if fit or scaler is None:
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        print(" Features normalized using StandardScaler")
        print(f"   Features used: {FEATURE_COLS}")
    else:
        features_scaled = scaler.transform(features)

    return features_scaled, scaler


def prepare_data(filepath="data/spotify_songs.csv"):
    """
    Complete preprocessing pipeline.

    This is the main function that ties everything together.
    Returns everything needed for model training.

    Args:
        filepath: Path to dataset CSV

    Returns:
        dict with:
          - df: cleaned DataFrame
          - X: scaled feature matrix
          - y: mood labels array
          - scaler: fitted StandardScaler
          - feature_cols: list of feature names used
    """
    print("\n" + "="*50)
    print(" DATA PREPROCESSING PIPELINE")
    print("="*50)

    # Step 1: Load
    df = load_data(filepath)

    # Step 2: Clean
    df = handle_missing_values(df)

    # Step 3: Assign labels (if needed)
    df = assign_mood_labels(df)

    # Step 4: Normalize features
    X_scaled, scaler = normalize_features(df)

    # Step 5: Extract labels
    y = np.array(df["mood"].tolist())  # Convert ArrowStringArray to plain numpy

    print("\n Preprocessing complete!")
    print(f"   X shape: {X_scaled.shape}")
    print(f"   y shape: {y.shape}")
    print(f"   Classes: {np.unique(y)}")
    print("="*50 + "\n")

    return {
        "df": df,
        "X": X_scaled,
        "y": y,
        "scaler": scaler,
        "feature_cols": FEATURE_COLS,
    }


def save_scaler(scaler, path="model/scaler.pkl"):
    """Save the fitted scaler for later use."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(scaler, f)
    print(f" Scaler saved to {path}")


def load_scaler(path="model/scaler.pkl"):
    """Load a previously saved scaler."""
    with open(path, "rb") as f:
        scaler = pickle.load(f)
    print(f" Scaler loaded from {path}")
    return scaler
