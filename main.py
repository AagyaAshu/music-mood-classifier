"""
main.py
--------
Entry point for the Music Mood Classifier project.

This script:
1. Generates / loads the dataset
2. Preprocesses data
3. Trains the KNN classifier
4. Builds the playlist recommender
5. Evaluates performance
6. Generates all visualizations
7. Saves trained models

Run this FIRST before launching the web app!

Usage:
    python main.py
    python main.py --k 7          # Custom K value
    python main.py --songs 1000   # Custom dataset size
    python main.py --no-plots     # Skip visualizations
"""

import argparse
import os
import sys
import numpy as np

# ─── Add project root to Python path ───
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.generate_dataset import generate_dataset
from utils.preprocessing import prepare_data, save_scaler
from model.knn_model import (
    train_classifier,
    build_recommender,
    save_models,
    find_best_k,
    recommend_songs,
)
from utils.visualizations import generate_all_plots


def print_banner():
    """Simple startup banner."""
    print("\n" + "=" * 40)
    print("MUSIC MOOD CLASSIFIER & PLAYLIST GENERATOR")
    print("Powered by KNN")
    print("=" * 40 + "\n")
    print("-" * 25 + "\n")


def parse_args():
    parser = argparse.ArgumentParser(description="Train Music Mood Classifier")
    parser.add_argument(
        "--k",
        type=int,
        default=None,
        help="K value for KNN (auto-detect if not specified)",
    )
    parser.add_argument(
        "--songs",
        type=int,
        default=500,
        help="Songs per mood for dataset generation (default: 500)",
    )
    parser.add_argument(
        "--no-plots", action="store_true", help="Skip generating visualizations"
    )
    parser.add_argument(
        "--optimize-k",
        action="store_true",
        help="Find optimal K by testing K=3 to K=20",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="data/spotify_songs.csv",
        help="Path to dataset CSV",
    )
    return parser.parse_args()


def demo_recommendations(df, X_scaled, recommender, classifier):
    """
    Show sample recommendations for each mood to verify everything works.
    """
    print("\n" + "=" * 60)
    print(" DEMO: PLAYLIST RECOMMENDATIONS")
    print("=" * 60)

    moods = ["happy", "sad", "energetic", "calm"]
    emojis = {"happy": "happy", "sad": "sad", "energetic": "energetic", "calm": "calm"}

    for mood in moods:
        print(f"\n{emojis[mood]} Top 3 songs for '{mood}' mood:")
        print("-" * 40)

        try:
            recommendations = recommend_songs(
                mood_or_features=mood,
                df=df,
                X_scaled=X_scaled,
                recommender=recommender,
                k=3,
            )

            for i, (_, row) in enumerate(recommendations.iterrows(), 1):
                print(f"  {i}. {row['song_name']} - {row['artist']}")
                print(
                    f"     Mood: {row['mood']} | Popularity: {row.get('popularity', 'N/A')}"
                )
                print(
                    f"     Valence: {row['valence']:.2f} | Energy: {row['energy']:.2f} | "
                    f"Tempo: {row['tempo']:.0f} BPM"
                )

        except Exception as e:
            print(f"   Error: {e}")

    print("\n" + "=" * 60)


def main():
    print_banner()
    args = parse_args()

    # ─────────────────────────────────────
    # STEP 1: Generate/Load Dataset
    # ─────────────────────────────────────
    print("STEP 1: DATASET SETUP")
    print("-" * 40)

    if not os.path.exists(args.data_path):
        print(f"Dataset not found at '{args.data_path}'")
        print(f"Generating synthetic dataset with {args.songs} songs per mood...")
        generate_dataset(n_per_mood=args.songs, save_path=args.data_path)
    else:
        print(f" Found existing dataset at '{args.data_path}'")

    # ─────────────────────────────────────
    # STEP 2: Preprocess Data
    # ─────────────────────────────────────
    print("\nSTEP 2: DATA PREPROCESSING")
    print("-" * 40)

    data = prepare_data(filepath=args.data_path)
    df = data["df"]
    X = data["X"]
    y = data["y"]
    scaler = data["scaler"]

    # ─────────────────────────────────────
    # STEP 3: Find Best K (Optional)
    # ─────────────────────────────────────
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    if args.optimize_k or args.k is None:
        print("\nSTEP 3: FINDING OPTIMAL K")
        print("-" * 40)
        k_range = range(3, 21)
        best_k, k_accuracies = find_best_k(X_train, y_train, X_test, y_test, k_range)
        optimal_k = best_k
        k_values_list = list(k_range)
    else:
        optimal_k = args.k
        k_accuracies = None
        k_values_list = None
        print(f"\nSTEP 3: Using K={optimal_k} (manually specified)")

    # ─────────────────────────────────────
    # STEP 4: Train KNN Classifier
    # ─────────────────────────────────────
    print(f"\nSTEP 4: TRAINING KNN CLASSIFIER (K={optimal_k})")
    print("-" * 40)

    results = train_classifier(X, y, n_neighbors=optimal_k)
    classifier = results["model"]

    # ─────────────────────────────────────
    # STEP 5: Build Playlist Recommender
    # ─────────────────────────────────────
    print("\nSTEP 5: BUILDING PLAYLIST RECOMMENDER")
    print("-" * 40)

    recommender = build_recommender(X, df, n_neighbors=20)

    # ─────────────────────────────────────
    # STEP 6: Save Models
    # ─────────────────────────────────────
    print("\nSTEP 6: SAVING MODELS")
    print("-" * 40)

    save_models(classifier, recommender)
    save_scaler(scaler)

    # Also save the DataFrame for the app to use
    os.makedirs("data", exist_ok=True)
    df.to_csv("data/processed_songs.csv", index=False)
    print(" Processed dataset saved to data/processed_songs.csv")

    # ─────────────────────────────────────
    # STEP 7: Visualizations
    # ─────────────────────────────────────
    if not args.no_plots:
        print("\nSTEP 7: GENERATING VISUALIZATIONS")
        print("-" * 40)

        generate_all_plots(
            df=df,
            X_scaled=X,
            y=y,
            cm=results["confusion_matrix"],
            k_values=k_values_list,
            accuracies=k_accuracies,
            best_k=optimal_k if k_accuracies else None,
            save_dir="screenshots",
        )
    else:
        print("\nSTEP 7: Skipping visualizations (--no-plots)")

    # ─────────────────────────────────────
    # STEP 8: Demo Recommendations
    # ─────────────────────────────────────
    print("\nSTEP 8: DEMO RECOMMENDATIONS")
    print("-" * 40)

    demo_recommendations(df, X, recommender, classifier)

    # ─────────────────────────────────────
    # FINAL SUMMARY
    # ─────────────────────────────────────
    print("\n" + "=" * 25)
    print(f"""
  TRAINING COMPLETE!

  Results:
     - Dataset:  {len(df)} songs across 4 moods
     - K value:  {optimal_k}
     - Accuracy: {results["accuracy"] * 100:.2f}%

  Saved Files:
     - model/knn_classifier.pkl
     - model/knn_recommender.pkl
     - model/scaler.pkl
     - data/processed_songs.csv
     - screenshots/

  Next Steps:
     Run the web app:
     streamlit run app/streamlit_app.py
""")
    print("=" * 25 + "\n")


if __name__ == "__main__":
    main()
