"""
utils/visualizations.py
------------------------
All plotting functions for the Music Mood Classifier project.

Plots generated:
1. Mood distribution (bar chart + pie chart)
2. Feature correlation heatmap
3. PCA 2D scatter plot of songs
4. Confusion matrix
5. K vs Accuracy curve
6. Feature importance by mood (radar chart)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# Style Configuration
# ─────────────────────────────────────────────
MOOD_COLORS = {
    "happy": "#FFD700",
    "sad": "#4169E1",
    "energetic": "#FF4500",
    "calm": "#2E8B57",
}

MOOD_EMOJIS = {
    "happy": "😊 Happy",
    "sad": "😢 Sad",
    "energetic": "⚡ Energetic",
    "calm": "🌿 Calm",
}

plt.style.use("seaborn-v0_8-darkgrid")
FIGSIZE_LARGE = (14, 6)
FIGSIZE_MEDIUM = (10, 6)
FIGSIZE_SMALL = (8, 6)


def plot_mood_distribution(df, save_path=None):
    """
    Plot mood distribution as bar chart and pie chart side by side.
    Shows how many songs belong to each mood category.
    """
    mood_counts = df["mood"].value_counts()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=FIGSIZE_LARGE)
    fig.suptitle("🎵 Mood Distribution in Dataset", fontsize=16, fontweight="bold", y=1.02)

    # ── Bar Chart ──
    colors = [MOOD_COLORS.get(mood, "#888") for mood in mood_counts.index]
    bars = ax1.bar(mood_counts.index, mood_counts.values, color=colors,
                   edgecolor="white", linewidth=1.5, alpha=0.9)

    # Add value labels on bars
    for bar, count in zip(bars, mood_counts.values):
        ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 5,
                 f'{count}\n({count/len(df)*100:.1f}%)',
                 ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax1.set_title("Songs per Mood", fontsize=13)
    ax1.set_xlabel("Mood", fontsize=11)
    ax1.set_ylabel("Number of Songs", fontsize=11)
    ax1.tick_params(axis='x', labelsize=11)
    ax1.set_ylim(0, mood_counts.max() * 1.2)

    # ── Pie Chart ──
    wedges, texts, autotexts = ax2.pie(
        mood_counts.values,
        labels=[MOOD_EMOJIS.get(m, m) for m in mood_counts.index],
        colors=colors,
        autopct='%1.1f%%',
        startangle=90,
        explode=[0.05] * len(mood_counts),
        shadow=True,
    )

    for text in autotexts:
        text.set_fontsize(11)
        text.set_fontweight('bold')

    ax2.set_title("Mood Proportions", fontsize=13)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"✅ Saved mood distribution plot to {save_path}")

    return fig


def plot_feature_correlation(df, save_path=None):
    """
    Heatmap showing correlation between audio features.
    High correlation = features move together.
    Low correlation = features are independent.
    """
    feature_cols = [
        "valence", "energy", "tempo", "danceability",
        "loudness", "acousticness", "instrumentalness",
        "speechiness", "liveness"
    ]

    # Calculate correlation matrix
    corr_matrix = df[feature_cols].corr()

    fig, ax = plt.subplots(figsize=(10, 8))
    fig.suptitle("🔥 Feature Correlation Heatmap", fontsize=16, fontweight="bold")

    # Draw heatmap
    mask = np.zeros_like(corr_matrix, dtype=bool)  # Show full matrix
    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt=".2f",
        cmap="RdYlBu_r",
        center=0,
        vmin=-1, vmax=1,
        square=True,
        linewidths=0.5,
        ax=ax,
        annot_kws={"size": 9},
        cbar_kws={"shrink": 0.8}
    )

    ax.set_title(
        "Values close to 1.0 = strong positive correlation\n"
        "Values close to -1.0 = strong negative correlation",
        fontsize=10, color="gray"
    )
    ax.tick_params(axis='both', labelsize=10)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"✅ Saved correlation heatmap to {save_path}")

    return fig


def plot_pca_visualization(X_scaled, y, save_path=None):
    """
    Reduce 9D features to 2D using PCA and scatter plot songs.

    PCA = Principal Component Analysis
    It finds the 2 directions with the most variance,
    letting us visualize high-dimensional data in 2D.

    Args:
        X_scaled: Normalized feature matrix
        y: Mood labels
    """
    # Reduce to 2 dimensions
    pca = PCA(n_components=2, random_state=42)
    X_2d = pca.fit_transform(X_scaled)

    # Explained variance tells us how much info is retained
    var_explained = pca.explained_variance_ratio_
    total_var = sum(var_explained) * 100

    fig, ax = plt.subplots(figsize=FIGSIZE_LARGE)
    fig.suptitle("🗺️ Songs in 2D Feature Space (PCA)", fontsize=16, fontweight="bold")

    # Plot each mood with different color
    moods = np.unique(y)
    for mood in moods:
        mask = y == mood
        ax.scatter(
            X_2d[mask, 0], X_2d[mask, 1],
            c=MOOD_COLORS.get(mood, "#888"),
            label=MOOD_EMOJIS.get(mood, mood),
            alpha=0.5,
            s=30,
            edgecolors='none'
        )

    ax.set_xlabel(f"PC1 ({var_explained[0]*100:.1f}% variance)", fontsize=12)
    ax.set_ylabel(f"PC2 ({var_explained[1]*100:.1f}% variance)", fontsize=12)
    ax.set_title(
        f"Each dot = one song | Total variance explained: {total_var:.1f}%\n"
        "Clustered dots = songs with similar audio features",
        fontsize=10, color="gray"
    )
    ax.legend(fontsize=11, markerscale=2)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"✅ Saved PCA visualization to {save_path}")

    return fig


def plot_confusion_matrix(cm, classes, save_path=None):
    """
    Visualize the confusion matrix.

    Diagonal = correct predictions ✅
    Off-diagonal = mistakes ❌

    Args:
        cm: Confusion matrix (numpy array)
        classes: List of class names
    """
    fig, ax = plt.subplots(figsize=FIGSIZE_SMALL)
    fig.suptitle("🎯 Confusion Matrix", fontsize=16, fontweight="bold")

    # Normalize to percentages
    cm_normalized = cm.astype(float) / cm.sum(axis=1)[:, np.newaxis] * 100

    sns.heatmap(
        cm_normalized,
        annot=True,
        fmt=".1f",
        cmap="Blues",
        xticklabels=classes,
        yticklabels=classes,
        ax=ax,
        annot_kws={"size": 12},
        linewidths=1,
        cbar_kws={"label": "% of true class"}
    )

    ax.set_xlabel("Predicted Mood", fontsize=12)
    ax.set_ylabel("True Mood", fontsize=12)
    ax.set_title(
        "Diagonal = correct predictions | Off-diagonal = misclassifications",
        fontsize=9, color="gray"
    )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"✅ Saved confusion matrix to {save_path}")

    return fig


def plot_k_vs_accuracy(k_values, accuracies, best_k, save_path=None):
    """
    Plot how accuracy changes with different K values.
    Helps us pick the best K for KNN.

    Args:
        k_values: List of K values tested
        accuracies: Corresponding accuracy for each K
        best_k: The K with highest accuracy
    """
    fig, ax = plt.subplots(figsize=FIGSIZE_MEDIUM)
    fig.suptitle("🔍 Finding Optimal K Value", fontsize=16, fontweight="bold")

    acc_list = [accuracies[k] * 100 for k in k_values]

    ax.plot(k_values, acc_list, "b-o", linewidth=2, markersize=8, label="Accuracy")
    ax.axvline(x=best_k, color="red", linestyle="--", linewidth=2,
               label=f"Best K = {best_k}")
    ax.scatter([best_k], [accuracies[best_k]*100], color="red", s=200, zorder=5)

    ax.set_xlabel("K (Number of Neighbors)", fontsize=12)
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_title(
        "Too small K → overfitting | Too large K → underfitting",
        fontsize=10, color="gray"
    )
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(min(acc_list) - 2, 102)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"✅ Saved K vs Accuracy plot to {save_path}")

    return fig


def plot_feature_by_mood(df, save_path=None):
    """
    Box plots showing distribution of each feature per mood.
    This visualizes what makes each mood unique.
    """
    feature_cols = ["valence", "energy", "tempo", "danceability",
                    "acousticness", "loudness"]

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle("📊 Audio Feature Distribution by Mood", fontsize=16, fontweight="bold")

    axes = axes.flatten()

    for i, feature in enumerate(feature_cols):
        ax = axes[i]

        moods = ["happy", "sad", "energetic", "calm"]
        data_by_mood = [df[df["mood"] == mood][feature].values for mood in moods]
        colors = [MOOD_COLORS[m] for m in moods]

        bp = ax.boxplot(
            data_by_mood,
            labels=moods,
            patch_artist=True,
            notch=True,
            medianprops=dict(color="black", linewidth=2),
        )

        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        ax.set_title(feature.capitalize(), fontsize=12, fontweight="bold")
        ax.tick_params(axis='x', labelsize=9)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"✅ Saved feature by mood plot to {save_path}")

    return fig


def generate_all_plots(df, X_scaled, y, cm, k_values=None, accuracies=None,
                       best_k=None, save_dir="screenshots/"):
    """
    Generate and save ALL visualization plots.

    Args:
        df: DataFrame
        X_scaled: Scaled features
        y: Labels
        cm: Confusion matrix
        k_values, accuracies, best_k: From K optimization
        save_dir: Directory to save plots
    """
    import os
    os.makedirs(save_dir, exist_ok=True)

    print("\n📊 Generating all visualizations...")

    plot_mood_distribution(df, save_path=f"{save_dir}/1_mood_distribution.png")
    plot_feature_correlation(df, save_path=f"{save_dir}/2_feature_correlation.png")
    plot_pca_visualization(X_scaled, y, save_path=f"{save_dir}/3_pca_visualization.png")
    plot_confusion_matrix(cm, ["happy", "sad", "energetic", "calm"],
                          save_path=f"{save_dir}/4_confusion_matrix.png")
    plot_feature_by_mood(df, save_path=f"{save_dir}/5_feature_by_mood.png")

    if k_values and accuracies and best_k:
        plot_k_vs_accuracy(k_values, accuracies, best_k,
                           save_path=f"{save_dir}/6_k_vs_accuracy.png")

    print(f"\n✅ All plots saved to '{save_dir}/'")
