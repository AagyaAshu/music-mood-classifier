"""
app/streamlit_app.py
---------------------
Streamlit UI for the Music Mood Classifier & Playlist Generator

Features:
- Select mood or enter a song name
- View recommended playlist
- See feature radar chart for any song
- Filter by language, popularity
- Dynamic K selection
- Beautiful dark-themed UI

Run with:
    streamlit run app/streamlit_app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import pickle
import os
import sys
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors

# ─── Add project root to Python path ───
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.preprocessing import FEATURE_COLS, MOOD_COLORS, MOOD_EMOJIS

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="🎵 Music Mood Classifier",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown(
    """
<style>
    /* Main background */
    .stApp {
        background: linear-gradient(135deg, #0a0a1a 0%, #1a0a2e 50%, #0a1a2e 100%);
    }

    /* Headers */
    h1, h2, h3 {
        color: #ffffff !important;
    }

    /* Mood cards */
    .mood-card {
        background: rgba(255,255,255,0.05);
        border-radius: 16px;
        padding: 20px;
        border: 1px solid rgba(255,255,255,0.1);
        text-align: center;
        transition: all 0.3s;
        cursor: pointer;
    }

    .mood-card:hover {
        background: rgba(255,255,255,0.1);
        transform: translateY(-2px);
    }

    /* Song card */
    .song-card {
        background: rgba(255,255,255,0.05);
        border-radius: 12px;
        padding: 16px;
        margin: 8px 0;
        border-left: 4px solid #7B2FBE;
    }

    /* Metric cards */
    .metric-box {
        background: rgba(123, 47, 190, 0.2);
        border-radius: 10px;
        padding: 12px;
        text-align: center;
        border: 1px solid rgba(123, 47, 190, 0.4);
    }

    /* Feature badge */
    .feature-badge {
        display: inline-block;
        background: rgba(123, 47, 190, 0.3);
        padding: 4px 10px;
        border-radius: 20px;
        font-size: 12px;
        margin: 2px;
        color: white;
    }

    /* Sidebar */
    .css-1d391kg {
        background: rgba(0,0,0,0.5);
    }

    /* Streamlit buttons */
    .stButton > button {
        background: linear-gradient(135deg, #7B2FBE, #4A90E2);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 10px 24px;
        font-weight: 600;
        transition: all 0.3s;
    }

    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 20px rgba(123, 47, 190, 0.4);
    }

    /* Selectbox */
    .stSelectbox > div > div {
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.2);
        color: white;
    }

    /* Divider */
    hr {
        border-color: rgba(255,255,255,0.1);
    }

    /* Info boxes */
    .stAlert {
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.1);
    }
</style>
""",
    unsafe_allow_html=True,
)


# ─────────────────────────────────────────────
# LOAD MODELS (with caching for speed)
# ─────────────────────────────────────────────
@st.cache_resource
def load_models_and_data():
    """Load all trained models and data. Cached so it only runs once."""

    # Check if models exist, if not train them first
    model_files = [
        "model/knn_classifier.pkl",
        "model/knn_recommender.pkl",
        "model/scaler.pkl",
        "data/processed_songs.csv",
    ]

    # Auto-train if models don't exist
    if not all(os.path.exists(f) for f in model_files):
        st.info("🤖 First run! Training models... (this takes ~30 seconds)")
        import subprocess

        result = subprocess.run(
            ["python", "main.py", "--no-plots"],
            capture_output=True,
            text=True,
            cwd=os.path.dirname(os.path.dirname(__file__)),
        )
        if result.returncode != 0:
            st.error(f"Training failed: {result.stderr}")
            return None, None, None, None

    # Load everything
    base = os.path.dirname(os.path.dirname(__file__))

    with open(os.path.join(base, "model/knn_classifier.pkl"), "rb") as f:
        classifier = pickle.load(f)

    with open(os.path.join(base, "model/knn_recommender.pkl"), "rb") as f:
        recommender = pickle.load(f)

    with open(os.path.join(base, "model/scaler.pkl"), "rb") as f:
        scaler = pickle.load(f)

    df = pd.read_csv(os.path.join(base, "data/processed_songs.csv"))

    # Rebuild scaled features for recommendations
    X_scaled = scaler.transform(df[FEATURE_COLS].values)

    return classifier, recommender, scaler, df, X_scaled


# ─────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────
def get_mood_gradient(mood):
    gradients = {
        "happy": "linear-gradient(135deg, #FFD700, #FFA500)",
        "sad": "linear-gradient(135deg, #4169E1, #6495ED)",
        "energetic": "linear-gradient(135deg, #FF4500, #FF6347)",
        "calm": "linear-gradient(135deg, #2E8B57, #3CB371)",
    }
    return gradients.get(mood, "linear-gradient(135deg, #7B2FBE, #4A90E2)")


def plot_radar_chart(song_features, song_name):
    """Create radar chart for song features."""

    features = [
        "valence",
        "energy",
        "danceability",
        "acousticness",
        "speechiness",
        "liveness",
        "instrumentalness",
    ]

    # Normalize loudness (it's negative) and tempo to 0-1
    values = []
    for feat in features:
        val = song_features.get(feat, 0)
        values.append(float(val))

    # Number of variables
    N = len(features)
    angles = [n / float(N) * 2 * 3.14159 for n in range(N)]
    angles += angles[:1]  # Close the circle
    values += values[:1]

    fig, ax = plt.subplots(figsize=(5, 5), subplot_kw=dict(polar=True))
    fig.patch.set_facecolor("#0a0a1a")
    ax.set_facecolor("#0a0a1a")

    ax.plot(angles, values, "o-", linewidth=2, color="#7B2FBE")
    ax.fill(angles, values, alpha=0.25, color="#7B2FBE")

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(features, size=9, color="white")
    ax.set_ylim(0, 1)
    ax.tick_params(colors="white")
    ax.spines["polar"].set_visible(False)
    ax.grid(color=(1, 1, 1, 0.1))
    ax.set_title(
        f"Audio Features\n{song_name[:30]}...", color="white", fontsize=10, pad=20
    )

    return fig


def plot_mood_bar(probabilities):
    """Create mood probability bar chart."""
    moods = list(probabilities.keys())
    probs = list(probabilities.values())
    colors = [MOOD_COLORS.get(m, "#888") for m in moods]

    fig, ax = plt.subplots(figsize=(6, 3))
    fig.patch.set_facecolor("#0a0a1a")
    ax.set_facecolor("#0a0a1a")

    bars = ax.barh(moods, [p * 100 for p in probs], color=colors, alpha=0.8, height=0.5)

    for bar, prob in zip(bars, probs):
        ax.text(
            bar.get_width() + 1,
            bar.get_y() + bar.get_height() / 2,
            f"{prob * 100:.1f}%",
            va="center",
            color="white",
            fontsize=10,
        )

    ax.set_xlabel("Probability (%)", color="white")
    ax.tick_params(colors="white")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color((1, 1, 1, 0.2))
    ax.spines["bottom"].set_color((1, 1, 1, 0.2))
    ax.set_xlim(0, 115)
    ax.set_title("Mood Probabilities", color="white", fontsize=11)

    return fig


# ─────────────────────────────────────────────
# MAIN APP
# ─────────────────────────────────────────────
def main():

    # ── Header ──
    st.markdown(
        """
    <div style="text-align: center; padding: 20px 0;">
        <h1 style="font-size: 2.5rem; background: linear-gradient(135deg, #FFD700, #FF4500, #4169E1, #2E8B57);
                   -webkit-background-clip: text; -webkit-text-fill-color: transparent;
                   background-clip: text;">
            🎵 Music Mood Classifier
        </h1>
        <p style="color: #aaa; font-size: 1.1rem;">
            Discover your playlist based on how you feel · Powered by K-Nearest Neighbors AI
        </p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    st.divider()

    # ── Load Models ──
    result = load_models_and_data()
    if result is None or len(result) != 5:
        st.error("❌ Failed to load models. Please run `python main.py` first.")
        return

    classifier, recommender, scaler, df, X_scaled = result

    # ─────────────────────────────────────────────
    # SIDEBAR
    # ─────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### ⚙️ Settings")

        k_value = st.slider(
            "🔢 Number of recommendations (K)",
            min_value=3,
            max_value=20,
            value=5,
            help="How many similar songs to recommend",
        )

        st.divider()

        st.markdown("### 🔍 Filters")

        lang_options = ["All"] + sorted(df["language"].unique().tolist())
        language_filter = st.selectbox("🌍 Language", lang_options)

        min_popularity = st.slider(
            "⭐ Minimum Popularity",
            min_value=0,
            max_value=100,
            value=0,
            help="Filter songs by minimum popularity score",
        )

        mood_filter_options = ["None (show all moods)"] + [
            "happy",
            "sad",
            "energetic",
            "calm",
        ]
        mood_filter = st.selectbox("🎭 Filter results by mood", mood_filter_options)
        if mood_filter == "None (show all moods)":
            mood_filter = None

        st.divider()

        st.markdown("### 📊 Dataset Info")
        st.metric("Total Songs", len(df))
        st.metric("Happy 😊", len(df[df["mood"] == "happy"]))
        st.metric("Sad 😢", len(df[df["mood"] == "sad"]))
        st.metric("Energetic ⚡", len(df[df["mood"] == "energetic"]))
        st.metric("Calm 🌿", len(df[df["mood"] == "calm"]))

    # ─────────────────────────────────────────────
    # MAIN TABS
    # ─────────────────────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs(
        ["🎭 Mood Playlist", "🔍 Search by Song", "🎛️ Custom Features", "📊 Insights"]
    )

    # ─────────────────────────────────────────────
    # TAB 1: MOOD-BASED PLAYLIST
    # ─────────────────────────────────────────────
    with tab1:
        st.markdown("## 🎭 How are you feeling today?")
        st.markdown("Select your mood and we'll generate the perfect playlist!")

        # Mood selection buttons
        col1, col2, col3, col4 = st.columns(4)

        mood_selected = st.session_state.get("selected_mood", None)

        with col1:
            if st.button(
                "😊 Happy\n\nUpbeat & joyful", use_container_width=True, key="btn_happy"
            ):
                st.session_state["selected_mood"] = "happy"
                st.rerun()

        with col2:
            if st.button(
                "😢 Sad\n\nMelancholic & slow", use_container_width=True, key="btn_sad"
            ):
                st.session_state["selected_mood"] = "sad"
                st.rerun()

        with col3:
            if st.button(
                "⚡ Energetic\n\nHigh BPM & intense",
                use_container_width=True,
                key="btn_energetic",
            ):
                st.session_state["selected_mood"] = "energetic"
                st.rerun()

        with col4:
            if st.button(
                "🌿 Calm\n\nPeaceful & acoustic",
                use_container_width=True,
                key="btn_calm",
            ):
                st.session_state["selected_mood"] = "calm"
                st.rerun()

        # Show selected mood and recommendations
        mood = st.session_state.get("selected_mood")

        if mood:
            st.divider()

            emoji = {"happy": "😊", "sad": "😢", "energetic": "⚡", "calm": "🌿"}[mood]
            color = MOOD_COLORS[mood]

            st.markdown(
                f"""
            <div style="background: {get_mood_gradient(mood)};
                        border-radius: 12px; padding: 16px 24px; margin: 16px 0;
                        display: flex; align-items: center; gap: 12px;">
                <span style="font-size: 2rem;">{emoji}</span>
                <div>
                    <h3 style="margin: 0; color: white;">
                        {mood.upper()} Playlist
                    </h3>
                    <p style="margin: 0; color: rgba(255,255,255,0.8); font-size: 0.9rem;">
                        Top {k_value} songs curated for your mood
                    </p>
                </div>
            </div>
            """,
                unsafe_allow_html=True,
            )

            # Get recommendations
            try:
                from model.knn_model import recommend_songs

                recommendations = recommend_songs(
                    mood_or_features=mood,
                    df=df,
                    X_scaled=X_scaled,
                    recommender=recommender,
                    k=k_value,
                    mood_filter=mood_filter,
                    lang_filter=language_filter,
                    min_popularity=min_popularity,
                )

                if len(recommendations) == 0:
                    st.warning(
                        "No songs match your filters. Try adjusting the settings in the sidebar."
                    )
                else:
                    st.markdown(f"### 🎵 Your Playlist ({len(recommendations)} songs)")

                    for i, (_, song) in enumerate(recommendations.iterrows(), 1):
                        song_mood_color = MOOD_COLORS.get(song["mood"], "#888")
                        song_emoji = {
                            "happy": "😊",
                            "sad": "😢",
                            "energetic": "⚡",
                            "calm": "🌿",
                        }.get(song["mood"], "🎵")

                        with st.container():
                            c1, c2 = st.columns([4, 1])
                            with c1:
                                st.markdown(
                                    f"""
                                <div class="song-card" style="border-left-color: {song_mood_color}">
                                    <div style="display: flex; align-items: center; gap: 12px;">
                                        <div style="background: {song_mood_color}; border-radius: 50%;
                                                    width: 36px; height: 36px; display: flex;
                                                    align-items: center; justify-content: center;
                                                    font-weight: bold; color: white; font-size: 14px;">
                                            {i}
                                        </div>
                                        <div>
                                            <strong style="color: white; font-size: 16px;">
                                                {song["song_name"]}
                                            </strong><br>
                                            <span style="color: #aaa;">
                                                {song["artist"]}
                                            </span>
                                            &nbsp;&nbsp;
                                            <span style="background: {song_mood_color}33;
                                                         color: {song_mood_color}; padding: 2px 8px;
                                                         border-radius: 12px; font-size: 12px;">
                                                {song_emoji} {song["mood"]}
                                            </span>
                                        </div>
                                    </div>
                                    <div style="margin-top: 10px; display: flex; gap: 16px; flex-wrap: wrap;">
                                        <span style="color: #aaa; font-size: 13px;">
                                            💃 Dance: <strong style="color: white">{song["danceability"]:.2f}</strong>
                                        </span>
                                        <span style="color: #aaa; font-size: 13px;">
                                            ⚡ Energy: <strong style="color: white">{song["energy"]:.2f}</strong>
                                        </span>
                                        <span style="color: #aaa; font-size: 13px;">
                                            😊 Valence: <strong style="color: white">{song["valence"]:.2f}</strong>
                                        </span>
                                        <span style="color: #aaa; font-size: 13px;">
                                            🎵 Tempo: <strong style="color: white">{song["tempo"]:.0f} BPM</strong>
                                        </span>
                                        <span style="color: #aaa; font-size: 13px;">
                                            ⭐ Pop: <strong style="color: white">{song.get("popularity", "N/A")}</strong>
                                        </span>
                                        <span style="color: #aaa; font-size: 13px;">
                                            🌍 {song.get("language", "Unknown")}
                                        </span>
                                    </div>
                                </div>
                                """,
                                    unsafe_allow_html=True,
                                )

            except Exception as e:
                st.error(f"Error generating recommendations: {e}")

    # ─────────────────────────────────────────────
    # TAB 2: SEARCH BY SONG NAME
    # ─────────────────────────────────────────────
    with tab2:
        st.markdown("## 🔍 Find Similar Songs")
        st.markdown("Search for a song and discover music with similar vibes!")

        search_query = st.text_input(
            "🎵 Enter a song name:",
            placeholder="e.g. Walking on Sunshine, Creep, Sandstorm...",
            key="song_search",
        )

        if search_query:
            from model.knn_model import find_song_by_name

            matches = find_song_by_name(search_query, df)

            if len(matches) == 0:
                st.warning(
                    f"No songs found matching '{search_query}'. Try a different name."
                )
            else:
                st.success(f"Found {len(matches)} matching song(s):")

                # Show matching songs
                selected_song_name = st.selectbox(
                    "Select a song:", matches["song_name"].tolist(), key="song_select"
                )

                selected_song = matches[
                    matches["song_name"] == selected_song_name
                ].iloc[0]
                song_mood = selected_song["mood"]
                song_color = MOOD_COLORS.get(song_mood, "#888")
                song_emoji = {
                    "happy": "😊",
                    "sad": "😢",
                    "energetic": "⚡",
                    "calm": "🌿",
                }.get(song_mood, "🎵")

                # Show selected song details
                col_info, col_chart = st.columns([1, 1])

                with col_info:
                    st.markdown(
                        f"""
                    <div style="background: rgba(255,255,255,0.05); border-radius: 12px;
                                padding: 20px; border: 1px solid rgba(255,255,255,0.1);">
                        <h3 style="color: white; margin: 0 0 4px 0;">{selected_song["song_name"]}</h3>
                        <p style="color: #aaa; margin: 0 0 16px 0;">by {selected_song["artist"]}</p>

                        <div style="background: {song_color}22; padding: 8px 16px;
                                    border-radius: 8px; display: inline-block;
                                    border: 1px solid {song_color}55; margin-bottom: 16px;">
                            <span style="color: {song_color}; font-weight: 600;">
                                {song_emoji} Mood: {song_mood.upper()}
                            </span>
                        </div>

                        <table style="width: 100%; color: white; font-size: 14px;">
                            <tr><td style="color: #aaa;">Valence</td>
                                <td>{selected_song["valence"]:.3f}</td></tr>
                            <tr><td style="color: #aaa;">Energy</td>
                                <td>{selected_song["energy"]:.3f}</td></tr>
                            <tr><td style="color: #aaa;">Tempo</td>
                                <td>{selected_song["tempo"]:.0f} BPM</td></tr>
                            <tr><td style="color: #aaa;">Danceability</td>
                                <td>{selected_song["danceability"]:.3f}</td></tr>
                            <tr><td style="color: #aaa;">Acousticness</td>
                                <td>{selected_song["acousticness"]:.3f}</td></tr>
                            <tr><td style="color: #aaa;">Popularity</td>
                                <td>⭐ {selected_song.get("popularity", "N/A")}/100</td></tr>
                            <tr><td style="color: #aaa;">Language</td>
                                <td>🌍 {selected_song.get("language", "Unknown")}</td></tr>
                        </table>
                    </div>
                    """,
                        unsafe_allow_html=True,
                    )

                with col_chart:
                    # Radar chart
                    radar = plot_radar_chart(
                        selected_song.to_dict(), selected_song["song_name"]
                    )
                    st.pyplot(radar, use_container_width=True)
                    plt.close()

                # Mood prediction
                st.divider()
                st.markdown("### 🤖 AI Mood Analysis")

                song_features = scaler.transform(
                    selected_song[FEATURE_COLS].values.reshape(1, -1)
                )
                predicted_mood, probs = classifier.predict(song_features)[0], {}

                # Get probabilities
                proba = classifier.predict_proba(song_features)[0]
                for cls, p in zip(classifier.classes_, proba):
                    probs[cls] = float(p)

                col_pred, col_probs = st.columns([1, 2])

                with col_pred:
                    pred_color = MOOD_COLORS.get(predicted_mood, "#888")
                    pred_emoji = {
                        "happy": "😊",
                        "sad": "😢",
                        "energetic": "⚡",
                        "calm": "🌿",
                    }.get(predicted_mood, "🎵")
                    st.markdown(
                        f"""
                    <div style="background: {pred_color}22; border-radius: 12px;
                                padding: 24px; text-align: center;
                                border: 2px solid {pred_color};">
                        <div style="font-size: 3rem;">{pred_emoji}</div>
                        <h3 style="color: {pred_color}; margin: 8px 0 4px 0;">
                            {predicted_mood.upper()}
                        </h3>
                        <p style="color: #aaa; margin: 0; font-size: 0.85rem;">
                            AI Predicted Mood
                        </p>
                    </div>
                    """,
                        unsafe_allow_html=True,
                    )

                with col_probs:
                    prob_chart = plot_mood_bar(probs)
                    st.pyplot(prob_chart, use_container_width=True)
                    plt.close()

                # Similar songs
                st.divider()
                st.markdown(f"### 🎵 Songs Similar to '{selected_song_name}'")

                # Get the index of this song
                song_idx = df[df["song_name"] == selected_song_name].index[0]
                query_features = X_scaled[song_idx].reshape(1, -1)

                from model.knn_model import recommend_songs

                similar = recommend_songs(
                    mood_or_features=query_features,
                    df=df,
                    X_scaled=X_scaled,
                    recommender=recommender,
                    k=k_value,
                    mood_filter=mood_filter,
                    lang_filter=language_filter,
                    min_popularity=min_popularity,
                )

                for i, (_, song) in enumerate(similar.iterrows(), 1):
                    song_mood_c = MOOD_COLORS.get(song["mood"], "#888")
                    song_e = {
                        "happy": "😊",
                        "sad": "😢",
                        "energetic": "⚡",
                        "calm": "🌿",
                    }.get(song["mood"], "🎵")
                    sim_pct = song.get("similarity_score", 0) * 100

                    st.markdown(
                        f"""
                    <div class="song-card" style="border-left-color: {song_mood_c}">
                        <strong style="color: white;">{i}. {song["song_name"]}</strong>
                        <span style="color: #aaa;"> — {song["artist"]}</span>
                        <span style="float: right; color: {song_mood_c};">{song_e} {song["mood"]}</span><br>
                        <small style="color: #aaa;">
                            Match: {sim_pct:.1f}% |
                            Energy: {song["energy"]:.2f} |
                            Valence: {song["valence"]:.2f} |
                            🎵 {song["tempo"]:.0f} BPM
                        </small>
                    </div>
                    """,
                        unsafe_allow_html=True,
                    )

    # ─────────────────────────────────────────────
    # TAB 3: CUSTOM FEATURE INPUT
    # ─────────────────────────────────────────────
    with tab3:
        st.markdown("## 🎛️ Describe Your Ideal Song")
        st.markdown(
            "Adjust audio features and let AI classify the mood & find similar songs!"
        )

        col_sliders, col_result = st.columns([1, 1])

        with col_sliders:
            st.markdown("### Adjust Audio Features")

            valence = st.slider("😊 Valence (musical positivity)", 0.0, 1.0, 0.5, 0.01)
            energy = st.slider("⚡ Energy (intensity)", 0.0, 1.0, 0.5, 0.01)
            tempo = st.slider("🎵 Tempo (BPM)", 60.0, 200.0, 120.0, 1.0)
            danceability = st.slider("💃 Danceability", 0.0, 1.0, 0.5, 0.01)
            loudness = st.slider("🔊 Loudness (dB)", -60.0, 0.0, -10.0, 0.5)
            acousticness = st.slider("🎸 Acousticness", 0.0, 1.0, 0.3, 0.01)
            instrumentalness = st.slider("🎹 Instrumentalness", 0.0, 1.0, 0.1, 0.01)
            speechiness = st.slider("🗣️ Speechiness", 0.0, 1.0, 0.05, 0.01)
            liveness = st.slider("🎤 Liveness", 0.0, 1.0, 0.15, 0.01)

        with col_result:
            # Predict mood from custom features
            custom_features = np.array(
                [
                    [
                        valence,
                        energy,
                        tempo,
                        danceability,
                        loudness,
                        acousticness,
                        instrumentalness,
                        speechiness,
                        liveness,
                    ]
                ]
            )

            custom_scaled = scaler.transform(custom_features)
            predicted_mood = classifier.predict(custom_scaled)[0]
            proba = classifier.predict_proba(custom_scaled)[0]
            probs = {cls: float(p) for cls, p in zip(classifier.classes_, proba)}

            mood_color = MOOD_COLORS.get(predicted_mood, "#888")
            mood_emoji_str = {
                "happy": "😊",
                "sad": "😢",
                "energetic": "⚡",
                "calm": "🌿",
            }.get(predicted_mood, "🎵")

            st.markdown(
                f"""
            <div style="background: {get_mood_gradient(predicted_mood)};
                        border-radius: 16px; padding: 30px; text-align: center;
                        margin-bottom: 16px;">
                <div style="font-size: 4rem;">{mood_emoji_str}</div>
                <h2 style="color: white; margin: 8px 0 4px 0;">{predicted_mood.upper()}</h2>
                <p style="color: rgba(255,255,255,0.8);">Predicted Mood</p>
            </div>
            """,
                unsafe_allow_html=True,
            )

            # Probability bars
            prob_chart = plot_mood_bar(probs)
            st.pyplot(prob_chart, use_container_width=True)
            plt.close()

        # Find similar songs
        st.divider()
        st.markdown("### 🎵 Songs Matching Your Audio Profile")

        from model.knn_model import recommend_songs

        custom_recs = recommend_songs(
            mood_or_features=custom_scaled,
            df=df,
            X_scaled=X_scaled,
            recommender=recommender,
            k=k_value,
            mood_filter=mood_filter,
            lang_filter=language_filter,
            min_popularity=min_popularity,
        )

        cols = st.columns(max(1, min(3, len(custom_recs))))

        for i, (_, song) in enumerate(custom_recs.iterrows()):
            with cols[i % 3]:
                song_mood_c = MOOD_COLORS.get(song["mood"], "#888")
                song_e = {
                    "happy": "😊",
                    "sad": "😢",
                    "energetic": "⚡",
                    "calm": "🌿",
                }.get(song["mood"], "🎵")
                st.markdown(
                    f"""
                <div style="background: rgba(255,255,255,0.05); border-radius: 12px;
                            padding: 14px; margin: 8px 0; border: 1px solid {song_mood_c}44;
                            text-align: center;">
                    <div style="font-size: 1.5rem;">{song_e}</div>
                    <strong style="color: white;">{song["song_name"][:25]}</strong><br>
                    <small style="color: #aaa;">{song["artist"]}</small><br>
                    <small style="color: {song_mood_c};">● {song["mood"]}</small>
                </div>
                """,
                    unsafe_allow_html=True,
                )

    # ─────────────────────────────────────────────
    # TAB 4: INSIGHTS
    # ─────────────────────────────────────────────
    with tab4:
        st.markdown("## 📊 Dataset Insights")

        # Summary metrics
        c1, c2, c3, c4 = st.columns(4)
        mood_counts = df["mood"].value_counts()

        for col, (mood, count) in zip([c1, c2, c3, c4], mood_counts.items()):
            color = MOOD_COLORS.get(mood, "#888")
            emoji = {"happy": "😊", "sad": "😢", "energetic": "⚡", "calm": "🌿"}.get(
                mood, "🎵"
            )
            with col:
                st.markdown(
                    f"""
                <div style="background: {color}22; border-radius: 12px;
                            padding: 20px; text-align: center;
                            border: 1px solid {color}55;">
                    <div style="font-size: 2rem;">{emoji}</div>
                    <h3 style="color: {color}; margin: 4px 0;">{count}</h3>
                    <p style="color: #aaa; margin: 0; font-size: 13px;">{mood.capitalize()}</p>
                </div>
                """,
                    unsafe_allow_html=True,
                )

        st.divider()

        # Charts
        col_plot1, col_plot2 = st.columns(2)

        with col_plot1:
            st.markdown("#### 📈 Feature Averages by Mood")
            feature_avg = df.groupby("mood")[
                ["valence", "energy", "danceability", "acousticness"]
            ].mean()

            fig, ax = plt.subplots(figsize=(7, 4))
            fig.patch.set_facecolor("#0a0a1a")
            ax.set_facecolor("#0a0a1a")

            x = np.arange(len(feature_avg.columns))
            width = 0.2

            for i, (mood, row) in enumerate(feature_avg.iterrows()):
                color = MOOD_COLORS.get(mood, "#888")
                ax.bar(
                    x + i * width,
                    row.values,
                    width,
                    label=mood,
                    color=color,
                    alpha=0.85,
                )

            ax.set_xticks(x + width * 1.5)
            ax.set_xticklabels(feature_avg.columns, color="white", fontsize=10)
            ax.tick_params(colors="white")
            ax.legend(facecolor="#1a1a2e", labelcolor="white")
            ax.spines["bottom"].set_color((1, 1, 1, 0.2))
            ax.spines["left"].set_color((1, 1, 1, 0.2))
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.set_title("Feature Averages by Mood", color="white")
            ax.set_ylim(0, 1.1)

            st.pyplot(fig, use_container_width=True)
            plt.close()

        with col_plot2:
            st.markdown("#### 🔗 Feature Correlation Heatmap")

            corr = df[
                ["valence", "energy", "tempo", "danceability", "acousticness"]
            ].corr()

            fig, ax = plt.subplots(figsize=(6, 5))
            fig.patch.set_facecolor("#0a0a1a")
            ax.set_facecolor("#0a0a1a")

            sns.heatmap(
                corr,
                annot=True,
                fmt=".2f",
                cmap="coolwarm",
                center=0,
                ax=ax,
                linewidths=0.5,
                cbar=False,
                annot_kws={"color": "white", "size": 9},
            )
            ax.tick_params(colors="white", labelsize=9)
            ax.set_title("Correlation Heatmap", color="white")

            st.pyplot(fig, use_container_width=True)
            plt.close()

        # About KNN Section
        st.divider()
        st.markdown("### 🤖 How Does KNN Work?")

        st.markdown(
            """
        <div style="background: rgba(123, 47, 190, 0.1); border-radius: 12px;
                    padding: 24px; border: 1px solid rgba(123, 47, 190, 0.3);">

        <h4 style="color: #A78BFA;">K-Nearest Neighbors (KNN) — Simple Explanation</h4>

        <p style="color: #ddd;">
        Imagine you moved to a new city and want to know your neighborhood's vibe.
        You'd look at your <strong>K nearest neighbors</strong> — if most of them are
        calm musicians, chances are your area is calm too! KNN works the same way:
        </p>

        <ol style="color: #ddd; line-height: 2;">
            <li>Every song is a point in 9-dimensional space (one dimension per feature)</li>
            <li>When you input a new song, KNN calculates <em>distances</em> to all other songs</li>
            <li>It finds the K closest songs (nearest neighbors)</li>
            <li>It counts which mood appears most among those K songs</li>
            <li>That mood wins! 🎉</li>
        </ol>

        <p style="color: #aaa; font-size: 0.9rem;">
        Distance = √[(v₁−v₂)² + (e₁−e₂)² + (t₁−t₂)² + ...] (Euclidean Distance)
        </p>

        </div>
        """,
            unsafe_allow_html=True,
        )

    # ─────────────────────────────────────────────
    # FOOTER
    # ─────────────────────────────────────────────
    st.divider()
    st.markdown(
        """
    <div style="text-align: center; color: #555; padding: 16px 0; font-size: 0.85rem;">
        🎵 Music Mood Classifier · Built with KNN + Streamlit · 
        <span style="color: #7B2FBE;">Made for learning ML</span>
    </div>
    """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
