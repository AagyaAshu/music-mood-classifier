🔗 **[Live Demo](https://music-mood-classifier-3hqgutjgsypaidq3daa2uj.streamlit.app)**

# 🎵 Music Mood Classifier & Playlist Generator

> Classify songs into moods and get personalized playlists — powered by **K-Nearest Neighbors (KNN)**

---

## 📌 What Does This Project Do?

This project takes a song's **audio features** (like energy, tempo, valence) and:

1. **Classifies** it into one of 4 moods: `happy 😊`, `sad 😢`, `energetic ⚡`, `calm 🌿`
2. **Recommends** a playlist of similar songs based on your selected mood or a song you like
3. **Visualizes** patterns in music data through beautiful charts

All of this is powered by **KNN (K-Nearest Neighbors)** — one of the simplest yet most effective machine learning algorithms.

---

## 🎯 Features

| Feature | Description |
|---|---|
| 🤖 Mood Classification | KNN predicts mood from audio features |
| 🎵 Playlist Generator | Find K most similar songs using Euclidean distance |
| 🎛️ Custom Feature Input | Adjust sliders and get instant mood prediction |
| 🔍 Song Search | Find songs by name and discover similar ones |
| 📊 Visualizations | PCA, heatmaps, confusion matrix, mood distribution |
| ⚙️ Dynamic K | Adjust how many recommendations you want |
| 🔍 Smart Filters | Filter by language, popularity |
| 💾 Model Saving | Trained models saved with pickle |

---

## 🧠 What is KNN? (Simple Explanation)

**KNN = K-Nearest Neighbors**

Imagine you walk into a party and want to know the "vibe" (mood).
You look at the **K people closest to you** and see what mood *they* are in.
If 4 out of 5 nearby people are happy → the vibe is **happy**!

KNN works the same way with songs:

```
New Song → Calculate distance to all songs → Find K closest → Vote → Predict Mood
```

**Distance formula (Euclidean):**
```
distance = √[(v₁-v₂)² + (e₁-e₂)² + (t₁-t₂)² + ...]
```
Where v=valence, e=energy, t=tempo, etc.

**Why K matters:**
- **K too small (K=1)**: Very sensitive to noise — overfitting
- **K too large (K=100)**: Too generalized — underfitting
- **Best K**: Found by testing K=3 to K=20 and picking the most accurate

---

## 📁 Project Structure

```
music_mood_classifier/
│
├── 📂 data/
│   ├── spotify_songs.csv        ← Raw dataset (auto-generated or Kaggle)
│   └── processed_songs.csv      ← Cleaned & preprocessed data
│
├── 📂 model/
│   ├── knn_classifier.pkl       ← Trained mood classifier
│   ├── knn_recommender.pkl      ← Playlist recommender model
│   └── scaler.pkl               ← Feature normalizer
│
├── 📂 app/
│   └── streamlit_app.py         ← Web UI (Streamlit)
│
├── 📂 utils/
│   ├── generate_dataset.py      ← Synthetic data generator
│   ├── preprocessing.py         ← Data cleaning & normalization
│   └── visualizations.py        ← All plot functions
│
├── 📂 screenshots/              ← Auto-generated plot images
│
├── main.py                      ← Training pipeline (run this first!)
├── requirements.txt             ← Python dependencies
└── README.md                    ← You are here!
```

---

## 🎵 Audio Features Explained

| Feature | Range | What it means |
|---|---|---|
| **valence** | 0.0 – 1.0 | Musical positivity. 1.0 = very happy, 0.0 = very sad |
| **energy** | 0.0 – 1.0 | Intensity & activity. 1.0 = loud & fast |
| **tempo** | 60 – 200 | Beats per minute (BPM) |
| **danceability** | 0.0 – 1.0 | How suitable for dancing |
| **loudness** | -60 – 0 dB | Overall volume |
| **acousticness** | 0.0 – 1.0 | Likelihood of being acoustic |
| **instrumentalness** | 0.0 – 1.0 | No vocals → closer to 1.0 |
| **speechiness** | 0.0 – 1.0 | Presence of spoken words |
| **liveness** | 0.0 – 1.0 | Audience presence (live feel) |

---

## 🏷️ Mood Classification Logic

```
HAPPY     → High valence (≥ median) + Medium-high energy
SAD       → Low valence (< median) + Low energy (< median)
ENERGETIC → High energy (≥ median) + High tempo (≥ median)
CALM      → Low energy (< median) + High acousticness (≥ median)
```

---

## 🚀 Quick Start

### Step 1: Clone / Download the Project

```bash
# If using git:
git clone https://github.com/yourname/music-mood-classifier.git
cd music-mood-classifier

# Or just extract the ZIP and open the folder
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 3: Train the Model

```bash
python main.py
```

This will:
- ✅ Generate a synthetic Spotify-like dataset (1600 songs)
- ✅ Preprocess and normalize features
- ✅ Find the optimal K value (K=3 to K=20)
- ✅ Train KNN classifier (~92% accuracy!)
- ✅ Build playlist recommender
- ✅ Save all models to `/model/`
- ✅ Generate visualization plots to `/screenshots/`

### Step 4: Launch the Web App

```bash
streamlit run app/streamlit_app.py
```

Then open **http://localhost:8501** in your browser 🌐

---

## 🗂️ Using a Real Kaggle Dataset (Optional)

If you want to use real Spotify data instead of the synthetic dataset:

1. Go to: https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset
2. Download `dataset.csv`
3. Place it in the `data/` folder and rename to `spotify_songs.csv`
4. Make sure it has these columns: `valence`, `energy`, `tempo`, `danceability`, `loudness`, `acousticness`, `instrumentalness`
5. Run `python main.py` — it will auto-detect and use your real data!

---

## ⚙️ Training Options

```bash
# Use custom K value (skip auto-detection)
python main.py --k 7

# Generate larger dataset
python main.py --songs 1000

# Skip visualization plots (faster)
python main.py --no-plots

# Force K optimization search
python main.py --optimize-k

# Use your own dataset
python main.py --data-path data/my_songs.csv
```

---

## 📊 Model Performance

| Metric | Score |
|---|---|
| **Overall Accuracy** | ~92.5% |
| **Happy F1-Score** | 0.96 |
| **Sad F1-Score** | 0.90 |
| **Energetic F1-Score** | 0.95 |
| **Calm F1-Score** | 0.89 |
| **Best K** | 18 |

---

## 📈 Visualizations Generated

| Plot | Description |
|---|---|
| `1_mood_distribution.png` | Bar + pie chart of mood counts |
| `2_feature_correlation.png` | Heatmap of feature relationships |
| `3_pca_visualization.png` | 2D scatter plot of all songs |
| `4_confusion_matrix.png` | Model accuracy per mood |
| `5_feature_by_mood.png` | Box plots of features by mood |
| `6_k_vs_accuracy.png` | How accuracy changes with K |

---

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| **Python 3.8+** | Main language |
| **pandas** | Data loading & manipulation |
| **numpy** | Numerical operations |
| **scikit-learn** | KNN models, preprocessing |
| **matplotlib** | Plotting |
| **seaborn** | Statistical visualizations |
| **streamlit** | Web application UI |
| **pickle** | Saving/loading trained models |

---

## 💡 How the Playlist Generator Works

```
User selects mood: "happy"
        ↓
Find songs labeled "happy" → Pick anchor song (highest popularity)
        ↓
Get scaled feature vector of anchor song
        ↓
NearestNeighbors.kneighbors(anchor) → Returns K closest songs
        ↓
Apply filters (language, popularity, mood filter)
        ↓
Return top K songs as playlist ✅
```

The **key insight**: Two songs are "similar" if their feature vectors are close in Euclidean space — regardless of genre, artist, or language!

---

## 🧪 For Students: Key Concepts to Understand

1. **Why normalize features?**
   Tempo ranges from 60–200, but valence ranges from 0–1. Without normalization, tempo would dominate the distance calculation unfairly. StandardScaler fixes this by making all features have mean=0 and std=1.

2. **Why train/test split?**
   We train on 80% of data, test on 20% that the model has **never seen**. This gives us an honest estimate of real-world performance.

3. **What is the confusion matrix?**
   A grid showing which moods the model gets right vs. wrong. Perfect model = only the diagonal has values.

4. **What is PCA?**
   Principal Component Analysis reduces 9 features to 2 dimensions for visualization, while keeping as much information as possible.

---

## 🙋 FAQ

**Q: Can I use this with real Spotify songs?**
A: Yes! Use the Kaggle Spotify dataset or the Spotify API to get real audio features for any song.

**Q: Why does the model sometimes predict wrong moods?**
A: Moods can be subjective! A calm song with high acousticness but medium-high valence might genuinely be between "calm" and "happy".

**Q: How do I improve accuracy?**
A: Try more training data, tune K, or try other algorithms like Random Forest or SVM.

---

## 📄 License

MIT License — Free to use for learning and projects!

---

*Built with ❤️ for learning ML | Powered by scikit-learn & Streamlit*
