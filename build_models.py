import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import MultiLabelBinarizer, MinMaxScaler
from sentence_transformers import SentenceTransformer
import pickle
import os

print("=" * 50)
print("CineMatch — Model Builder")
print("=" * 50)

# ── Load data ─────────────────────────────────────────
print("\n[1/7] Loading movies.csv...")
df = pd.read_csv("movies.csv")
df["overview"]  = df["overview"].fillna("")
df["genres"]    = df["genres"].fillna("")
df["runtime"]   = df["runtime"].fillna(0).astype(int)
df["rating"]    = df["rating"].fillna(0).astype(float)
df["year"]      = df["year"].fillna("0").astype(str)
df = df.reset_index(drop=True)
print(f"    Loaded {len(df)} movies.")


# ── MODEL 1: Sentence embeddings → KNN ───────────────
print("\n[2/7] Building sentence embeddings (takes 2-4 mins)...")
model_st = SentenceTransformer("all-MiniLM-L6-v2")

# Combine title + genres + overview into one rich text per movie
df["text"] = (
    df["title"] + ". " +
    "Genres: " + df["genres"] + ". " +
    df["overview"]
)

embeddings = model_st.encode(
    df["text"].tolist(),
    show_progress_bar=True,
    batch_size=64
)
np.save("embeddings.npy", embeddings)
print(f"    Embeddings shape: {embeddings.shape}")


# ── MODEL 2: KNN on embeddings ────────────────────────
print("\n[3/7] Fitting KNN model...")
knn = NearestNeighbors(n_neighbors=20, metric="cosine", algorithm="brute")
knn.fit(embeddings)
with open("knn_model.pkl", "wb") as f:
    pickle.dump(knn, f)
print("    KNN fitted and saved.")


# ── MODEL 3: K-Means clustering ───────────────────────
print("\n[4/7] Fitting K-Means clusters...")

# Build genre feature matrix
mlb = MultiLabelBinarizer()
genre_list = df["genres"].apply(lambda x: [g.strip() for g in x.split(",") if g.strip()])
genre_matrix = mlb.fit_transform(genre_list)

# Combine genre + normalised rating + normalised runtime
scaler = MinMaxScaler()
num_features = scaler.fit_transform(df[["rating", "runtime", "popularity"]].fillna(0))
cluster_features = np.hstack([genre_matrix, num_features])

kmeans = KMeans(n_clusters=20, random_state=42, n_init=10)
df["cluster"] = kmeans.fit_predict(cluster_features)

with open("kmeans_model.pkl", "wb") as f:
    pickle.dump(kmeans, f)
with open("mlb.pkl", "wb") as f:
    pickle.dump(mlb, f)
with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

# Print cluster summaries so you can see what each cluster represents
print("    Cluster summaries:")
for c in range(20):
    top = df[df["cluster"] == c].nlargest(3, "popularity")["title"].tolist()
    print(f"      Cluster {c:02d}: {', '.join(top)}")


# ── MODEL 4: Collaborative filtering (TruncatedSVD) ──
print("\n[5/7] Building collaborative filter...")

# Simulate a user-item matrix from ratings + popularity
# In production this would be real user ratings — here we use
# TMDB rating × log(votes) as a proxy for "community preference strength"
df["cf_score"] = df["rating"] * np.log1p(df["votes"])

# Build a synthetic user-item matrix: 500 virtual users × all movies
# Each virtual user has affinity for a random subset of clusters
np.random.seed(42)
n_users  = 500
n_movies = len(df)
user_item = np.zeros((n_users, n_movies))

for u in range(n_users):
    # Each user likes 2-4 random clusters
    liked_clusters = np.random.choice(20, size=np.random.randint(2, 5), replace=False)
    for c in liked_clusters:
        movie_indices = df[df["cluster"] == c].index.tolist()
        for idx in movie_indices:
            # Score = cf_score + small noise to differentiate users
            user_item[u, idx] = df.loc[idx, "cf_score"] + np.random.normal(0, 0.5)

user_item = np.clip(user_item, 0, None)  # no negative scores

# Fit SVD
svd = TruncatedSVD(n_components=50, random_state=42)
user_factors  = svd.fit_transform(user_item)
movie_factors = svd.components_.T   # shape: (n_movies, 50)

np.save("user_factors.npy",  user_factors)
np.save("movie_factors.npy", movie_factors)
with open("svd_model.pkl", "wb") as f:
    pickle.dump(svd, f)
print(f"    SVD fitted — {svd.explained_variance_ratio_.sum():.1%} variance explained.")


# ── Save enriched dataframe ───────────────────────────
print("\n[6/7] Saving enriched movie data...")
df.to_csv("movies_enriched.csv", index=False)
df.to_pickle("movies_enriched.pkl")   # faster to load in FastAPI
print(f"    Saved {len(df)} movies with cluster labels.")


# ── Sanity check ─────────────────────────────────────
print("\n[7/7] Sanity check — recommendations for 'The Dark Knight'...")
target = df[df["title"].str.contains("Dark Knight", case=False)]
if not target.empty:
    idx = target.index[0]
    distances, indices = knn.kneighbors([embeddings[idx]])
    recs = df.iloc[indices[0][1:6]]["title"].tolist()
    print(f"    Input : The Dark Knight")
    print(f"    Top 5 : {recs}")
else:
    print("    'The Dark Knight' not in dataset — try another title manually.")

print("\n" + "=" * 50)
print("✅ All models built and saved!")
print("   Files created:")
print("   - embeddings.npy      (sentence embeddings)")
print("   - knn_model.pkl       (KNN on embeddings)")
print("   - kmeans_model.pkl    (K-Means clusters)")
print("   - svd_model.pkl       (TruncatedSVD collab filter)")
print("   - movie_factors.npy   (movie latent vectors)")
print("   - user_factors.npy    (user latent vectors)")
print("   - movies_enriched.pkl (enriched dataframe)")
print("=" * 50)
print("\nNext step: run main.py to start the FastAPI server.")