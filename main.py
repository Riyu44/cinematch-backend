import os
import pickle
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

# ── App setup ─────────────────────────────────────────────────────────────────
app = FastAPI(title="CineMatch API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten this in production
    allow_methods=["*"],
    allow_headers=["*"],
)

app = FastAPI(title="CineMatch API", version="1.0")

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ── This must respond BEFORE models load ─────────────
@app.get("/health")
def health():
    return {"status": "ok"}

# ── Load all models and data at startup (runs once) ───────────────────────────
print("Loading models and data...")

df          = pd.read_pickle("movies_enriched.pkl")
embeddings  = np.load("embeddings.npy")
movie_factors = np.load("movie_factors.npy")

with open("knn_model.pkl",   "rb") as f: knn    = pickle.load(f)
with open("kmeans_model.pkl","rb") as f: kmeans = pickle.load(f)
with open("mlb.pkl",         "rb") as f: mlb    = pickle.load(f)
with open("scaler.pkl",      "rb") as f: scaler = pickle.load(f)

sentence_model = SentenceTransformer("all-MiniLM-L6-v2")

TMDB_IMAGE_BASE = "https://image.tmdb.org/t/p/w500"

print(f"✅ Ready — {len(df)} movies loaded.")


# ── Request / Response schemas ────────────────────────────────────────────────
class RecommendRequest(BaseModel):
    watched_titles: List[str]          # e.g. ["Inception", "The Dark Knight"]
    genres:         List[str]          # e.g. ["Action", "Thriller"]
    max_runtime:    Optional[int] = None   # minutes, e.g. 120  (None = no limit)
    min_runtime:    Optional[int] = None   # e.g. 60
    min_rating:     Optional[float] = 5.0
    top_n:          Optional[int] = 10    # always default to 10

class Movie(BaseModel):
    id:          int
    title:       str
    genres:      str
    rating:      float
    runtime:     int
    year:        str
    poster_url:  str
    popularity:  float
    why:         str       # plain-English reason for recommendation
    match_score: float     # 0–100 composite score

class RecommendResponse(BaseModel):
    recommendations: List[Movie]
    total_candidates: int   # how many movies were in the filtered pool
    message:          str


# ── Helper: resolve movie indices from titles ─────────────────────────────────
def resolve_indices(titles: List[str]) -> List[int]:
    """Fuzzy match watched titles to dataframe rows."""
    indices = []
    for title in titles:
        match = df[df["title"].str.lower() == title.strip().lower()]
        if match.empty:
            # try partial match if exact fails
            match = df[df["title"].str.lower().str.contains(title.strip().lower(), na=False)]
        if not match.empty:
            indices.append(match.index[0])
    return indices


# ── Helper: KNN scores ────────────────────────────────────────────────────────
def get_knn_scores(watched_indices: List[int], candidate_indices: List[int]) -> np.ndarray:
    """
    Average the embeddings of watched movies → find cosine similarity
    against every candidate. Returns array of shape (len(candidate_indices),).
    """
    if not watched_indices:
        return np.zeros(len(candidate_indices))

    avg_embedding = embeddings[watched_indices].mean(axis=0, keepdims=True)
    candidate_embs = embeddings[candidate_indices]

    # cosine similarity = dot product of unit vectors
    avg_norm  = avg_embedding  / (np.linalg.norm(avg_embedding)  + 1e-9)
    cand_norm = candidate_embs / (np.linalg.norm(candidate_embs, axis=1, keepdims=True) + 1e-9)
    similarities = cand_norm @ avg_norm.T   # shape (n_candidates, 1)
    return similarities.flatten()


# ── Helper: SVD collaborative filter scores ───────────────────────────────────
def get_svd_scores(watched_indices: List[int], candidate_indices: List[int]) -> np.ndarray:
    """
    Project the user into latent factor space via their watched movies.
    Cosine similarity against candidate movie factors.
    Cold start: returns zeros if fewer than 2 watched movies found.
    """
    if len(watched_indices) < 2:
        return np.zeros(len(candidate_indices))

    # User taste vector = mean of watched movie latent factors
    user_vector = movie_factors[watched_indices].mean(axis=0, keepdims=True)

    candidate_factors = movie_factors[candidate_indices]
    u_norm = user_vector        / (np.linalg.norm(user_vector)                       + 1e-9)
    c_norm = candidate_factors  / (np.linalg.norm(candidate_factors, axis=1, keepdims=True) + 1e-9)
    scores = c_norm @ u_norm.T
    return scores.flatten()


# ── Helper: cluster proximity scores ─────────────────────────────────────────
def get_cluster_scores(watched_indices: List[int], candidate_indices: List[int]) -> np.ndarray:
    """
    Score candidates by how many share the same cluster as the watched movies.
    Candidates in the majority cluster of watched movies score 1.0, others 0.0.
    """
    if not watched_indices:
        return np.zeros(len(candidate_indices))

    watched_clusters = df.loc[watched_indices, "cluster"].tolist()
    # most common cluster among watched movies
    majority_cluster = max(set(watched_clusters), key=watched_clusters.count)

    candidate_clusters = df.loc[candidate_indices, "cluster"].values
    return (candidate_clusters == majority_cluster).astype(float)


# ── Helper: build "why" label ─────────────────────────────────────────────────
def build_why(knn_score: float, svd_score: float, cluster_score: float,
              watched_titles: List[str]) -> str:
    title_str = " & ".join(watched_titles[:2]) if watched_titles else "your selections"
    if knn_score > 0.6 and svd_score > 0.4:
        return f"Strong match — similar to {title_str} in both content and taste profile"
    elif knn_score > 0.5:
        return f"Similar themes and tone to {title_str}"
    elif svd_score > 0.4:
        return f"Loved by viewers with the same taste profile as you"
    elif cluster_score == 1.0:
        return "Fits your genre and runtime preference"
    else:
        return f"Recommended based on your interest in {title_str}"


# ── Main recommendation logic ─────────────────────────────────────────────────
def recommend(req: RecommendRequest) -> RecommendResponse:

    # 1. Resolve watched titles → indices
    watched_indices = resolve_indices(req.watched_titles)

    # 2. Build genre filter text for embedding query
    genre_text = ", ".join(req.genres) if req.genres else ""

    # 3. K-Means candidate filtering
    #    — If genres specified: keep movies matching ANY requested genre
    #    — Also apply runtime + rating filters
    mask = pd.Series([True] * len(df), index=df.index)

    if req.genres:
        genre_mask = df["genres"].apply(
            lambda g: any(genre.lower() in g.lower() for genre in req.genres)
        )
        mask = mask & genre_mask

    if req.min_rating is not None:
        mask = mask & (df["rating"] >= req.min_rating)

    if req.max_runtime is not None:
        mask = mask & (df["runtime"] <= req.max_runtime) & (df["runtime"] > 0)

    if req.min_runtime is not None:
        mask = mask & (df["runtime"] >= req.min_runtime)

    # Exclude movies the user has already watched
    if watched_indices:
        mask = mask & (~df.index.isin(watched_indices))

    candidate_df = df[mask].copy()

    # Fallback: if genre filter is too strict and returns < 20 results,
    # drop genre filter and keep only runtime + rating constraints
    if len(candidate_df) < 20:
        mask2 = pd.Series([True] * len(df), index=df.index)
        if req.min_rating   is not None: mask2 = mask2 & (df["rating"]  >= req.min_rating)
        if req.max_runtime  is not None: mask2 = mask2 & (df["runtime"] <= req.max_runtime) & (df["runtime"] > 0)
        if req.min_runtime  is not None: mask2 = mask2 & (df["runtime"] >= req.min_runtime)
        if watched_indices:              mask2 = mask2 & (~df.index.isin(watched_indices))
        candidate_df = df[mask2].copy()

    candidate_indices = candidate_df.index.tolist()
    total_candidates  = len(candidate_indices)

    if total_candidates == 0:
        return RecommendResponse(
            recommendations=[],
            total_candidates=0,
            message="No movies matched your filters. Try relaxing genre or runtime constraints."
        )

    # 4. Score all candidates with all 3 models
    knn_scores     = get_knn_scores(watched_indices, candidate_indices)
    svd_scores     = get_svd_scores(watched_indices, candidate_indices)
    cluster_scores = get_cluster_scores(watched_indices, candidate_indices)

    # 5. Dynamic weights based on how many watched movies we resolved
    n_watched = len(watched_indices)
    if n_watched == 0:
        w_knn, w_svd, w_cluster = 0.0, 0.0, 1.0   # pure popularity/genre filter
    elif n_watched < 2:
        w_knn, w_svd, w_cluster = 0.7, 0.1, 0.2   # KNN dominates, cold start
    else:
        w_knn, w_svd, w_cluster = 0.5, 0.3, 0.2   # full ensemble

    composite = (
        w_knn     * knn_scores     +
        w_svd     * svd_scores     +
        w_cluster * cluster_scores
    )

    # 6. Rank and take top_n
    top_n = min(req.top_n or 10, total_candidates)
    top_indices_local = np.argsort(composite)[::-1][:top_n]   # descending

    # 7. Build response
    recommendations = []
    for local_idx in top_indices_local:
        global_idx = candidate_indices[local_idx]
        row = df.loc[global_idx]

        knn_s     = float(knn_scores[local_idx])
        svd_s     = float(svd_scores[local_idx])
        cluster_s = float(cluster_scores[local_idx])
        score     = float(composite[local_idx])

        recommendations.append(Movie(
            id          = int(row["id"]),
            title       = str(row["title"]),
            genres      = str(row["genres"]),
            rating      = round(float(row["rating"]), 1),
            runtime     = int(row["runtime"]),
            year        = str(row["year"]),
            poster_url  = f"{TMDB_IMAGE_BASE}{row['poster']}" if row["poster"] else "",
            popularity  = round(float(row["popularity"]), 1),
            why         = build_why(knn_s, svd_s, cluster_s, req.watched_titles),
            match_score = round(score * 100, 1),
        ))

    n_resolved = len(watched_indices)
    unresolved = [t for t in req.watched_titles
                  if t.strip().lower() not in
                  df["title"].str.lower().values]

    msg_parts = [f"Returning {len(recommendations)} recommendations from {total_candidates} candidates."]
    if n_resolved == 0:
        msg_parts.append("None of your watched titles were found — showing genre/filter matches only.")
    elif unresolved:
        msg_parts.append(f"Could not find: {', '.join(unresolved)}. Try exact TMDB titles.")

    return RecommendResponse(
        recommendations  = recommendations,
        total_candidates = total_candidates,
        message          = " ".join(msg_parts),
    )


# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {"status": "CineMatch API is running", "movies": len(df)}


@app.get("/movies/search")
def search_movies(q: str, limit: int = 10):
    """Search movies by title — used by the app's search-as-you-type input."""
    if not q or len(q) < 2:
        return {"results": []}
    matches = df[df["title"].str.lower().str.contains(q.lower(), na=False)]
    results = []
    for _, row in matches.head(limit).iterrows():
        results.append({
            "id":         int(row["id"]),
            "title":      str(row["title"]),
            "year":       str(row["year"]),
            "rating":     round(float(row["rating"]), 1),
            "genres":     str(row["genres"]),
            "poster_url": f"{TMDB_IMAGE_BASE}{row['poster']}" if row["poster"] else "",
        })
    return {"results": results}


@app.get("/movies/genres")
def list_genres():
    """Return all unique genres — used to populate the genre picker in the app."""
    all_genres = set()
    for g in df["genres"].dropna():
        for genre in g.split(","):
            genre = genre.strip()
            if genre:
                all_genres.add(genre)
    return {"genres": sorted(all_genres)}


@app.post("/recommend", response_model=RecommendResponse)
def get_recommendations(req: RecommendRequest):
    """
    Main recommendation endpoint.

    Body example:
    {
      "watched_titles": ["Inception", "The Dark Knight", "Interstellar"],
      "genres": ["Action", "Thriller"],
      "max_runtime": 150,
      "min_runtime": 80,
      "min_rating": 6.0,
      "top_n": 10
    }
    """
    try:
        return recommend(req)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Run locally ───────────────────────────────────────────────────────────────
# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)