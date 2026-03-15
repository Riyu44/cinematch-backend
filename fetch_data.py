import requests
import pandas as pd
import os
import time
from dotenv import load_dotenv

load_dotenv()
TOKEN = os.getenv("TMDB_TOKEN")

HEADERS = {
    "Authorization": f"Bearer {TOKEN}",
    "accept": "application/json"
}

GENRE_MAP = {
    28:"Action", 12:"Adventure", 16:"Animation", 35:"Comedy",
    80:"Crime", 99:"Documentary", 18:"Drama", 10751:"Family",
    14:"Fantasy", 36:"History", 27:"Horror", 10402:"Music",
    9648:"Mystery", 10749:"Romance", 878:"Sci-Fi",
    10770:"TV Movie", 53:"Thriller", 10752:"War", 37:"Western"
}


# ── Robust GET with retry + exponential backoff ───────────────────────────────
def safe_get(url, retries=5, backoff=2.0):
    """GET with automatic retry on connection errors or 429 rate limits."""
    for attempt in range(retries):
        try:
            resp = requests.get(url, headers=HEADERS, timeout=15)
            if resp.status_code == 429:
                wait = backoff * (2 ** attempt)
                print(f"  Rate limited — waiting {wait:.0f}s before retry...")
                time.sleep(wait)
                continue
            resp.raise_for_status()
            return resp
        except (requests.exceptions.ConnectionError,
                requests.exceptions.Timeout) as e:
            wait = backoff * (2 ** attempt)
            print(f"  Connection error (attempt {attempt+1}/{retries}) — retrying in {wait:.0f}s... [{e}]")
            time.sleep(wait)
    print(f"  Failed after {retries} attempts: {url}")
    return None


# ── Fetch paginated movie list ─────────────────────────────────────────────────
def fetch_movies(total_pages=100):
    movies = []
    for page in range(1, total_pages + 1):
        url = (
            f"https://api.themoviedb.org/3/discover/movie"
            f"?include_adult=false&sort_by=popularity.desc&page={page}"
        )
        resp = safe_get(url)
        if resp is None:
            print(f"  Skipping page {page} after failed retries.")
            continue

        for m in resp.json().get("results", []):
            movies.append({
                "id":         m["id"],
                "title":      m["title"],
                "overview":   m.get("overview", ""),
                "genres":     m.get("genre_ids", []),
                "rating":     m.get("vote_average", 0),
                "votes":      m.get("vote_count", 0),
                "runtime":    0,
                "year":       m.get("release_date", "")[:4],
                "poster":     m.get("poster_path", ""),
                "popularity": m.get("popularity", 0),
            })

        print(f"Page {page}/{total_pages} — {len(movies)} movies so far")
        time.sleep(0.25)  # gentler pacing — 4 req/sec instead of 10

    return movies


# ── Fetch runtime for a single movie ──────────────────────────────────────────
def fetch_runtime(movie_id):
    url = f"https://api.themoviedb.org/3/movie/{movie_id}"
    resp = safe_get(url)
    if resp is None:
        return 0
    return resp.json().get("runtime", 0) or 0


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 50)
    print("Step 1/3 — Fetching movie list from TMDB...")
    print("=" * 50)
    movies = fetch_movies(total_pages=100)

    print("\n" + "=" * 50)
    print("Step 2/3 — Fetching runtimes & resolving genres...")
    print("  (takes ~8 mins — go make a coffee ☕)")
    print("=" * 50)

    for i, m in enumerate(movies):
        m["runtime"] = fetch_runtime(m["id"])
        m["genres"]  = [GENRE_MAP.get(g, "") for g in m["genres"]]
        m["genres"]  = ", ".join([g for g in m["genres"] if g])
        if i % 50 == 0:
            print(f"  {i}/{len(movies)} done...")
        time.sleep(0.12)   # ~8 req/sec — safe for TMDB free tier

    print("\n" + "=" * 50)
    print("Step 3/3 — Cleaning and saving...")
    print("=" * 50)

    df = pd.DataFrame(movies)
    before = len(df)
    df = df[df["votes"] > 50]        # drop obscure films
    df = df[df["overview"] != ""]    # drop empty overviews
    df = df[df["runtime"] > 0]       # drop missing runtimes
    df = df.drop_duplicates("id")
    df = df.reset_index(drop=True)

    df.to_csv("movies.csv", index=False)
    print(f"  Raw movies fetched : {before}")
    print(f"  After cleaning     : {len(df)}")
    print(f"\n✅ Done! Saved {len(df)} movies to movies.csv")
    print("   Ready for the next step.")