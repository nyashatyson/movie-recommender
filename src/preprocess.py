import pandas as pd


def clean_movies(movies: pd.DataFrame) -> pd.DataFrame:
    movies = movies.copy()
    movies["genres"] = movies["genres"].fillna("").replace("(no genres listed)", "")
    movies["genres_clean"] = movies["genres"].str.replace("|", " ", regex=False)
    return movies


def aggregate_tags(tags: pd.DataFrame) -> pd.DataFrame:
    tags = tags.copy()
    tags["tag"] = tags["tag"].fillna("").astype(str).str.lower().str.strip()

    grouped = (
        tags.groupby("movieId")["tag"]
        .apply(lambda x: " ".join(tag for tag in x if tag))
        .reset_index()
        .rename(columns={"tag": "tags_text"})
    )
    return grouped


def build_movie_features(movies: pd.DataFrame, tags: pd.DataFrame) -> pd.DataFrame:
    movies = clean_movies(movies)
    tag_features = aggregate_tags(tags)

    merged = movies.merge(tag_features, on="movieId", how="left")
    merged["tags_text"] = merged["tags_text"].fillna("")

    merged["content"] = (
        merged["title"].fillna("") + " "
        + merged["genres_clean"].fillna("") + " "
        + merged["tags_text"].fillna("")
    ).str.lower()

    return merged
