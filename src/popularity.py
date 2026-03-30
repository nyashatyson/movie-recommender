import pandas as pd


class PopularityRecommender:
    def __init__(self) -> None:
        self.popular_movies = None

    def fit(
        self,
        movies: pd.DataFrame,
        ratings: pd.DataFrame,
        min_votes_quantile: float = 0.90
    ) -> None:
        movie_stats = (
            ratings.groupby("movieId")["rating"]
            .agg(["mean", "count"])
            .reset_index()
            .rename(columns={"mean": "avg_rating", "count": "rating_count"})
        )

        df = movies.merge(movie_stats, on="movieId", how="left")

        df["avg_rating"] = df["avg_rating"].fillna(0)
        df["rating_count"] = df["rating_count"].fillna(0)

        C = df["avg_rating"].mean()
        m = df["rating_count"].quantile(min_votes_quantile)

        v = df["rating_count"]
        R = df["avg_rating"]

        df["weighted_score"] = ((v / (v + m)) * R) + ((m / (v + m)) * C)

        self.popular_movies = df.sort_values(
            by=["weighted_score", "rating_count", "avg_rating"],
            ascending=False
        ).reset_index(drop=True)

    def recommend(self, top_n: int = 10, genre: str | None = None) -> pd.DataFrame:
        if self.popular_movies is None:
            raise ValueError("Popularity model has not been fitted yet.")

        df = self.popular_movies.copy()

        if genre:
            genre = genre.lower().strip()
            df = df[df["genres"].str.lower().str.contains(genre, na=False)]

        columns = [
            "movieId",
            "title",
            "genres",
            "avg_rating",
            "rating_count",
            "weighted_score",
        ]
        return df[columns].head(top_n).reset_index(drop=True)
