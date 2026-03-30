import pandas as pd


class HybridRecommender:
    def __init__(self, content_model, collaborative_model, popularity_model) -> None:
        self.content_model = content_model
        self.collaborative_model = collaborative_model
        self.popularity_model = popularity_model
        self.movies = None

    def fit(self, movies: pd.DataFrame) -> None:
        self.movies = movies.copy()

    @staticmethod
    def _normalize_scores(series: pd.Series) -> pd.Series:
        series = series.copy()
        if series.empty:
            return series

        min_val = series.min()
        max_val = series.max()

        if max_val == min_val:
            return pd.Series(1.0, index=series.index)

        return (series - min_val) / (max_val - min_val)

    def recommend(
        self,
        user_id: int,
        reference_title: str | None = None,
        top_n: int = 10,
        collab_weight: float = 0.5,
        content_weight: float = 0.3,
        popularity_weight: float = 0.2,
        top_k_similar_users: int = 20
    ) -> pd.DataFrame:
        if self.movies is None:
            raise ValueError("Hybrid model has not been fitted yet.")

        collab_df = self.collaborative_model.recommend(
            user_id=user_id,
            top_n=200,
            top_k_similar_users=top_k_similar_users
        ).copy()

        collab_df["collab_score"] = self._normalize_scores(collab_df["predicted_rating"])

        pop_df = self.popularity_model.recommend(top_n=500).copy()
        pop_df["popularity_score_norm"] = self._normalize_scores(pop_df["weighted_score"])

        hybrid_df = collab_df.merge(
            pop_df[["movieId", "weighted_score", "popularity_score_norm"]],
            on="movieId",
            how="left"
        )

        if reference_title and reference_title.strip():
            content_df = self.content_model.recommend(reference_title, top_n=200).copy()
            content_df["content_score"] = self._normalize_scores(content_df["similarity_score"])

            hybrid_df = hybrid_df.merge(
                content_df[["movieId", "content_score"]],
                on="movieId",
                how="left"
            )
        else:
            hybrid_df["content_score"] = 0.0

        hybrid_df["content_score"] = hybrid_df["content_score"].fillna(0)
        hybrid_df["popularity_score_norm"] = hybrid_df["popularity_score_norm"].fillna(0)

        hybrid_df["hybrid_score"] = (
            collab_weight * hybrid_df["collab_score"] +
            content_weight * hybrid_df["content_score"] +
            popularity_weight * hybrid_df["popularity_score_norm"]
        )

        hybrid_df = hybrid_df.sort_values("hybrid_score", ascending=False)

        return hybrid_df[
            [
                "movieId",
                "title",
                "genres",
                "predicted_rating",
                "collab_score",
                "content_score",
                "weighted_score",
                "popularity_score_norm",
                "hybrid_score",
            ]
        ].head(top_n).reset_index(drop=True)
