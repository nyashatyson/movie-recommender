import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


class CollaborativeRecommender:
    def __init__(self) -> None:
        self.user_item_matrix = None
        self.user_similarity = None
        self.movies = None
        self.ratings = None

    def fit(self, movies: pd.DataFrame, ratings: pd.DataFrame, min_ratings_per_user: int = 5) -> None:
        self.movies = movies.copy()
        self.ratings = ratings.copy()

        user_counts = ratings["userId"].value_counts()
        active_users = user_counts[user_counts >= min_ratings_per_user].index

        filtered_ratings = ratings[ratings["userId"].isin(active_users)].copy()

        self.user_item_matrix = filtered_ratings.pivot_table(
            index="userId",
            columns="movieId",
            values="rating"
        )

        user_item_filled = self.user_item_matrix.fillna(0)

        self.user_similarity = pd.DataFrame(
            cosine_similarity(user_item_filled),
            index=self.user_item_matrix.index,
            columns=self.user_item_matrix.index
        )

    def predict_scores_for_user(self, user_id: int, top_k_similar_users: int = 20) -> pd.Series:
        if self.user_item_matrix is None or self.user_similarity is None:
            raise ValueError("Collaborative model has not been fitted yet.")

        if user_id not in self.user_item_matrix.index:
            raise ValueError(f"User {user_id} not found in training data.")

        user_ratings = self.user_item_matrix.loc[user_id]

        similar_users = self.user_similarity[user_id].drop(index=user_id)
        similar_users = similar_users.sort_values(ascending=False).head(top_k_similar_users)

        similar_user_ids = similar_users.index
        similarity_weights = similar_users.values

        similar_users_ratings = self.user_item_matrix.loc[similar_user_ids]

        weighted_sum = np.dot(similarity_weights, similar_users_ratings.fillna(0).values)

        rated_mask = (~similar_users_ratings.isna()).astype(int)
        weight_sums = np.dot(similarity_weights, rated_mask.values)

        predicted_scores = np.divide(
            weighted_sum,
            weight_sums,
            out=np.zeros_like(weighted_sum, dtype=float),
            where=weight_sums != 0
        )

        predicted_scores = pd.Series(predicted_scores, index=self.user_item_matrix.columns)

        unseen_movies = user_ratings[user_ratings.isna()].index
        predicted_scores = predicted_scores.loc[unseen_movies]

        return predicted_scores.sort_values(ascending=False)

    def recommend(self, user_id: int, top_n: int = 10, top_k_similar_users: int = 20) -> pd.DataFrame:
        predicted_scores = self.predict_scores_for_user(
            user_id=user_id,
            top_k_similar_users=top_k_similar_users
        )

        top_recommendations = predicted_scores.head(top_n).reset_index()
        top_recommendations.columns = ["movieId", "predicted_rating"]

        results = top_recommendations.merge(
            self.movies[["movieId", "title", "genres"]],
            on="movieId",
            how="left"
        )

        return results[["movieId", "title", "genres", "predicted_rating"]]
