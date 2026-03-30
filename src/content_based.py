import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class ContentRecommender:
    def __init__(self) -> None:
        self.movies = None
        self.tfidf = None
        self.tfidf_matrix = None
        self.similarity_matrix = None
        self.title_to_index = None

    def fit(self, movies_df: pd.DataFrame) -> None:
        self.movies = movies_df.reset_index(drop=True).copy()

        self.tfidf = TfidfVectorizer(stop_words="english")
        self.tfidf_matrix = self.tfidf.fit_transform(self.movies["content"])
        self.similarity_matrix = cosine_similarity(self.tfidf_matrix)

        self.title_to_index = pd.Series(
            self.movies.index,
            index=self.movies["title"].str.lower()
        ).drop_duplicates()

    def recommend(self, title: str, top_n: int = 10) -> pd.DataFrame:
        if self.movies is None or self.similarity_matrix is None:
            raise ValueError("Model has not been fitted yet.")

        title = title.lower().strip()

        if title not in self.title_to_index:
            raise ValueError(f"Movie '{title}' not found in dataset.")

        idx = self.title_to_index[title]
        similarity_scores = list(enumerate(self.similarity_matrix[idx]))
        similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)

        similarity_scores = similarity_scores[1: top_n + 1]
        movie_indices = [i[0] for i in similarity_scores]
        scores = [i[1] for i in similarity_scores]

        recommendations = self.movies.iloc[movie_indices][["movieId", "title", "genres"]].copy()
        recommendations["similarity_score"] = scores

        return recommendations.reset_index(drop=True)
