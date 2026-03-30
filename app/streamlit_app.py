import sys
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.data_loader import load_movies, load_tags, load_ratings
from src.preprocess import build_movie_features
from src.content_based import ContentRecommender
from src.popularity import PopularityRecommender
from src.collaborative import CollaborativeRecommender
from src.hybrid import HybridRecommender
from src.ui_helpers import render_movie_grid

st.set_page_config(page_title="Movie Recommender", page_icon="🎬", layout="wide")

st.title("🎬 CineMatch")
st.caption("A portfolio movie recommender using content-based, popularity, collaborative, and hybrid methods")


@st.cache_data
def load_data():
    movies = load_movies()
    tags = load_tags()
    ratings = load_ratings()
    features = build_movie_features(movies, tags)

    movie_rating_stats = (
        ratings.groupby("movieId")["rating"]
        .agg(["mean", "count"])
        .reset_index()
        .rename(columns={"mean": "avg_rating", "count": "rating_count"})
    )

    return movies, tags, ratings, features, movie_rating_stats


@st.cache_resource
def train_models():
    movies, tags, ratings, features, movie_rating_stats = load_data()

    content_model = ContentRecommender()
    content_model.fit(features)

    popularity_model = PopularityRecommender()
    popularity_model.fit(movies, ratings)

    collaborative_model = CollaborativeRecommender()
    collaborative_model.fit(movies, ratings)

    hybrid_model = HybridRecommender(
        content_model=content_model,
        collaborative_model=collaborative_model,
        popularity_model=popularity_model,
    )
    hybrid_model.fit(movies)

    return content_model, popularity_model, collaborative_model, hybrid_model, features, ratings, movie_rating_stats


content_model, popularity_model, collaborative_model, hybrid_model, features, ratings, movie_rating_stats = train_models()

tab1, tab2, tab3, tab4 = st.tabs(
    ["Because You Liked...", "Trending", "For This User", "Best Overall Mix"]
)

with tab1:
    st.subheader("Because You Liked...")
    movie_titles = sorted(features["title"].dropna().unique().tolist())
    selected_movie = st.selectbox("Pick a movie", movie_titles, key="content_movie")
    top_n_content = st.slider("How many similar movies?", 5, 20, 10, key="content_slider")

    if st.button("Find Similar Movies"):
        results = content_model.recommend(selected_movie, top_n=top_n_content)
        results = results.merge(movie_rating_stats, on="movieId", how="left")
        render_movie_grid(
            results,
            mode="content",
            score_column="similarity_score",
            score_label="Similarity",
            extra_rating_column="avg_rating",
        )

        with st.expander("Show raw results table"):
            st.dataframe(results, use_container_width=True)

with tab2:
    st.subheader("Trending / Popular")
    genre_input = st.text_input(
        "Optional genre filter",
        placeholder="Comedy, Drama, Action...",
        key="popular_genre"
    )
    top_n_popular = st.slider("How many popular movies?", 5, 20, 10, key="popular_slider")

    if st.button("Show Popular Movies"):
        genre_filter = genre_input if genre_input.strip() else None
        results = popularity_model.recommend(top_n=top_n_popular, genre=genre_filter)
        render_movie_grid(
            results,
            mode="popular",
            score_column="weighted_score",
            score_label="Weighted score",
            extra_rating_column="avg_rating",
        )

        with st.expander("Show raw results table"):
            st.dataframe(results, use_container_width=True)

with tab3:
    st.subheader("Personalized Recommendations")
    available_users = sorted(collaborative_model.user_item_matrix.index.tolist())
    selected_user = st.selectbox("Choose a user ID", available_users, key="collab_user")
    top_n_collab = st.slider("How many recommendations?", 5, 20, 10, key="collab_slider")
    top_k_users = st.slider("How many similar users?", 5, 50, 20, key="top_k_users")

    if st.button("Recommend For This User"):
        results = collaborative_model.recommend(
            user_id=selected_user,
            top_n=top_n_collab,
            top_k_similar_users=top_k_users
        )
        results = results.merge(movie_rating_stats, on="movieId", how="left")
        render_movie_grid(
            results,
            mode="collab",
            score_column="predicted_rating",
            score_label="Predicted rating",
            extra_rating_column="avg_rating",
        )

        with st.expander("Show raw results table"):
            st.dataframe(results, use_container_width=True)

with tab4:
    st.subheader("Hybrid Recommendations")
    available_users = sorted(collaborative_model.user_item_matrix.index.tolist())
    movie_titles = sorted(features["title"].dropna().unique().tolist())

    selected_hybrid_user = st.selectbox("Choose a user ID", available_users, key="hybrid_user")
    reference_movie = st.selectbox(
        "Optional movie to guide recommendations",
        [""] + movie_titles,
        key="hybrid_movie"
    )

    top_n_hybrid = st.slider("How many hybrid recommendations?", 5, 20, 10, key="hybrid_slider")

    c1, c2, c3 = st.columns(3)
    with c1:
        collab_weight = st.slider("Collaborative", 0.0, 1.0, 0.5, 0.1)
    with c2:
        content_weight = st.slider("Content", 0.0, 1.0, 0.3, 0.1)
    with c3:
        popularity_weight = st.slider("Popularity", 0.0, 1.0, 0.2, 0.1)

    weight_sum = collab_weight + content_weight + popularity_weight
    st.write(f"Total weight: {weight_sum:.1f}")

    if st.button("Get Best Mixed Recommendations"):
        if weight_sum == 0:
            st.error("At least one weight must be greater than 0.")
        else:
            results = hybrid_model.recommend(
                user_id=selected_hybrid_user,
                reference_title=reference_movie if reference_movie else None,
                top_n=top_n_hybrid,
                collab_weight=collab_weight,
                content_weight=content_weight,
                popularity_weight=popularity_weight,
                top_k_similar_users=20,
            )
            results = results.merge(movie_rating_stats, on="movieId", how="left")
            render_movie_grid(
                results,
                mode="hybrid",
                score_column="hybrid_score",
                score_label="Hybrid score",
                extra_rating_column="avg_rating",
            )

            with st.expander("Show raw results table"):
                st.dataframe(results, use_container_width=True)
