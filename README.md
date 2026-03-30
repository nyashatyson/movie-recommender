# 🎬 CineMatch – Hybrid Movie Recommendation System

A machine learning movie recommendation system built with Python, Scikit-Learn, and Streamlit.

The system combines multiple recommendation strategies to generate high-quality movie suggestions.

## Features

- Content-based recommendations
- Popularity-based ranking
- Collaborative filtering
- Hybrid recommender combining all methods
- Movie posters from the TMDB API
- Interactive UI built with Streamlit

## Recommendation Models

### Content-Based Filtering
Uses TF-IDF vectorization and cosine similarity to recommend movies with similar genres, tags, and titles.

### Popularity-Based Ranking
Uses a weighted rating formula to avoid movies with only a few ratings dominating the rankings.

### Collaborative Filtering
Uses user-user collaborative filtering with cosine similarity on the ratings matrix.

### Hybrid Recommender
Combines collaborative score, content similarity, and popularity score into one final recommendation score.

## Dataset

MovieLens ml-latest-small

Contains:
- 100k ratings
- 9k movies
- 600 users

## Tech Stack

- Python
- Pandas
- NumPy
- Scikit-Learn
- Streamlit
- TMDB API

## Installation

```bash
python -m venv .venv