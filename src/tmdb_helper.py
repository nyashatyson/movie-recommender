from __future__ import annotations

import requests
import streamlit as st

TMDB_SEARCH_URL = "https://api.themoviedb.org/3/search/movie"
TMDB_IMAGE_BASE = "https://image.tmdb.org/t/p/w500"


def clean_title_for_search(title: str) -> str:
    if title.endswith(")") and "(" in title:
        return title.rsplit("(", 1)[0].strip()
    return title.strip()


@st.cache_data(show_spinner=False)
def fetch_poster_url(movie_title: str) -> str | None:
    api_key = st.secrets.get("TMDB_API_KEY", None)
    if not api_key:
        return None

    query_title = clean_title_for_search(movie_title)

    try:
        response = requests.get(
            TMDB_SEARCH_URL,
            params={"api_key": api_key, "query": query_title},
            timeout=10,
        )
        response.raise_for_status()
        data = response.json()

        results = data.get("results", [])
        if not results:
            return None

        poster_path = results[0].get("poster_path")
        if not poster_path:
            return None

        return f"{TMDB_IMAGE_BASE}{poster_path}"

    except Exception:
        return None
