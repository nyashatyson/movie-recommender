import re
import pandas as pd
import streamlit as st

from src.tmdb_helper import fetch_poster_url


def extract_year(title: str) -> str:
    match = re.search(r"\((\d{4})\)$", str(title).strip())
    return match.group(1) if match else "Unknown"


def render_stars(score: float, max_score: float = 5.0) -> str:
    if score is None:
        return "No rating"
    filled = round((score / max_score) * 5)
    filled = max(0, min(5, filled))
    return "★" * filled + "☆" * (5 - filled)


def recommendation_reason(row: pd.Series, mode: str) -> str:
    if mode == "content":
        return "Recommended because it is similar in genre/tags to your selected movie."
    if mode == "popular":
        return "Recommended because it is highly rated and has strong rating volume."
    if mode == "collab":
        return "Recommended because users with similar rating behavior also liked it."
    if mode == "hybrid":
        return "Recommended by combining similar users, movie similarity, and popularity."
    return "Recommended for you."


def render_movie_grid(
    df: pd.DataFrame,
    mode: str,
    score_column: str | None = None,
    score_label: str = "Score",
    extra_rating_column: str | None = None,
):
    if df.empty:
        st.warning("No results found.")
        return

    cols_per_row = 5

    for start in range(0, len(df), cols_per_row):
        row_slice = df.iloc[start:start + cols_per_row]
        cols = st.columns(cols_per_row)

        for col, (_, row) in zip(cols, row_slice.iterrows()):
            with col:
                poster_url = fetch_poster_url(row["title"])
                if poster_url:
                    st.image(poster_url, use_container_width=True)
                else:
                    st.markdown("### 🎬")
                    st.caption("No poster found")

                title = str(row["title"])
                year = extract_year(title)

                st.markdown(f"**{title}**")
                st.caption(f"{year} • {row.get('genres', 'Unknown')}")

                if extra_rating_column and extra_rating_column in row and pd.notna(row[extra_rating_column]):
                    st.write(f"{render_stars(float(row[extra_rating_column]))}  ({row[extra_rating_column]:.2f})")

                if score_column and score_column in row and pd.notna(row[score_column]):
                    try:
                        st.write(f"**{score_label}:** {float(row[score_column]):.3f}")
                    except Exception:
                        st.write(f"**{score_label}:** {row[score_column]}")

                st.caption(recommendation_reason(row, mode))
