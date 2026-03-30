from pathlib import Path
import pandas as pd

DATA_DIR = Path("data/raw")


def load_movies() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "movies.csv")


def load_ratings() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "ratings.csv")


def load_tags() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "tags.csv")


def load_links() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "links.csv")
