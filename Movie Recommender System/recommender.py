from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
from zipfile import ZipFile

import pandas as pd
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"
MOVIES_ZIP = DATA_DIR / "tmdb_5000_movies.csv.zip"
CREDITS_ZIP = DATA_DIR / "tmdb_5000_credits.csv.zip"

stemmer = PorterStemmer()


@dataclass(frozen=True)
class RecommenderModel:
    movies: pd.DataFrame
    vectorizer: CountVectorizer
    similarity_matrix: object


def _read_csv_from_zip(path: Path) -> pd.DataFrame:
    with ZipFile(path) as archive:
        csv_name = next(
            name
            for name in archive.namelist()
            if name.endswith(".csv") and not name.startswith("__MACOSX/")
        )
        with archive.open(csv_name) as csv_file:
            return pd.read_csv(csv_file)


def load_tmdb_data(
    movies_zip: Path = MOVIES_ZIP,
    credits_zip: Path = CREDITS_ZIP,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    return _read_csv_from_zip(movies_zip), _read_csv_from_zip(credits_zip)


def _parse_people_list(value: object, limit: int | None = None) -> list[str]:
    if not isinstance(value, str):
        return []

    try:
        parsed = ast.literal_eval(value)
    except (SyntaxError, ValueError):
        return []

    if not isinstance(parsed, list):
        return []

    names = [
        item.get("name", "")
        for item in parsed
        if isinstance(item, dict) and item.get("name")
    ]
    return names[:limit] if limit else names


def _parse_director(value: object) -> str:
    if not isinstance(value, str):
        return ""

    try:
        parsed = ast.literal_eval(value)
    except (SyntaxError, ValueError):
        return ""

    for item in parsed:
        if isinstance(item, dict) and item.get("job") == "Director":
            return item.get("name", "")
    return ""


def _compact_tokens(values: list[str]) -> list[str]:
    return [value.replace(" ", "") for value in values]


def _stem_text(text: str) -> str:
    return " ".join(stemmer.stem(word) for word in text.split())


def build_movie_features(movies: pd.DataFrame, credits: pd.DataFrame) -> pd.DataFrame:
    merged = movies.merge(credits, on="title")
    useful_columns = [
        "id",
        "title",
        "overview",
        "genres",
        "keywords",
        "cast",
        "crew",
        "vote_average",
        "vote_count",
        "release_date",
        "runtime",
        "popularity",
    ]
    merged = merged[useful_columns].dropna(subset=["overview"]).copy()

    merged["movie_id"] = merged["id"]
    merged["genres_list"] = merged["genres"].apply(_parse_people_list)
    merged["keywords_list"] = merged["keywords"].apply(_parse_people_list)
    merged["cast_list"] = merged["cast"].apply(lambda value: _parse_people_list(value, 3))
    merged["director"] = merged["crew"].apply(_parse_director)
    merged["release_year"] = (
        pd.to_datetime(merged["release_date"], errors="coerce")
        .dt.year.astype("Int64")
        .astype(str)
        .replace("<NA>", "Unknown")
    )

    compact_genres = merged["genres_list"].apply(_compact_tokens)
    compact_keywords = merged["keywords_list"].apply(_compact_tokens)
    compact_cast = merged["cast_list"].apply(_compact_tokens)
    compact_director = merged["director"].apply(lambda value: value.replace(" ", ""))

    merged["tags"] = (
        merged["overview"].fillna("").str.split()
        + compact_genres
        + compact_keywords
        + compact_cast
        + compact_director.apply(lambda value: [value] if value else [])
    ).apply(lambda tokens: _stem_text(" ".join(tokens)))

    return merged[
        [
            "movie_id",
            "title",
            "overview",
            "genres_list",
            "keywords_list",
            "cast_list",
            "director",
            "release_year",
            "runtime",
            "vote_average",
            "vote_count",
            "popularity",
            "tags",
        ]
    ].reset_index(drop=True)


def build_recommender_model(features: pd.DataFrame) -> RecommenderModel:
    vectorizer = CountVectorizer(max_features=5000, stop_words="english")
    vectors = vectorizer.fit_transform(features["tags"]).toarray()
    similarity_matrix = cosine_similarity(vectors)
    return RecommenderModel(
        movies=features.reset_index(drop=True),
        vectorizer=vectorizer,
        similarity_matrix=similarity_matrix,
    )


def create_recommender() -> RecommenderModel:
    movies, credits = load_tmdb_data()
    features = build_movie_features(movies, credits)
    return build_recommender_model(features)


def recommend_movies(
    title: str,
    model: RecommenderModel,
    top_n: int = 5,
) -> list[dict[str, object]]:
    matches = model.movies.index[model.movies["title"] == title].tolist()
    if not matches:
        raise ValueError(f"Movie not found: {title}")

    movie_index = matches[0]
    distances = model.similarity_matrix[movie_index]
    candidates = sorted(
        enumerate(distances),
        reverse=True,
        key=lambda item: item[1],
    )
    recommendations = [
        (index, score)
        for index, score in candidates
        if index != movie_index
    ][:top_n]

    results = []
    for index, score in recommendations:
        movie = model.movies.iloc[index].to_dict()
        movie["similarity"] = round(float(score), 4)
        results.append(movie)
    return results
