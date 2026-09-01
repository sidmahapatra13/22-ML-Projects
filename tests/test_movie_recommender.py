from pathlib import Path
import importlib.util
import sys

import pandas as pd


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "Movie Recommender System"
    / "recommender.py"
)


def load_recommender_module():
    spec = importlib.util.spec_from_file_location("movie_recommender", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_build_movie_features_combines_tmdb_metadata():
    recommender = load_recommender_module()
    movies = pd.DataFrame(
        [
            {
                "id": 1,
                "title": "Space Quest",
                "overview": "A crew explores a distant galaxy.",
                "genres": '[{"name": "Science Fiction"}, {"name": "Adventure"}]',
                "keywords": '[{"name": "space travel"}, {"name": "found family"}]',
                "vote_average": 7.5,
                "vote_count": 100,
                "release_date": "2001-07-20",
                "runtime": 121,
                "popularity": 20.1,
            },
            {
                "id": 2,
                "title": "Quiet Drama",
                "overview": "A family reconnects in a small town.",
                "genres": '[{"name": "Drama"}]',
                "keywords": '[{"name": "family"}]',
                "vote_average": 6.4,
                "vote_count": 50,
                "release_date": "1999-01-01",
                "runtime": 98,
                "popularity": 4.2,
            },
        ]
    )
    credits = pd.DataFrame(
        [
            {
                "movie_id": 1,
                "title": "Space Quest",
                "cast": '[{"name": "Ada Star"}, {"name": "Leo Moon"}, {"name": "Nia Ray"}, {"name": "Extra Person"}]',
                "crew": '[{"job": "Director", "name": "Mira Nova"}]',
            },
            {
                "movie_id": 2,
                "title": "Quiet Drama",
                "cast": '[{"name": "Pat Stone"}]',
                "crew": '[{"job": "Director", "name": "Rae Fields"}]',
            },
        ]
    )

    features = recommender.build_movie_features(movies, credits)

    assert list(features["title"]) == ["Space Quest", "Quiet Drama"]
    assert features.loc[0, "genres_list"] == ["Science Fiction", "Adventure"]
    assert features.loc[0, "cast_list"] == ["Ada Star", "Leo Moon", "Nia Ray"]
    assert features.loc[0, "director"] == "Mira Nova"
    assert "sciencefict" in features.loc[0, "tags"]
    assert "miranova" in features.loc[0, "tags"]
    assert features.loc[0, "release_year"] == "2001"


def test_recommend_movies_returns_similar_titles_with_scores():
    recommender = load_recommender_module()
    features = pd.DataFrame(
        [
            {
                "movie_id": 1,
                "title": "Space Quest",
                "overview": "space crew galaxy adventure",
                "genres_list": ["Science Fiction", "Adventure"],
                "keywords_list": ["space travel"],
                "cast_list": ["Ada Star"],
                "director": "Mira Nova",
                "release_year": "2001",
                "runtime": 121,
                "vote_average": 7.5,
                "vote_count": 100,
                "popularity": 20.1,
                "tags": "space crew galaxy adventure ScienceFiction Adventure spacetravel AdaStar MiraNova",
            },
            {
                "movie_id": 2,
                "title": "Galaxy Run",
                "overview": "space mission galaxy rescue",
                "genres_list": ["Science Fiction", "Adventure"],
                "keywords_list": ["space travel"],
                "cast_list": ["Ada Star"],
                "director": "Mira Nova",
                "release_year": "2002",
                "runtime": 110,
                "vote_average": 7.1,
                "vote_count": 90,
                "popularity": 18.0,
                "tags": "space mission galaxy rescue ScienceFiction Adventure spacetravel AdaStar MiraNova",
            },
            {
                "movie_id": 3,
                "title": "Kitchen Hearts",
                "overview": "restaurant romance family",
                "genres_list": ["Romance"],
                "keywords_list": ["cooking"],
                "cast_list": ["Lee Cook"],
                "director": "Morgan Table",
                "release_year": "2003",
                "runtime": 100,
                "vote_average": 6.8,
                "vote_count": 70,
                "popularity": 6.0,
                "tags": "restaurant romance family Romance cooking LeeCook MorganTable",
            },
        ]
    )

    model = recommender.build_recommender_model(features)
    recommendations = recommender.recommend_movies("Space Quest", model, top_n=2)

    assert recommendations[0]["title"] == "Galaxy Run"
    assert recommendations[0]["similarity"] > recommendations[1]["similarity"]
    assert recommendations[1]["title"] == "Kitchen Hearts"


def test_recommend_movies_raises_for_unknown_title():
    recommender = load_recommender_module()
    features = pd.DataFrame(
        [
            {
                "movie_id": 1,
                "title": "Space Quest",
                "overview": "space crew",
                "genres_list": ["Science Fiction"],
                "keywords_list": ["space"],
                "cast_list": ["Ada Star"],
                "director": "Mira Nova",
                "release_year": "2001",
                "runtime": 121,
                "vote_average": 7.5,
                "vote_count": 100,
                "popularity": 20.1,
                "tags": "space crew ScienceFiction AdaStar MiraNova",
            }
        ]
    )
    model = recommender.build_recommender_model(features)

    try:
        recommender.recommend_movies("Missing Movie", model)
    except ValueError as exc:
        assert "Missing Movie" in str(exc)
    else:
        raise AssertionError("Expected ValueError for an unknown movie title")


def test_build_movie_features_does_not_cross_join_duplicate_titles():
    recommender = load_recommender_module()
    movies = pd.DataFrame(
        [
            {
                "id": 10,
                "title": "Batman",
                "overview": "A caped hero patrols Gotham.",
                "genres": '[{"name": "Action"}]',
                "keywords": '[{"name": "vigilante"}]',
                "vote_average": 7.2,
                "vote_count": 200,
                "release_date": "1989-06-23",
                "runtime": 126,
                "popularity": 30.0,
            },
            {
                "id": 11,
                "title": "Batman",
                "overview": "A campier caped hero patrols Gotham.",
                "genres": '[{"name": "Comedy"}]',
                "keywords": '[{"name": "camp"}]',
                "vote_average": 6.1,
                "vote_count": 80,
                "release_date": "1966-07-30",
                "runtime": 105,
                "popularity": 9.0,
            },
        ]
    )
    credits = pd.DataFrame(
        [
            {
                "movie_id": 10,
                "title": "Batman",
                "cast": '[{"name": "Michael Dark"}]',
                "crew": '[{"job": "Director", "name": "Tim Gothic"}]',
            },
            {
                "movie_id": 11,
                "title": "Batman",
                "cast": '[{"name": "Adam Bright"}]',
                "crew": '[{"job": "Director", "name": "Leslie Camp"}]',
            },
        ]
    )

    features = recommender.build_movie_features(movies, credits)

    # Joining on title would fan two rows out into four.
    assert len(features) == 2
    by_id = features.set_index("movie_id")
    assert by_id.loc[10, "director"] == "Tim Gothic"
    assert by_id.loc[10, "cast_list"] == ["Michael Dark"]
    assert by_id.loc[11, "director"] == "Leslie Camp"
    assert by_id.loc[11, "cast_list"] == ["Adam Bright"]
