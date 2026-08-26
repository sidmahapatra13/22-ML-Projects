from pathlib import Path
import importlib.util
import sys


APP_PATH = (
    Path(__file__).resolve().parents[1]
    / "Movie Recommender System"
    / "app.py"
)
PROJECT_PATH = APP_PATH.parent


def load_app_module():
    sys.path.insert(0, str(PROJECT_PATH))
    spec = importlib.util.spec_from_file_location("movie_recommender_app", APP_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_format_movie_meta_keeps_unknown_values_out():
    app = load_app_module()
    movie = {
        "release_year": "2001",
        "runtime": 121,
        "vote_average": 7.5,
        "similarity": 0.81234,
    }

    assert app.format_movie_meta(movie) == "2001 | 121 min | Rating 7.5 | 81% match"


def test_format_movie_meta_handles_sparse_rows():
    app = load_app_module()
    movie = {
        "release_year": "Unknown",
        "runtime": None,
        "vote_average": 0,
        "similarity": 0.0,
    }

    assert app.format_movie_meta(movie) == "0% match"
