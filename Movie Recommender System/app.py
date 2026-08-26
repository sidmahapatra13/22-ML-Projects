from __future__ import annotations

import math

import streamlit as st

from recommender import create_recommender, recommend_movies


st.set_page_config(
    page_title="Movie Recommender",
    page_icon="🎬",
    layout="wide",
)


def _has_value(value: object) -> bool:
    if value is None:
        return False
    if isinstance(value, float) and math.isnan(value):
        return False
    return str(value).strip() not in {"", "Unknown", "<NA>", "nan"}


def format_movie_meta(movie: dict[str, object]) -> str:
    parts = []
    release_year = movie.get("release_year")
    runtime = movie.get("runtime")
    vote_average = movie.get("vote_average")
    similarity = movie.get("similarity", 0)

    if _has_value(release_year):
        parts.append(str(release_year))
    if _has_value(runtime):
        parts.append(f"{int(float(runtime))} min")
    if _has_value(vote_average) and float(vote_average) > 0:
        parts.append(f"Rating {float(vote_average):.1f}")

    parts.append(f"{round(float(similarity) * 100):.0f}% match")
    return " | ".join(parts)


@st.cache_resource(show_spinner="Building recommender...")
def get_model():
    return create_recommender()


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        .stApp {
            background:
                radial-gradient(circle at top left, rgba(230, 57, 70, 0.16), transparent 32rem),
                linear-gradient(135deg, #101114 0%, #1d1a22 48%, #111c1f 100%);
            color: #f8f3ed;
        }

        [data-testid="stHeader"] {
            background: transparent;
        }

        [data-testid="stSidebar"] {
            background: #14161b;
            border-right: 1px solid rgba(255, 255, 255, 0.08);
        }

        .main-title {
            font-size: clamp(2.2rem, 5vw, 4.5rem);
            font-weight: 800;
            letter-spacing: 0;
            margin: 1rem 0 0.25rem;
        }

        .subtitle {
            color: #d9c7b8;
            font-size: 1.05rem;
            max-width: 54rem;
            margin-bottom: 1.5rem;
        }

        .metric-row {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 0.75rem;
            margin: 1.25rem 0 1.5rem;
        }

        .metric {
            border: 1px solid rgba(255, 255, 255, 0.09);
            background: rgba(255, 255, 255, 0.065);
            border-radius: 8px;
            padding: 0.9rem 1rem;
        }

        .metric span {
            color: #c7d6d8;
            display: block;
            font-size: 0.78rem;
            text-transform: uppercase;
        }

        .metric strong {
            display: block;
            font-size: 1.3rem;
            margin-top: 0.25rem;
        }

        .movie-card {
            min-height: 18rem;
            border: 1px solid rgba(255, 255, 255, 0.1);
            background: rgba(255, 255, 255, 0.075);
            border-radius: 8px;
            padding: 1rem;
        }

        .movie-title {
            color: #fffaf2;
            font-size: 1.08rem;
            font-weight: 750;
            line-height: 1.22;
            margin-bottom: 0.4rem;
        }

        .movie-meta {
            color: #f1c27d;
            font-size: 0.82rem;
            margin-bottom: 0.75rem;
        }

        .movie-overview {
            color: #e7dfd6;
            font-size: 0.9rem;
            line-height: 1.45;
            margin-bottom: 0.8rem;
        }

        .pill {
            display: inline-block;
            color: #d7fbff;
            background: rgba(42, 157, 143, 0.18);
            border: 1px solid rgba(42, 157, 143, 0.36);
            border-radius: 999px;
            font-size: 0.74rem;
            padding: 0.18rem 0.5rem;
            margin: 0 0.25rem 0.25rem 0;
        }

        @media (max-width: 760px) {
            .metric-row {
                grid-template-columns: 1fr;
            }

            .movie-card {
                min-height: auto;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def movie_card(movie: dict[str, object]) -> None:
    genres = movie.get("genres_list") or []
    cast = movie.get("cast_list") or []
    overview = str(movie.get("overview") or "No overview available.")
    if len(overview) > 240:
        overview = overview[:237].rstrip() + "..."

    tags = "".join(f'<span class="pill">{genre}</span>' for genre in genres[:3])
    cast_line = ", ".join(cast[:3])
    director = movie.get("director") or "Unknown"

    st.markdown(
        f"""
        <div class="movie-card">
            <div class="movie-title">{movie.get("title", "Untitled")}</div>
            <div class="movie-meta">{format_movie_meta(movie)}</div>
            <div class="movie-overview">{overview}</div>
            <div>{tags}</div>
            <div class="movie-overview"><strong>Director:</strong> {director}</div>
            <div class="movie-overview"><strong>Cast:</strong> {cast_line or "Unknown"}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    inject_styles()

    st.markdown('<h1 class="main-title">Movie Recommender</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p class="subtitle">Content-based recommendations from TMDB metadata.</p>',
        unsafe_allow_html=True,
    )

    model = get_model()
    movie_titles = model.movies["title"].sort_values().tolist()

    st.markdown(
        f"""
        <div class="metric-row">
            <div class="metric"><span>Catalog</span><strong>{len(movie_titles):,} movies</strong></div>
            <div class="metric"><span>Signals</span><strong>Genres + cast + crew</strong></div>
            <div class="metric"><span>Model</span><strong>Cosine similarity</strong></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        selected_movie = st.selectbox("Movie", movie_titles, index=movie_titles.index("Avatar"))
        result_count = st.slider("Results", min_value=3, max_value=10, value=5)
        recommend = st.button("Recommend", use_container_width=True, type="primary")

    if recommend or "recommendations" not in st.session_state:
        st.session_state.recommendations = recommend_movies(
            selected_movie,
            model,
            top_n=result_count,
        )
        st.session_state.selected_movie = selected_movie

    st.subheader(f"Because you picked {st.session_state.selected_movie}")
    columns = st.columns(3)
    for index, movie in enumerate(st.session_state.recommendations):
        with columns[index % 3]:
            movie_card(movie)


if __name__ == "__main__":
    main()
