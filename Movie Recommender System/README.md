# Movie Recommender System

Content-based movie recommendation app built with Streamlit. It uses the TMDB 5000 movie metadata already included in `data/`, builds tags from overview, genres, keywords, top cast, and director, then ranks similar movies with bag-of-words vectors and cosine similarity.

## Run

```bash
cd "Movie Recommender System"
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/streamlit run app.py
```

## Files

- `app.py`: Streamlit frontend.
- `recommender.py`: reusable recommendation logic.
- `movie-recommender-system.ipynb`: notebook exploration and original model workflow.
- `data/`: zipped TMDB CSV files used by the app.
