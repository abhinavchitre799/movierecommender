# AI Movie Recommendation Agent

Chat-style movie recommender built with FastAPI + a lightweight UI. It uses summary-aware embeddings, content similarity, and adjacent-genre exploration to suggest titles, then learns from your picks.

## Features
- Strategy-aware recommendations (cold start vs content-based)
- Summary + genre similarity for richer matching
- Persistent picks stored in SQLite
- Card-based UI with quick actions

## Requirements
- Python 3.10+

## Install
```
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run
```
python3 -m uvicorn app:app --reload
```
Open http://127.0.0.1:8000

## Reset state (optional)
If you want a clean start:
```
rm -f state.db state.db-shm state.db-wal
```

## Tests
```
pytest -q
```

## Configuration
Environment variables (optional):
- `OPENAI_API_KEY`: enable embeddings + intent parsing
- `OPENAI_API_KEY_OVERRIDE`: override the API key
- `OPENAI_EMBED_MODEL`: default `text-embedding-3-small`
- `MOVIE_DB_PATH`: path to SQLite DB (default `state.db`)

## Data
Seed data lives in `data/seed.json` and is loaded into the DB on first run.

## Project Structure
- `app.py`: FastAPI backend + recommendation logic
- `templates/index.html`: frontend UI
- `data/seed.json`: movie catalog + starter user histories
- `tests/test_app.py`: unit tests
