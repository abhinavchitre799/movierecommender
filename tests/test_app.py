import importlib
import sys
from pathlib import Path


def load_app(tmp_path, monkeypatch):
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    monkeypatch.setenv("MOVIE_DB_PATH", str(tmp_path / "state.db"))
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY_OVERRIDE", raising=False)
    if "app" in sys.modules:
        del sys.modules["app"]
    return importlib.import_module("app")


def test_seeded_catalog_loaded(tmp_path, monkeypatch):
    app = load_app(tmp_path, monkeypatch)
    assert app.MOVIES, "Expected seeded movies to load from DB."
    sample = app.MOVIES[0]
    assert sample["title"]
    assert sample["genres"]
    assert sample["embedding"]
    assert sample["summary_embedding"]


def test_pick_is_excluded_from_future_recs(tmp_path, monkeypatch):
    app = load_app(tmp_path, monkeypatch)
    user_id = "user1"
    picks = app.get_picks(user_id)
    recs = app.content_recs(user_id, exclude=picks, limit=8)
    assert recs, "Expected initial recommendations."
    chosen_id = recs[0]["id"]
    app.add_pick(user_id, chosen_id)
    updated_picks = app.get_picks(user_id)
    new_recs = app.content_recs(user_id, exclude=updated_picks, limit=8)
    assert chosen_id not in {m["id"] for m in new_recs}


def test_strategy_not_cold_start_with_signal(tmp_path, monkeypatch):
    app = load_app(tmp_path, monkeypatch)
    user_id = "user1"
    intent = app.stage_intent(user_id, "")
    profile = app.stage_profile(user_id)
    picks = app.get_picks(user_id)
    has_signal = bool(app.get_user_history(user_id) or picks)
    plan = app.stage_planner(intent, profile, picks, has_signal)
    assert "cold" not in plan["tools"]
    assert plan["strategy"] != "cold_start"


def test_cold_start_for_new_user(tmp_path, monkeypatch):
    app = load_app(tmp_path, monkeypatch)
    user_id = "new_user"
    intent = app.stage_intent(user_id, "")
    profile = app.stage_profile(user_id)
    picks = app.get_picks(user_id)
    has_signal = bool(app.get_user_history(user_id) or picks)
    plan = app.stage_planner(intent, profile, picks, has_signal)
    assert plan["tools"] == ["cold"]
    assert plan["strategy"] == "cold_start"
