import json
import math
import os
import random
import sqlite3
import threading
from pathlib import Path
from typing import Dict, List, Optional, Tuple, TypedDict

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from langgraph.graph import END, StateGraph

app = FastAPI(title="Movie Recommendation Agent")
templates = Jinja2Templates(directory="templates")

DB_PATH = Path(os.getenv("MOVIE_DB_PATH", str(Path(__file__).resolve().parent / "state.db")))
SEED_PATH = Path(__file__).resolve().parent / "data" / "seed.json"
DB_LOCK = threading.Lock()
DB_CONN: Optional[sqlite3.Connection] = None

EMBED_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")
OPENAI_API_KEY_OVERRIDE = os.getenv("OPENAI_API_KEY_OVERRIDE")

random.seed(42)

# ------------------------
# Data
# ------------------------
GENRE_BASE: Dict[str, List[float]] = {
    "suspense": [0.9, 0.12, 0.1, 0.08],
    "thriller": [0.88, 0.14, 0.12, 0.1],
    "crime": [0.86, 0.15, 0.1, 0.1],
    "psychological": [0.89, 0.12, 0.14, 0.07],
    "sci-fi": [0.45, 0.18, 0.75, 0.12],
    "comedy": [0.08, 0.9, 0.1, 0.06],
    "romance": [0.12, 0.88, 0.15, 0.06],
    "drama": [0.45, 0.42, 0.25, 0.15],
    "action": [0.48, 0.15, 0.28, 0.75],
}

ADJACENT = {
    "suspense": ["thriller", "crime", "psychological", "drama"],
    "thriller": ["suspense", "action", "crime", "sci-fi"],
    "crime": ["thriller", "suspense", "drama"],
    "psychological": ["suspense", "thriller", "drama"],
    "sci-fi": ["thriller", "action", "suspense"],
    "comedy": ["romance", "drama"],
    "romance": ["comedy", "drama"],
    "drama": ["suspense", "thriller", "romance"],
    "action": ["thriller", "sci-fi"],
}

MOVIES: List[Dict[str, object]] = []
MOVIE_BY_ID: Dict[str, Dict[str, object]] = {}


def jitter(vec: List[float], scale: float = 0.04) -> List[float]:
    return [max(0.0, min(1.0, v + random.uniform(-scale, scale))) for v in vec]


def get_openai_client():
    key = OPENAI_API_KEY_OVERRIDE or os.getenv("OPENAI_API_KEY")
    if not key:
        return None
    try:
        from openai import OpenAI
    except Exception:
        return None
    try:
        return OpenAI(api_key=key)
    except Exception:
        return None


def embed_summary(text: str, base_vec: List[float]) -> List[float]:
    client = get_openai_client()
    if client:
        try:
            resp = client.embeddings.create(model=EMBED_MODEL, input=text)
            return resp.data[0].embedding[: len(base_vec)]
        except Exception:
            pass
    return jitter(base_vec, scale=0.06)


def load_seed_data() -> Dict[str, object]:
    if not SEED_PATH.exists():
        return {"movies": [], "users": []}
    with SEED_PATH.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def refresh_catalog():
    global MOVIES, MOVIE_BY_ID
    MOVIES = load_movies()
    MOVIE_BY_ID = {m["id"]: m for m in MOVIES}

# ------------------------
# DB
# ------------------------


def get_db():
    global DB_CONN
    if DB_CONN is None:
        DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        DB_CONN = sqlite3.connect(DB_PATH, check_same_thread=False)
        DB_CONN.execute("PRAGMA journal_mode=WAL;")
        DB_CONN.execute("PRAGMA synchronous=NORMAL;")
    return DB_CONN


def init_db():
    conn = get_db()
    with DB_LOCK:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS movies (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                genres TEXT NOT NULL,
                summary TEXT,
                embedding TEXT,
                summary_embedding TEXT
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id TEXT PRIMARY KEY
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS user_history (
                user_id TEXT NOT NULL,
                movie_id TEXT NOT NULL,
                PRIMARY KEY (user_id, movie_id)
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS picks (
                user_id TEXT NOT NULL,
                movie_id TEXT NOT NULL,
                PRIMARY KEY (user_id, movie_id)
            )
            """
        )
        conn.commit()
    seed_db_if_empty()
    refresh_catalog()


def load_movies() -> List[Dict[str, object]]:
    conn = get_db()
    with DB_LOCK:
        rows = conn.execute(
            "SELECT id, title, genres, summary, embedding, summary_embedding FROM movies"
        ).fetchall()
    movies = []
    for row in rows:
        genres = json.loads(row[2]) if row[2] else []
        embedding = json.loads(row[4]) if row[4] else []
        summary_embedding = json.loads(row[5]) if row[5] else []
        movies.append(
            {
                "id": row[0],
                "title": row[1],
                "genres": genres,
                "summary": row[3] or "",
                "embedding": embedding,
                "summary_embedding": summary_embedding,
            }
        )
    return movies


def seed_db_if_empty():
    conn = get_db()
    with DB_LOCK:
        count = conn.execute("SELECT COUNT(*) FROM movies").fetchone()[0]
    if count:
        return
    seed = load_seed_data()
    movies = seed.get("movies", [])
    users = seed.get("users", [])
    with DB_LOCK:
        for movie in movies:
            genres = movie.get("genres", [])
            if not genres:
                genres = ["drama"]
            base = GENRE_BASE.get(genres[0], [0.3, 0.3, 0.3, 0.3])
            summary = movie.get("summary") or f"{movie.get('title', 'A film')} is a {genres[0]} story."
            embedding = movie.get("embedding") or jitter(base)
            summary_embedding = movie.get("summary_embedding") or embed_summary(summary, base)
            conn.execute(
                """
                INSERT INTO movies (id, title, genres, summary, embedding, summary_embedding)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    movie.get("id"),
                    movie.get("title"),
                    json.dumps(genres, separators=(",", ":")),
                    summary,
                    json.dumps(embedding, separators=(",", ":")),
                    json.dumps(summary_embedding, separators=(",", ":")),
                ),
            )
        for user in users:
            user_id = user.get("id")
            if not user_id:
                continue
            conn.execute("INSERT OR IGNORE INTO users (id) VALUES (?)", (user_id,))
            for movie_id in user.get("history", []):
                conn.execute(
                    "INSERT OR IGNORE INTO user_history (user_id, movie_id) VALUES (?, ?)",
                    (user_id, movie_id),
                )
        conn.commit()


def add_pick(user_id: str, movie_id: str):
    conn = get_db()
    with DB_LOCK:
        conn.execute(
            "INSERT OR IGNORE INTO picks (user_id, movie_id) VALUES (?, ?)",
            (user_id, movie_id),
        )
        conn.commit()


def get_picks(user_id: str) -> set:
    conn = get_db()
    with DB_LOCK:
        rows = conn.execute(
            "SELECT movie_id FROM picks WHERE user_id=?", (user_id,)
        ).fetchall()
    return {r[0] for r in rows}


def get_user_history(user_id: str) -> List[str]:
    conn = get_db()
    with DB_LOCK:
        rows = conn.execute(
            "SELECT movie_id FROM user_history WHERE user_id=?", (user_id,)
        ).fetchall()
    return [r[0] for r in rows]


init_db()

# ------------------------
# Utils
# ------------------------


def cosine(a: List[float], b: List[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def primary_genre(m: Dict[str, object]) -> str:
    genres = m.get("genres") or []
    return genres[0] if genres else "drama"


def profile_vecs(user_id: str) -> Tuple[Optional[List[float]], Optional[List[float]]]:
    watched = set(get_user_history(user_id))
    picked = get_picks(user_id)
    ids = watched | picked
    if not ids:
        return None, None
    embs = [MOVIE_BY_ID[i]["embedding"] for i in ids if i in MOVIE_BY_ID]
    s_embs = [MOVIE_BY_ID[i]["summary_embedding"] for i in ids if i in MOVIE_BY_ID]
    dims = len(embs[0])
    p = [sum(e[i] for e in embs) / len(embs) for i in range(dims)] if embs else None
    sp = [sum(e[i] for e in s_embs) / len(s_embs) for i in range(len(s_embs[0]))] if s_embs else None
    return p, sp


# ------------------------
# Recommenders
# ------------------------


def content_recs(user_id: str, exclude: set, limit: int = 8) -> List[Dict[str, object]]:
    profile, s_profile = profile_vecs(user_id)
    if not profile and not s_profile:
        return cold_recs(exclude, limit)
    scored = []
    for m in MOVIES:
        if m["id"] in exclude:
            continue
        sim = 0.0
        if profile:
            sim += cosine(profile, m["embedding"])
        if s_profile:
            sim += cosine(s_profile, m["summary_embedding"])
        scored.append((sim, m))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [m for _, m in scored[:limit]]


def adjacent_recs(user_id: str, exclude: set, limit: int = 8) -> List[Dict[str, object]]:
    profile, s_profile = profile_vecs(user_id)
    history = get_user_history(user_id)
    picks = sorted(get_picks(user_id))
    anchor_id = history[0] if history else (picks[0] if picks else MOVIES[0]["id"])
    main_genre = primary_genre(MOVIE_BY_ID[anchor_id]) if anchor_id in MOVIE_BY_ID else None
    adj = ADJACENT.get(main_genre or "", [])
    candidates = [m for m in MOVIES if primary_genre(m) in adj and m["id"] not in exclude]
    if not candidates:
        return []
    if not profile and not s_profile:
        return candidates[:limit]
    scored = []
    for m in candidates:
        sim = 0.0
        if profile:
            sim += cosine(profile, m["embedding"])
        if s_profile:
            sim += cosine(s_profile, m["summary_embedding"])
        scored.append((sim, m))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [m for _, m in scored[:limit]]


def cold_recs(exclude: set, limit: int = 8) -> List[Dict[str, object]]:
    picks = []
    for g in ["suspense", "sci-fi", "comedy", "romance", "drama", "action", "thriller"]:
        for m in MOVIES:
            if primary_genre(m) == g and m["id"] not in exclude:
                picks.append(m)
                break
        if len(picks) >= limit:
            break
    return picks[:limit]


# ------------------------
# Stages
# ------------------------


def stage_intent(user_id: str, message: str) -> Dict[str, object]:
    client = get_openai_client()
    watched = get_user_history(user_id)
    picks = get_picks(user_id)
    default_label = "cold_start" if not watched and not picks else "recommend"
    intent = {"label": default_label, "rationale": "default", "constraints": {"genres": [], "moods": [], "keywords": []}}
    if client and message.strip():
        try:
            msg = [
                {"role": "system", "content": "Label the request. Allowed: recommend | refine | dislike | ask_question | cold_start | fallback. Return JSON {\"label\":...,\"rationale\":...,\"constraints\":{\"genres\":[],\"moods\":[],\"keywords\":[]}}"},
                {"role": "user", "content": f"user:{user_id}\nwatched:{len(watched)}\npicks:{list(picks)}\nmessage:{message}"},
            ]
            resp = client.chat.completions.create(model="gpt-4o-mini", messages=msg, max_tokens=200, temperature=0)
            parsed = json.loads(resp.choices[0].message.content or "{}")
            label = str(parsed.get("label", default_label)).strip().lower()
            if label not in {"recommend", "refine", "dislike", "ask_question", "cold_start", "fallback"}:
                label = default_label
            intent = {
                "label": label,
                "rationale": parsed.get("rationale", "LLM intent"),
                "constraints": parsed.get("constraints", intent["constraints"]),
            }
        except Exception:
            pass
    # If we already have history or picks, never return cold_start
    if picks or watched:
        intent["label"] = "recommend"
    return intent


def stage_profile(user_id: str) -> Dict[str, object]:
    profile, s_profile = profile_vecs(user_id)
    history = get_user_history(user_id)
    picks = get_picks(user_id)
    profile_strength = len(history) + len(picks)
    anchor_id = history[0] if history else (sorted(picks)[0] if picks else None)
    main_genre = primary_genre(MOVIE_BY_ID[anchor_id]) if anchor_id else None
    return {
        "profile": profile,
        "summary_profile": s_profile,
        "main_genre": main_genre,
        "profile_strength": profile_strength,
    }


def stage_planner(
    intent: Dict[str, object],
    profile: Dict[str, object],
    picks: set,
    has_signal: bool,
) -> Dict[str, object]:
    profile_strength = int(profile.get("profile_strength") or 0)
    tools: List[str]
    if not has_signal:
        tools = ["cold"]
        strategy = "cold_start"
    else:
        tools = ["content", "adjacent"]
        strategy = "content_based_weak" if profile_strength < 3 else "content_based"
    plan = {
        "tools": tools,
        "limit": 8,
        "filters": {"banned_ids": list(picks)},
        "reason": intent.get("rationale", ""),
        "strategy": strategy,
    }
    return plan


def stage_candidates(user_id: str, plan: Dict[str, object], picks: set) -> Dict[str, List[Dict[str, object]]]:
    banned = set(plan.get("filters", {}).get("banned_ids", []))
    cands = {"content": [], "adjacent": [], "cold": []}
    if "content" in plan["tools"]:
        cands["content"] = content_recs(user_id, banned, plan["limit"])
    if "adjacent" in plan["tools"]:
        adj = adjacent_recs(user_id, banned, plan["limit"])
        cands["adjacent"] = adj
    if "cold" in plan["tools"]:
        cands["cold"] = cold_recs(banned, plan["limit"])
    return cands


def stage_ranker(plan: Dict[str, object], cands: Dict[str, List[Dict[str, object]]]) -> List[Dict[str, object]]:
    tools = plan.get("tools", [])
    combined = []
    for key in ["content", "adjacent", "cold"]:
        if key in tools:
            combined.extend(cands.get(key, []))
    seen = set()
    deduped = []
    for m in combined:
        if m["id"] in seen:
            continue
        seen.add(m["id"])
        deduped.append(m)
    return deduped[: plan.get("limit", 8)]


def stage_judge(
    user_id: str,
    recs: List[Dict[str, object]],
    banned: set,
    has_signal: bool,
) -> List[Dict[str, object]]:
    filtered = [m for m in recs if m["id"] not in banned]
    if len(filtered) >= 4:
        return filtered
    exclude = banned | {m["id"] for m in filtered}
    if has_signal:
        fallback = content_recs(user_id, exclude, limit=8)
        if len(fallback) < 4:
            fallback = adjacent_recs(user_id, exclude, limit=8)
    else:
        fallback = cold_recs(exclude, limit=8)
    merged = filtered + [m for m in fallback if m["id"] not in {x["id"] for x in filtered}]
    return merged[:8]


def stage_explain(strategy: str, recs: List[Dict[str, object]]) -> str:
    header = {
        "cold_start": "Starting you off with a mix.",
        "content_based": "Matched to your tastes.",
        "content_based_weak": "Starting from what you liked, with a bit of variety.",
        "adjacent_genres_exploration": "Exploring nearby genres.",
    }.get(strategy, "Here are some picks.")
    lines = [header, ""]
    for i, m in enumerate(recs, start=1):
        lines.append(f"{i}. {m['title']} ({', '.join(m['genres'])})")
    lines.append("")
    lines.append("Which one would you actually watch? Reply with a number, or 'none'.")
    return "\n".join(lines)


def parse_choice(user_id: str, message: str, recs: List[Dict[str, object]]) -> Optional[str]:
    msg = message.strip().lower()
    if not msg:
        return None
    if msg in {"none", "nope", "nah"} or msg.startswith("none"):
        return None
    if recs:
        token = msg.split()[0].strip(" .,)(")
        if token.isdigit():
            idx = int(token) - 1
            if 0 <= idx < len(recs):
                return recs[idx]["id"]
        title_map = {m["title"].lower(): m["id"] for m in recs}
        if msg in title_map:
            return title_map[msg]
        for t, mid in title_map.items():
            if msg in t or t in msg:
                return mid
    client = get_openai_client()
    if client:
        try:
            numbered = "\n".join(f"{i+1}. {m['title']}" for i, m in enumerate(recs))
            prompt = (
                "Map the user reply to one movie number from the list, or NONE.\n"
                f"List:\n{numbered}\n"
                f"Reply: {message}\n"
                "Return only the number or NONE."
            )
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=2,
                temperature=0,
            )
            content = (resp.choices[0].message.content or "").strip().lower()
            if "none" in content:
                return None
            tok = content.split()[0].strip(" .,)(")
            if tok.isdigit():
                idx = int(tok) - 1
                if 0 <= idx < len(recs):
                    return recs[idx]["id"]
        except Exception:
            pass
    for m in MOVIES:
        title = m["title"].lower()
        if msg == title or msg in title or title in msg:
            return m["id"]
    return None


# ------------------------
# LangGraph
# ------------------------


class RecState(TypedDict, total=False):
    user_id: str
    message: str
    watched: List[str]
    picks: List[str]
    has_signal: bool
    intent: Dict[str, object]
    profile: Dict[str, object]
    plan: Dict[str, object]
    candidates_content: List[Dict[str, object]]
    candidates_adjacent: List[Dict[str, object]]
    candidates_cold: List[Dict[str, object]]
    ranked_recs: List[Dict[str, object]]
    final_recs: List[Dict[str, object]]
    response: str


def _node_intent(state: RecState) -> Dict[str, object]:
    return {"intent": stage_intent(state["user_id"], state.get("message", ""))}


def _node_profile(state: RecState) -> Dict[str, object]:
    return {"profile": stage_profile(state["user_id"])}


def _node_planner(state: RecState) -> Dict[str, object]:
    picks = set(state.get("picks") or [])
    plan = stage_planner(
        state.get("intent", {}),
        state.get("profile", {}),
        picks,
        bool(state.get("has_signal")),
    )
    return {"plan": plan}


def _node_candidates_content(state: RecState) -> Dict[str, object]:
    plan = state.get("plan", {})
    picks = set(state.get("picks") or [])
    recs = content_recs(state["user_id"], picks, plan.get("limit", 8)) if "content" in plan.get("tools", []) else []
    return {"candidates_content": recs}


def _node_candidates_adjacent(state: RecState) -> Dict[str, object]:
    plan = state.get("plan", {})
    picks = set(state.get("picks") or [])
    recs = adjacent_recs(state["user_id"], picks, plan.get("limit", 8)) if "adjacent" in plan.get("tools", []) else []
    return {"candidates_adjacent": recs}


def _node_candidates_cold(state: RecState) -> Dict[str, object]:
    plan = state.get("plan", {})
    picks = set(state.get("picks") or [])
    recs = cold_recs(picks, plan.get("limit", 8)) if "cold" in plan.get("tools", []) else []
    return {"candidates_cold": recs}


def _node_ranker(state: RecState) -> Dict[str, object]:
    plan = state.get("plan", {})
    cands = {
        "content": state.get("candidates_content", []),
        "adjacent": state.get("candidates_adjacent", []),
        "cold": state.get("candidates_cold", []),
    }
    return {"ranked_recs": stage_ranker(plan, cands)}


def _node_judge(state: RecState) -> Dict[str, object]:
    picks = set(state.get("picks") or [])
    recs = stage_judge(
        state["user_id"],
        state.get("ranked_recs", []),
        picks,
        bool(state.get("has_signal")),
    )
    return {"final_recs": recs}


def _node_explain(state: RecState) -> Dict[str, object]:
    plan = state.get("plan", {})
    strategy = plan.get("strategy", "content_based")
    response = stage_explain(strategy, state.get("final_recs", []))
    return {"response": response}


def build_recommendation_graph():
    graph = StateGraph(RecState)
    graph.add_node("intent", _node_intent)
    graph.add_node("profile", _node_profile)
    graph.add_node("planner", _node_planner)
    graph.add_node("candidates_content", _node_candidates_content)
    graph.add_node("candidates_adjacent", _node_candidates_adjacent)
    graph.add_node("candidates_cold", _node_candidates_cold)
    graph.add_node("ranker", _node_ranker)
    graph.add_node("judge", _node_judge)
    graph.add_node("explain", _node_explain)
    graph.set_entry_point("intent")
    graph.add_edge("intent", "profile")
    graph.add_edge("profile", "planner")
    graph.add_edge("planner", "candidates_content")
    graph.add_edge("candidates_content", "candidates_adjacent")
    graph.add_edge("candidates_adjacent", "candidates_cold")
    graph.add_edge("candidates_cold", "ranker")
    graph.add_edge("ranker", "judge")
    graph.add_edge("judge", "explain")
    graph.add_edge("explain", END)
    return graph.compile()


recommendation_graph = build_recommendation_graph()

# ------------------------
# API
# ------------------------


class ChatRequest(BaseModel):
    userId: str
    message: str


@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/chat")
async def chat(req: ChatRequest):
    user_id = req.userId.strip()
    if not user_id:
        raise HTTPException(status_code=400, detail="userId is required")
    msg = req.message or ""

    # picks/history
    picks = list(get_picks(user_id))
    watched = get_user_history(user_id)
    has_signal = bool(watched or picks)

    initial_state: RecState = {
        "user_id": user_id,
        "message": msg,
        "watched": watched,
        "picks": picks,
        "has_signal": has_signal,
    }
    result_state = recommendation_graph.invoke(initial_state)
    recs = result_state.get("final_recs", [])
    plan = result_state.get("plan", {})
    intent = result_state.get("intent", {})

    # parse choice
    chosen = parse_choice(user_id, msg, recs)
    if chosen:
        add_pick(user_id, chosen)
        picks = list(get_picks(user_id))
        has_signal = True
        follow_state: RecState = {
            "user_id": user_id,
            "message": msg,
            "watched": watched,
            "picks": picks,
            "has_signal": has_signal,
        }
        result_state = recommendation_graph.invoke(follow_state)
        recs = result_state.get("final_recs", [])
        plan = result_state.get("plan", {})
        intent = result_state.get("intent", {})

    strategy = plan.get("strategy", "content_based")
    response = result_state.get("response") or stage_explain(strategy, recs)
    decision = {"strategy": strategy, "reason": plan.get("reason", intent.get("rationale"))}
    return {
        "strategy": strategy,
        "intent": intent,
        "decision": decision,
        "recommendations": [{"id": m["id"], "title": m["title"], "genres": m["genres"]} for m in recs],
        "picked": list(get_picks(user_id)),
        "response": response,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
