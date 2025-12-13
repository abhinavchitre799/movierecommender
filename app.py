import json
import math
import os
import random
import threading
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from langgraph.graph import END, StateGraph
from pydantic import BaseModel

app = FastAPI(title="Movie Recommendation Agent")
templates = Jinja2Templates(directory="templates")

random.seed(42)
# Optional override; prefer env vars OPENAI_API_KEY_OVERRIDE or OPENAI_API_KEY instead of hardcoding secrets.
OPENAI_API_KEY_OVERRIDE = os.getenv("OPENAI_API_KEY_OVERRIDE")
STATE_PATH = Path(__file__).resolve().parent / "state.json"
STATE_LOCK = threading.Lock()
picked_movies_by_user: Dict[str, List[str]] = defaultdict(list)
last_recs_by_user: Dict[str, List[str]] = defaultdict(list)
EMBEDDING_CACHE_PATH = Path(__file__).resolve().parent / "embeddings_cache.json"
EMBEDDING_CACHE_LOCK = threading.Lock()
EMBEDDING_CACHE: Dict[str, List[float]] = {}
EMBEDDING_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")


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


def load_state_from_disk():
    if not STATE_PATH.exists():
        return
    try:
        data = json.loads(STATE_PATH.read_text())
        picked = data.get("picked_movies", {})
        last = data.get("last_recs", {})
        picked_movies_by_user.update({k: list(v) for k, v in picked.items()})
        last_recs_by_user.update({k: list(v) for k, v in last.items()})
    except Exception:
        # Ignore corrupt state and continue
        return


def persist_state_to_disk():
    payload = {
        "picked_movies": picked_movies_by_user,
        "last_recs": last_recs_by_user,
    }
    try:
        with STATE_LOCK:
            STATE_PATH.write_text(json.dumps(payload, indent=2))
    except Exception:
        # If persisting fails, continue without crashing the app
        return


load_state_from_disk()


def load_embedding_cache():
    if not EMBEDDING_CACHE_PATH.exists():
        return
    try:
        data = json.loads(EMBEDDING_CACHE_PATH.read_text())
        if isinstance(data, dict):
            EMBEDDING_CACHE.update({k: list(v) for k, v in data.items()})
    except Exception:
        return


def persist_embedding_cache():
    try:
        with EMBEDDING_CACHE_LOCK:
            EMBEDDING_CACHE_PATH.write_text(json.dumps(EMBEDDING_CACHE))
    except Exception:
        return


load_embedding_cache()


class ChatRequest(BaseModel):
    userId: str
    message: str


Movie = Dict[str, object]

GENRE_BASE_VECTORS: Dict[str, List[float]] = {
    "suspense": [0.9, 0.12, 0.1, 0.08, 0.05, 0.05],
    "thriller": [0.88, 0.14, 0.12, 0.1, 0.05, 0.05],
    "crime": [0.86, 0.15, 0.1, 0.1, 0.05, 0.05],
    "psychological": [0.89, 0.12, 0.14, 0.07, 0.05, 0.05],
    "sci-fi": [0.45, 0.18, 0.75, 0.12, 0.05, 0.05],
    "comedy": [0.08, 0.9, 0.1, 0.06, 0.05, 0.05],
    "romance": [0.12, 0.88, 0.15, 0.06, 0.05, 0.05],
    "drama": [0.45, 0.42, 0.25, 0.15, 0.05, 0.05],
    "action": [0.48, 0.15, 0.28, 0.75, 0.05, 0.05],
}

GENRE_ADJACENCY: Dict[str, List[str]] = {
    "suspense": ["thriller", "crime", "psychological", "sci-fi", "drama"],
    "thriller": ["suspense", "crime", "action", "sci-fi"],
    "crime": ["thriller", "suspense", "drama"],
    "psychological": ["suspense", "thriller", "drama", "sci-fi"],
    "sci-fi": ["thriller", "suspense", "action"],
    "comedy": ["romance", "drama"],
    "romance": ["comedy", "drama"],
    "drama": ["romance", "thriller", "suspense"],
    "action": ["thriller", "sci-fi", "crime"],
}

GENRE_TITLES: Dict[str, List[str]] = {
    "suspense": [
        "Se7en",
        "Zodiac",
        "Prisoners",
        "Gone Girl",
        "The Girl with the Dragon Tattoo",
        "Shutter Island",
        "Memento",
        "Rear Window",
        "Vertigo",
        "Nightcrawler",
        "The Silence of the Lambs",
        "The Sixth Sense",
    ],
    "thriller": [
        "The Bourne Identity",
        "Sicario",
        "No Country for Old Men",
        "Skyfall",
        "The Fugitive",
        "Collateral",
        "The Game",
        "Insomnia",
        "Enemy of the State",
        "Argo",
        "Munich",
        "Captain Phillips",
    ],
    "crime": [
        "The Godfather",
        "The Godfather Part II",
        "Goodfellas",
        "The Departed",
        "City of God",
        "LA Confidential",
        "The Irishman",
        "Pulp Fiction",
        "Reservoir Dogs",
        "Mystic River",
        "Eastern Promises",
        "Training Day",
    ],
    "psychological": [
        "Black Swan",
        "Fight Club",
        "The Machinist",
        "Donnie Darko",
        "A Beautiful Mind",
        "Taxi Driver",
        "American Psycho",
        "Hereditary",
        "Joker",
        "Pi",
        "Enemy",
        "Mulholland Drive",
    ],
    "sci-fi": [
        "Blade Runner 2049",
        "Arrival",
        "Interstellar",
        "Inception",
        "Ex Machina",
        "Annihilation",
        "Minority Report",
        "Looper",
        "Edge of Tomorrow",
        "District 9",
        "The Matrix",
        "Snowpiercer",
    ],
    "comedy": [
        "The Big Lebowski",
        "Superbad",
        "Step Brothers",
        "Anchorman",
        "Mean Girls",
        "Bridesmaids",
        "Groundhog Day",
        "Napoleon Dynamite",
        "Hot Fuzz",
        "21 Jump Street",
        "Crazy Rich Asians",
        "Game Night",
    ],
    "romance": [
        "The Notebook",
        "La La Land",
        "Pride and Prejudice",
        "About Time",
        "Before Sunrise",
        "Notting Hill",
        "10 Things I Hate About You",
        "A Star Is Born",
        "Titanic",
        "Carol",
        "Silver Linings Playbook",
        "Brooklyn",
    ],
    "drama": [
        "The Shawshank Redemption",
        "Forrest Gump",
        "Whiplash",
        "Moonlight",
        "Manchester by the Sea",
        "Spotlight",
        "The Social Network",
        "Parasite",
        "The Pursuit of Happyness",
        "There Will Be Blood",
        "The King's Speech",
        "Moneyball",
    ],
    "action": [
        "Mad Max Fury Road",
        "John Wick",
        "Gladiator",
        "The Dark Knight",
        "Die Hard",
        "Casino Royale",
        "Mission Impossible Fallout",
        "The Raid",
        "Taken",
        "Speed",
        "Terminator 2",
        "The Avengers",
    ],
}


def jitter_vector(base: List[float], scale: float = 0.04) -> List[float]:
    jittered = []
    for value in base:
        val = value + random.uniform(-scale, scale)
        jittered.append(max(0.0, min(1.0, val)))
    return jittered


def embed_text(text: str) -> Optional[List[float]]:
    """Embed text via OpenAI; fallback to None if unavailable. Uses on-disk cache."""
    if not text:
        return None
    cached = EMBEDDING_CACHE.get(text)
    if cached:
        return cached
    client = get_openai_client()
    if not client:
        return None
    try:
        result = client.embeddings.create(
            input=[text],
            model=EMBEDDING_MODEL,
        )
        embedding = result.data[0].embedding
        if embedding:
            EMBEDDING_CACHE[text] = embedding
            persist_embedding_cache()
            return embedding
    except Exception:
        return None
    return None


def build_summary(title: str, genre: str) -> str:
    return f"{title} is a {genre} pick that blends the genre's core tone with character-driven stakes."


def generate_catalog() -> List[Movie]:
    catalog: List[Movie] = []
    idx = 1
    for genre, titles in GENRE_TITLES.items():
        for title in titles:
            base_vec = GENRE_BASE_VECTORS[genre]
            summary = build_summary(title, genre)
            emb = embed_text(f"{title} {genre}") or jitter_vector(base_vec)
            summary_emb = embed_text(summary) or jitter_vector(base_vec, scale=0.06)
            catalog.append(
                {
                    "id": f"m{idx}",
                    "title": title,
                    "genres": [genre],
                    "embedding": emb,
                    "summary": summary,
                    "summary_embedding": summary_emb,
                }
            )
            idx += 1
    return catalog


movie_catalog: List[Movie] = generate_catalog()
movie_by_id: Dict[str, Movie] = {m["id"]: m for m in movie_catalog}


def primary_genre(movie: Movie) -> str:
    return movie["genres"][0] if movie.get("genres") else "unknown"


def ids_by_genre(target_genre: str) -> List[str]:
    return [m["id"] for m in movie_catalog if primary_genre(m) == target_genre]


def build_user_history() -> Dict[str, List[str]]:
    suspense_ids = ids_by_genre("suspense")
    thriller_ids = ids_by_genre("thriller")
    crime_ids = ids_by_genre("crime")
    romance_ids = ids_by_genre("romance")
    comedy_ids = ids_by_genre("comedy")
    drama_ids = ids_by_genre("drama")
    scifi_ids = ids_by_genre("sci-fi")
    action_ids = ids_by_genre("action")

    user1_set = set(suspense_ids + thriller_ids[:8] + crime_ids[:5])
    user2_set = set(romance_ids[:8] + comedy_ids[:8] + drama_ids[:4])
    user3_set = set(suspense_ids[:3] + scifi_ids[:6] + drama_ids[4:10] + action_ids[:3])

    return {
        "user1": list(user1_set),
        "user2": list(user2_set),
        "user3": list(user3_set),
    }


user_history: Dict[str, List[str]] = build_user_history()


def score_movie_for_user(
    movie: Movie,
    profile: Optional[List[float]],
    summary_profile: Optional[List[float]],
    preferred_genres: set[str],
    intent_label: str,
    last_recs: Optional[List[str]],
) -> float:
    sim_vec = cosine_similarity(profile, movie["embedding"]) if profile else 0.0
    sim_summary = (
        cosine_similarity(summary_profile, movie.get("summary_embedding", movie["embedding"]))
        if summary_profile
        else 0.0
    )
    base = (sim_vec + sim_summary) / (2 if profile and summary_profile else 1)
    genre_bonus = 0.05 if preferred_genres and primary_genre(movie) in preferred_genres else 0.0
    intent_penalty = -0.05 if intent_label in {"dislike"} and movie["id"] in (last_recs or []) else 0.0
    return base + genre_bonus + intent_penalty


def cosine_similarity(a: List[float], b: List[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def get_user_profile(user_id: str) -> Optional[List[float]]:
    watched_ids = set(user_history.get(user_id, []))
    watched_embeddings = [
        movie["embedding"]
        for movie in movie_catalog
        if movie["id"] in watched_ids
    ]
    if not watched_embeddings:
        return None
    dims = len(watched_embeddings[0])
    return [
        sum(emb[i] for emb in watched_embeddings) / len(watched_embeddings)
        for i in range(dims)
    ]


def get_user_summary_profile(user_id: str) -> Optional[List[float]]:
    watched_ids = set(user_history.get(user_id, []))
    watched_embeddings = [
        movie.get("summary_embedding", movie["embedding"])
        for movie in movie_catalog
        if movie["id"] in watched_ids
    ]
    if not watched_embeddings:
        return None
    dims = len(watched_embeddings[0])
    return [
        sum(emb[i] for emb in watched_embeddings) / len(watched_embeddings)
        for i in range(dims)
    ]


def cold_start_recommendations(limit: int = 8) -> List[Movie]:
    seeds = ["suspense", "romance", "comedy", "sci-fi", "drama", "action", "crime"]
    recs: List[Movie] = []
    for genre in seeds:
        recs.extend([m for m in movie_catalog if primary_genre(m) == genre][:1])
        if len(recs) >= limit:
            break
    return recs[:limit]


def content_based_recommendations(
    user_id: str, limit: int = 8, exclude_ids: Optional[set[str]] = None
) -> List[Movie]:
    exclude_ids = exclude_ids or set()
    profile = get_user_profile(user_id)
    summary_profile = get_user_summary_profile(user_id)
    if not profile and not summary_profile:
        return cold_start_recommendations(limit)

    watched = set(user_history.get(user_id, []))
    scored: List[tuple[float, Movie]] = []
    for movie in movie_catalog:
        if movie["id"] in watched or movie["id"] in exclude_ids:
            continue
        sim_vec = cosine_similarity(profile, movie["embedding"]) if profile else 0.0
        sim_summary = (
            cosine_similarity(summary_profile, movie.get("summary_embedding", movie["embedding"]))
            if summary_profile
            else 0.0
        )
        sim = (sim_vec + sim_summary) / (2 if profile and summary_profile else 1)
        scored.append((sim, movie))

    scored.sort(key=lambda x: x[0], reverse=True)
    filtered = [m for _, m in scored if m["id"] not in exclude_ids][:limit]
    return filtered


def adjacent_genre_recommendations(
    user_id: str, limit: int = 8, exclude_ids: Optional[set[str]] = None
) -> List[Movie]:
    exclude_ids = exclude_ids or set()
    watched = set(user_history.get(user_id, []))
    main_genre = get_main_genre(user_id)
    adjacent = GENRE_ADJACENCY.get(main_genre or "", [])
    profile = get_user_profile(user_id)
    summary_profile = get_user_summary_profile(user_id)

    if not adjacent:
        return content_based_recommendations(user_id, limit)

    candidates = [
        m
        for m in movie_catalog
        if primary_genre(m) in adjacent and m["id"] not in watched and m["id"] not in exclude_ids
    ]
    if not candidates:
        return cold_start_recommendations(limit)

    if not profile and not summary_profile:
        return candidates[:limit]

    scored = []
    for m in candidates:
        sim_vec = cosine_similarity(profile, m["embedding"]) if profile else 0.0
        sim_summary = (
            cosine_similarity(summary_profile, m.get("summary_embedding", m["embedding"]))
            if summary_profile
            else 0.0
        )
        sim = (sim_vec + sim_summary) / (2 if profile and summary_profile else 1)
        scored.append((sim, m))

    scored.sort(key=lambda x: x[0], reverse=True)
    filtered = [m for _, m in scored if m["id"] not in exclude_ids][:limit]
    return filtered


def get_main_genre(user_id: str) -> Optional[str]:
    watched_ids = set(user_history.get(user_id, []))
    genre_counts: Dict[str, int] = {}
    for movie in movie_catalog:
        pg = primary_genre(movie)
        if movie["id"] in watched_ids:
            genre_counts[pg] = genre_counts.get(pg, 0) + 1
    if not genre_counts:
        return None
    return max(genre_counts.items(), key=lambda x: x[1])[0]


def genre_coverage(user_id: str, genre: str) -> float:
    watched_ids = set(user_history.get(user_id, []))
    total = len([m for m in movie_catalog if primary_genre(m) == genre])
    watched = len(
        [m for m in movie_catalog if m["id"] in watched_ids and primary_genre(m) == genre]
    )
    if total == 0:
        return 0.0
    return watched / total


def decide_strategy(user_id: str, user_message: str) -> Dict[str, object]:
    watched = user_history.get(user_id, [])
    user_type = "new" if not watched else "existing"
    main_genre = get_main_genre(user_id)
    coverage = genre_coverage(user_id, main_genre) if main_genre else 0.0
    cluster_exhausted = bool(main_genre) and coverage >= 0.8

    heuristic_strategy = "cold_start" if not watched else (
        "adjacent_genres_exploration" if cluster_exhausted else "content_based"
    )
    heuristic_reason = (
        "No watch history yet, so using a varied cold-start set."
        if not watched
        else (
            f"You've seen nearly all {main_genre} picks; exploring adjacent genres."
            if cluster_exhausted
            else "Building on what you've already watched."
        )
    )

    client = get_openai_client()
    if client:
        try:
            messages = [
                {
                    "role": "system",
                    "content": (
                        "Pick a single strategy for recommendations based on user history.\n"
                        "Return only one token: cold_start | content_based | adjacent_genres_exploration."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"user_id: {user_id}\n"
                        f"user_type: {user_type}\n"
                        f"main_genre: {main_genre}\n"
                        f"coverage: {coverage:.2f}\n"
                        f"message: {user_message}\n"
                    ),
                },
            ]
            completion = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                max_tokens=2,
                temperature=0,
            )
            content = completion.choices[0].message.content or ""
            normalized = content.strip().split()[0].lower()
            if normalized in {"cold_start", "content_based", "adjacent_genres_exploration"}:
                heuristic_strategy = normalized
                heuristic_reason = "LLM-selected strategy based on history and message."
        except Exception:
            pass

    return {
        "userType": user_type,
        "clusterExhausted": cluster_exhausted,
        "strategy": heuristic_strategy,
        "reason": heuristic_reason,
        "mainGenre": main_genre,
        "coverage": coverage,
    }


def format_response(
    decision: Dict[str, object], recommendations: List[Movie], user_id: str
) -> str:
    choice_ack = decision.get("choice_ack") or ""
    strategy_phrases = {
        "cold_start": "Starting with a balanced mix to learn your taste.",
        "content_based": "Matched to the vibe of what you've seen.",
        "adjacent_genres_exploration": "You've exhausted your core genre, so I'm branching out.",
    }
    strategy = decision.get("strategy", "content_based")
    header = strategy_phrases.get(strategy, "Here are some ideas to explore.")
    reason = decision.get("reason", "")
    meta_line = f"Strategy: {strategy}."
    critic = decision.get("critic") or {}
    iteration = decision.get("iteration", 1)
    rec_lines = [
        f"{idx+1}. {movie['title']} ({', '.join(movie['genres'])})"
        for idx, movie in enumerate(recommendations)
    ]
    if not rec_lines:
        rec_lines = ["- I didn't find anything new, but I can widen the search."]
    display_user = user_id or "you"
    prompt_line = "Which one would you actually watch? Reply with a number, or 'none' if nothing fits."
    lines = [header, meta_line, f"Iteration: {iteration}", reason]
    if critic.get("verdict") == "retry":
        lines.append(f"Critic retry: {critic.get('reason')}")
    elif critic.get("reason"):
        lines.append(f"Critic: {critic.get('reason')}")
    if choice_ack:
        lines.append(f"Ack: {choice_ack}")
    lines.extend(["", f"Recommendations for {display_user}:", *rec_lines, "", prompt_line])
    return "\n".join(lines)


def _node_intent(state: Dict[str, object]) -> Dict[str, object]:
    """Classify the latest message into an intent and extract soft constraints."""
    user_id = str(state.get("userId", ""))
    message = str(state.get("message", "")).strip()
    watched = user_history.get(user_id, [])
    user_type = "new" if not watched else "existing"
    main_genre = get_main_genre(user_id)
    coverage = genre_coverage(user_id, main_genre) if main_genre else 0.0

    default_label = "cold_start" if user_type == "new" else "recommend"
    intent: Dict[str, object] = {
        "label": default_label,
        "rationale": "Default intent chosen based on watch history.",
        "constraints": {"genres": [], "moods": [], "keywords": []},
    }

    client = get_openai_client()
    if client and message:
        try:
            messages = [
                {
                    "role": "system",
                    "content": (
                        "You classify a user's ask into one intent: "
                        "recommend | refine | dislike | ask_question | cold_start | fallback. "
                        "Also extract any genres, moods, or keywords mentioned. "
                        "Return JSON: {\"label\": \"...\", \"rationale\": \"...\", \"constraints\": {\"genres\": [], \"moods\": [], \"keywords\": []}}. "
                        "Keep label to the allowed set; if uncertain, use \"recommend\" for existing users or \"cold_start\" for new users."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"user_id: {user_id}\n"
                        f"user_type: {user_type}\n"
                        f"main_genre: {main_genre}\n"
                        f"coverage: {coverage:.2f}\n"
                        f"message: {message}\n"
                        f"picked_movies: {picked_movies_by_user.get(user_id, [])}\n"
                    ),
                },
            ]
            completion = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                max_tokens=200,
                temperature=0,
            )
            content = completion.choices[0].message.content or ""
            parsed = json.loads(content)
            label = str(parsed.get("label", "")).strip().lower()
            if label not in {
                "recommend",
                "refine",
                "dislike",
                "ask_question",
                "cold_start",
                "fallback",
            }:
                label = default_label
            rationale = parsed.get("rationale") or intent["rationale"]
            constraints = parsed.get("constraints") or intent["constraints"]
            intent = {"label": label, "rationale": rationale, "constraints": constraints}
        except Exception:
            pass

    return {
        "intent": intent,
        "userType": user_type,
        "mainGenre": main_genre,
        "coverage": coverage,
    }


def _node_profile(state: Dict[str, object]) -> Dict[str, object]:
    user_id = state.get("userId", "")
    profile = get_user_profile(str(user_id))
    summary_profile = get_user_summary_profile(str(user_id))
    main_genre = get_main_genre(str(user_id))
    picked = picked_movies_by_user.get(str(user_id), [])
    coverage = genre_coverage(str(user_id), main_genre) if main_genre else 0.0
    last_recs = last_recs_by_user.get(str(user_id), [])
    return {
        "profile": profile,
        "summary_profile": summary_profile,
        "mainGenre": main_genre,
        "picked_movies": picked,
        "coverage": coverage,
        "last_recs": last_recs,
    }


def _node_choice(state: Dict[str, object]) -> Dict[str, object]:
    user_id = str(state.get("userId", ""))
    message_raw = str(state.get("message", ""))
    message = message_raw.strip().lower()
    picked = picked_movies_by_user.get(user_id, []).copy()
    last_recs = last_recs_by_user.get(user_id, [])
    ack = ""

    if message:
        token = message.split()[0].strip(" .,)")
        if token.isdigit():
            idx = int(token) - 1
            if 0 <= idx < len(last_recs):
                chosen_id = last_recs[idx]
                if chosen_id not in picked:
                    picked.append(chosen_id)
                    ack = f"Added your pick: {chosen_id}"
        elif "none" in message:
            ack = "Got it, skipping those picks."

    picked_movies_by_user[user_id] = picked
    persist_state_to_disk()
    return {"picked_movies": picked, "choice_ack": ack}


def _node_content(state: Dict[str, object]) -> Dict[str, object]:
    user_id = state.get("userId", "")
    plan = state.get("plan", {}) or {}
    if "content" not in plan.get("tools", ["content"]):
        return {"content_candidates": []}
    picked = set(state.get("picked_movies") or [])
    banned_ids = set((plan.get("filters") or {}).get("banned_ids", []))
    recs = content_based_recommendations(str(user_id), exclude_ids=picked | banned_ids)
    return {"content_candidates": recs}


def _node_adjacent(state: Dict[str, object]) -> Dict[str, object]:
    user_id = state.get("userId", "")
    plan = state.get("plan", {}) or {}
    if "adjacent" not in plan.get("tools", []):
        return {"adjacent_candidates": []}
    picked = set(state.get("picked_movies") or [])
    banned_ids = set((plan.get("filters") or {}).get("banned_ids", []))
    recs = adjacent_genre_recommendations(str(user_id), exclude_ids=picked | banned_ids)
    return {"adjacent_candidates": recs}


def _node_cold(state: Dict[str, object]) -> Dict[str, object]:
    plan = state.get("plan", {}) or {}
    if "cold" not in plan.get("tools", []):
        return {"cold_candidates": []}
    picked = set(state.get("picked_movies") or [])
    banned_ids = set((plan.get("filters") or {}).get("banned_ids", []))
    recs = [m for m in cold_start_recommendations() if m["id"] not in picked and m["id"] not in banned_ids]
    if not recs:
        recs = [m for m in movie_catalog if m["id"] not in picked and m["id"] not in banned_ids][:8]
    return {"cold_candidates": recs}


def _node_planner(state: Dict[str, object]) -> Dict[str, object]:
    """Pick which tools to use and how many results to return based on intent and profile."""
    intent = state.get("intent", {}) or {}
    label = intent.get("label", "recommend")
    main_genre = state.get("mainGenre")
    coverage = state.get("coverage") or 0.0
    profile = state.get("profile")
    picked = state.get("picked_movies") or []
    last_recs = state.get("last_recs") or []
    constraints = intent.get("constraints", {}) or {}
    message = str(state.get("message", "") or "").strip()
    retry_broaden = bool(state.get("retry_broaden"))
    retry_note = state.get("retry_note") or ""

    tools: List[str] = []
    reason = ""

    if retry_broaden:
        tools = ["content", "adjacent", "cold"]
        reason = "Critic retry; broadened toolset."
    elif label == "cold_start" or not profile:
        tools = ["cold"]
        reason = "Cold-start flow for new or profile-less user."
    elif label in {"refine", "dislike"}:
        tools = ["adjacent", "content", "cold"]
        reason = "User refining/disliking; mixing adjacent and fresh cold picks."
    elif label == "ask_question":
        tools = ["content", "adjacent"]
        reason = "User asked a question; giving contextual picks."
    else:
        tools = ["content", "adjacent"]
        reason = "Default recommend using content plus nearby genres."

    if coverage >= 0.8 and "adjacent" not in tools:
        tools.append("adjacent")
        reason += " Main genre nearly exhausted; branching to adjacent."

    banned_ids = list(set(picked + (last_recs if label in {"refine", "dislike"} else [])))
    plan = {
        "tools": tools,
        "limit": 8,
        "filters": {
            "genres": constraints.get("genres") or [],
            "moods": constraints.get("moods") or [],
            "keywords": constraints.get("keywords") or [],
            "banned_ids": banned_ids,
        },
        "reason": (reason or intent.get("rationale") or "") + (f" Retry note: {retry_note}" if retry_note else ""),
        "intent_label": label,
        "retry_broaden": retry_broaden,
    }
    return {"plan": plan}


def _node_ranker(state: Dict[str, object]) -> Dict[str, object]:
    plan = state.get("plan", {}) or {}
    tools = plan.get("tools", [])
    limit = plan.get("limit", 8)
    filters = plan.get("filters") or {}
    preferred_genres = set(filters.get("genres") or [])
    banned_ids = set(filters.get("banned_ids") or [])
    intent_label = plan.get("intent_label", "")

    candidates: List[Movie] = []
    for key, cands in [
        ("content", state.get("content_candidates") or []),
        ("adjacent", state.get("adjacent_candidates") or []),
        ("cold", state.get("cold_candidates") or []),
    ]:
        if key in tools:
            candidates.extend(cands)

    seen = set()
    deduped: List[Movie] = []
    for movie in candidates:
        if movie["id"] in seen or movie["id"] in banned_ids:
            continue
        seen.add(movie["id"])
        deduped.append(movie)

    profile = state.get("profile")
    summary_profile = state.get("summary_profile")
    last_recs = state.get("last_recs")

    scored = sorted(
        [
            (
                score_movie_for_user(
                    m,
                    profile,
                    summary_profile,
                    preferred_genres,
                    intent_label,
                    last_recs,
                ),
                m,
            )
            for m in deduped
        ],
        key=lambda x: x[0],
        reverse=True,
    )
    final_recs = [m for _, m in scored][:limit]
    user_id = str(state.get("userId", ""))
    last_recs_by_user[user_id] = [m["id"] for m in final_recs]
    persist_state_to_disk()
    return {"recommendations": final_recs}


def _autonomous_fallback_recs(state: Dict[str, object], force_broaden: bool = False) -> List[Movie]:
    """Broaden search if critic requests a retry."""
    plan = state.get("plan", {}) or {}
    limit = plan.get("limit", 8)
    filters = plan.get("filters") or {}
    preferred_genres = set(filters.get("genres") or [])
    banned_ids = set(filters.get("banned_ids") or [])
    picked = set(state.get("picked_movies") or [])
    intent_label = plan.get("intent_label", "")
    profile = state.get("profile")
    summary_profile = state.get("summary_profile")
    last_recs = state.get("last_recs")

    candidates: List[Movie] = []
    for key, cands in [
        ("content", state.get("content_candidates") or []),
        ("adjacent", state.get("adjacent_candidates") or []),
        ("cold", state.get("cold_candidates") or []),
    ]:
        if key in plan.get("tools", []):
            candidates.extend(cands)

    if force_broaden or not candidates:
        candidates.extend(
            [
                m
                for m in movie_catalog
                if m["id"] not in picked and m["id"] not in banned_ids
            ]
        )

    seen = set()
    deduped: List[Movie] = []
    for movie in candidates:
        if movie["id"] in seen or movie["id"] in banned_ids or movie["id"] in picked:
            continue
        seen.add(movie["id"])
        deduped.append(movie)

    scored = sorted(
        [
            (
                score_movie_for_user(
                    m,
                    profile,
                    summary_profile,
                    preferred_genres,
                    intent_label,
                    last_recs,
                ),
                m,
            )
            for m in deduped
        ],
        key=lambda x: x[0],
        reverse=True,
    )
    return [m for _, m in scored][:limit]


def _node_critic(state: Dict[str, object]) -> Dict[str, object]:
    """Self-check: validate recs vs intent/plan; retry with broader search if needed."""
    intent = state.get("intent", {}) or {}
    plan = state.get("plan", {}) or {}
    recs: List[Movie] = state.get("recommendations", []) or []
    verdict = "pass" if recs else "retry"
    reason = "Skipped LLM critic; using heuristic." if not recs else "Initial recs accepted."
    fix = ""
    client = get_openai_client()

    if client and recs:
        try:
            rec_lines = [
                f"{idx+1}. {m['title']} ({', '.join(m['genres'])})"
                for idx, m in enumerate(recs)
            ]
            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are a critic that checks if recommendations match intent and constraints. "
                        "Return JSON: {\"verdict\": \"pass\" | \"retry\", \"reason\": \"...\", \"fix\": \"broaden\"|\"adjacent\"|\"cold\"|\"ok\"}. "
                        "Choose retry if picks don't match intent, constraints, or main genre exhaustion."
                    ),
                },
                {
                    "role": "user",
                    "content": "\n".join(
                        [
                            f"Intent: {intent.get('label')}",
                            f"Intent rationale: {intent.get('rationale')}",
                            f"Plan tools: {plan.get('tools')}",
                            f"Plan reason: {plan.get('reason')}",
                            f"Main genre: {state.get('mainGenre')}",
                            f"Coverage: {state.get('coverage')}",
                            f"Constraints: {intent.get('constraints')}",
                            "Recs:",
                            *rec_lines,
                        ]
                    ),
                },
            ]
            completion = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                max_tokens=120,
                temperature=0,
            )
            content = completion.choices[0].message.content or ""
            parsed = json.loads(content)
            verdict_parsed = str(parsed.get("verdict", "")).strip().lower()
            if verdict_parsed in {"pass", "retry"}:
                verdict = verdict_parsed
            reason = parsed.get("reason") or reason
            fix = parsed.get("fix") or ""
        except Exception:
            pass

    if verdict == "retry":
        force_broaden = fix in {"broaden", "cold"} or not recs
        fallback = _autonomous_fallback_recs(state, force_broaden=force_broaden)
        if fallback:
            recs = fallback
            reason = f"{reason} | Critic requested retry; broadened search."
            user_id = str(state.get("userId", ""))
            last_recs_by_user[user_id] = [m["id"] for m in recs]
            persist_state_to_disk()
        else:
            reason = f"{reason} | Retry requested but no better options found."

    return {"recommendations": recs, "critic": {"verdict": verdict, "reason": reason, "fix": fix}}

def _node_explain(state: Dict[str, object]) -> Dict[str, object]:
    intent = state.get("intent", {}) or {}
    plan = state.get("plan", {}) or {}
    critic = state.get("critic", {}) or {}
    user_id = state.get("userId", "")
    recs: List[Movie] = state.get("recommendations", []) or []
    client = get_openai_client()
    choice_ack = state.get("choice_ack", "")
    decision = {
        "strategy": ", ".join(plan.get("tools", [])) or "content",
        "reason": plan.get("reason") or intent.get("rationale"),
        "intent": intent,
        "mainGenre": state.get("mainGenre"),
        "coverage": state.get("coverage"),
        "choice_ack": choice_ack,
        "critic": critic,
        "iteration": state.get("iteration") or 1,
    }
    if client and recs:
        try:
            rec_lines = [
                f"{idx+1}. {m['title']} ({', '.join(m['genres'])}): {m.get('summary', '')}"
                for idx, m in enumerate(recs)
            ]
            messages = [
                {
                    "role": "system",
                    "content": "You are a friendly movie recommender. Summarize picks and why they fit.",
                },
                {
                    "role": "user",
                    "content": "\n".join(
                        [
                            f"Intent: {intent.get('label')}",
                            f"Intent rationale: {intent.get('rationale')}",
                            f"Plan tools: {plan.get('tools')}",
                            f"Plan reason: {plan.get('reason')}",
                            f"Main genre: {decision.get('mainGenre')}",
                            f"Critic verdict: {critic.get('verdict')}",
                            f"Critic reason: {critic.get('reason')}",
                            f"Ack: {choice_ack}" if choice_ack else "",
                            "Recommendations:",
                            *rec_lines,
                            "Ask the user: Which one would you actually watch? Reply with a number, or 'none' if nothing fits.",
                        ]
                    ),
                },
            ]
            completion = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                max_tokens=180,
                temperature=0.7,
            )
            content = completion.choices[0].message.content or ""
            if content.strip():
                return {"response": content, "decision": decision}
        except Exception:
            pass

    # Embed choice ack into decision so format_response can surface it
    if choice_ack:
        decision = {**decision, "choice_ack": choice_ack}
    response_text = format_response(decision, recs, str(user_id))
    return {"response": response_text, "decision": decision}


graph_builder = StateGraph(dict)
graph_builder.add_node("choice", _node_choice)
graph_builder.add_node("intent", _node_intent)
graph_builder.add_node("profile", _node_profile)
graph_builder.add_node("planner", _node_planner)
graph_builder.add_node("content", _node_content)
graph_builder.add_node("adjacent", _node_adjacent)
graph_builder.add_node("cold", _node_cold)
graph_builder.add_node("ranker", _node_ranker)
graph_builder.add_node("critic", _node_critic)
graph_builder.add_node("explain", _node_explain)

# Flow: choice -> intent -> profile -> planner -> content -> adjacent -> cold -> ranker -> critic -> explain -> END
graph_builder.set_entry_point("choice")
graph_builder.add_edge("choice", "intent")
graph_builder.add_edge("intent", "profile")
graph_builder.add_edge("profile", "planner")
graph_builder.add_edge("planner", "content")
graph_builder.add_edge("content", "adjacent")
graph_builder.add_edge("adjacent", "cold")
graph_builder.add_edge("cold", "ranker")
graph_builder.add_edge("ranker", "critic")
graph_builder.add_edge("critic", "explain")
graph_builder.add_edge("explain", END)
recommendation_graph = graph_builder.compile()


@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/chat")
async def chat(body: ChatRequest):
    attempts = 2
    final_state: Dict[str, object] = {}
    critic_verdict = ""
    for attempt in range(attempts):
        state = {
            "userId": body.userId,
            "message": body.message,
            "retry_broaden": attempt > 0,
            "retry_note": final_state.get("critic", {}).get("reason", "") if attempt > 0 else "",
            "iteration": attempt + 1,
        }
        result_state = recommendation_graph.invoke(state)
        critic = result_state.get("critic", {}) or {}
        critic_verdict = critic.get("verdict", "")
        final_state = result_state
        if critic_verdict != "retry":
            break

    decision = final_state.get("decision", {}) or {}
    recs: List[Movie] = final_state.get("recommendations", []) or []
    response_text = final_state.get("response", "")
    return {
        "strategy": decision.get("strategy"),
        "decision": decision,
        "response": response_text,
        "critic_verdict": critic_verdict,
        "iterations": final_state.get("iteration") or (2 if critic_verdict == "retry" else 1),
        "recommendations": [
            {"id": m["id"], "title": m["title"], "genres": m["genres"]}
            for m in recs
        ],
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
