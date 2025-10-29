
# scripts/seed_identity.py - seed identity strictly from wizard output; no hard-coded traits/emotions.
import time
from typing import List, Dict, Any
from pathlib import Path

try:
    import click
except Exception as e:
    raise SystemExit("Missing dependency: click\nInstall with: pip install click")

import config
from stores.tabular import TabularStore
from stores.vector_store import VectorStore
from stores.graph_store import GraphStore
from core.traits import TraitEngine
from core.embeddings import embed_texts
from core.memcards import upsert_cards, ensure_user_name_fact

# Wizard persona helpers
try:
    from core.onboarding import load_persona, load_identity
except Exception:
    def load_persona() -> Dict[str, Any]: return {}
    def load_identity() -> Dict[str, Any]: return {}

def _now() -> float: return time.time()

def _sanitize(s: str) -> str:
    return " ".join((s or "").strip().split())

def _first_words(s: str, n: int = 12) -> str:
    w = _sanitize(s).split()
    return " ".join(w[:n])

@click.command()
@click.option("--user", "user_id", required=True, help="User id to bind identity under")
def main(user_id):
    vs = VectorStore(dim=getattr(config, "EMBEDDING_DIM", 1024),
                     path=config.FAISS_PATH,
                     normalize=getattr(config, "EMBEDDING_NORMALIZE", True)); vs.load()
    gs = GraphStore(config.GRAPH_PATH); gs.load()
    db = TabularStore(config.DB_PATH)
    te = TraitEngine(config.TRAITS_PATH)

    persona = load_persona() or {}
    identity = load_identity() or {}
    ai_name = identity.get("ai_name") or persona.get("ai_name") or getattr(config, "ASSISTANT_NAME", "Assistant")
    role = persona.get("role") or "chat_assistant"
    custom_role = persona.get("custom_role_desc") or ""

    ensure_user_name_fact(db, user_id)

    # Seed facts from wizard only (no traits/emotions here)
    facts: List[Dict[str, Any]] = []
    now = _now()
    facts.append({"user_id": user_id, "subject": "assistant", "key": "assistant.name", "value": ai_name, "confidence": 0.98, "last_seen": now, "usage_count": 0})
    if role:
        facts.append({"user_id": user_id, "subject": "assistant", "key": "assistant.role", "value": role, "confidence": 0.95, "last_seen": now, "usage_count": 0})
    if role == "custom" and custom_role:
        facts.append({"user_id": user_id, "subject": "assistant", "key": "assistant.role_desc", "value": custom_role, "confidence": 0.92, "last_seen": now, "usage_count": 0})

    # Optional: mirror objectives/skills as facts
    for o in (persona.get("objectives") or []):
        if o: facts.append({"user_id": user_id, "subject": "assistant", "key": "objective", "value": _sanitize(o), "confidence": 0.9, "last_seen": now, "usage_count": 0})
    for s in (persona.get("skills") or []):
        if s: facts.append({"user_id": user_id, "subject": "assistant", "key": "skill", "value": _sanitize(s), "confidence": 0.9, "last_seen": now, "usage_count": 0})

    if facts: upsert_cards(db, facts)

    # Build identity episode from wizard persona only
    parts = [f"{ai_name} is initialised."]
    if role == "custom" and custom_role:
        parts.append(f"Primary function: {custom_role}.")
    else:
        parts.append(f"Primary function: {role.replace('_',' ')}.")
    backstory = _sanitize(persona.get("backstory") or "")
    if backstory: parts.append(f"Background: {backstory}")
    objectives = [o for o in (persona.get("objectives") or []) if o]
    if objectives: parts.append("Objectives: " + ", ".join(objectives) + ".")
    skills = [s for s in (persona.get("skills") or []) if s]
    if skills: parts.append("Skills: " + ", ".join(skills) + ".")
    id_text = " ".join(parts).strip()

    qvec = embed_texts([id_text], getattr(config, "EMBEDDING_MODEL", "BAAI/bge-large-en-v1.5"))[0]

    import uuid
    eid = str(uuid.uuid4())
    ts = _now()
    row = {
        "id": eid, "user_id": user_id, "ts": float(ts),
        "summary": id_text[: getattr(config, "EPISODE_SUMMARY_CLAMP", 400)],
        "vector_dim": int(getattr(config, "EMBEDDING_DIM", 1024)),
        "strength": 0.6, "usage_count": 0, "last_accessed": float(ts),
        "gist": _first_words(id_text, 18), "emotion": ""  # no emotion seeding
    }
    db.insert_episode(row)

    vs.add(qvec.reshape(1,-1), [eid])
    if not gs.has_node(eid):
        gs.add_node(eid, type="Episode", user_id=user_id, ts=float(ts))

    # No trait creation here; wizard/real-time runtime handles traits
    try: vs.save()
    except Exception: pass
    try: gs.save()
    except Exception: pass
    try: db.export_snapshots(config.DATA_DIR)
    except Exception: pass

    print(f"Seeded identity for '{ai_name}' under user '{user_id}'. Episode={eid}")

if __name__ == "__main__":
    main()
