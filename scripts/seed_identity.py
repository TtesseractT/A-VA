# scripts/seed_identity.py
# Seed assistant identity from wizard files (no hard-coded content).

import time
from typing import List, Dict, Any

try:
    import click
except Exception as e:
    raise SystemExit("Missing dependency: click\nInstall with: pip install click")

import config
from pathlib import Path
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
    def load_persona() -> Dict[str, Any]:
        return {}
    def load_identity() -> Dict[str, Any]:
        return {}

# ------------- small helpers -------------
def _now() -> float:
    return time.time()

def _sanitize(s: str) -> str:
    return " ".join((s or "").strip().split())

def _first_words(s: str, n: int = 12) -> str:
    w = _sanitize(s).split()
    return " ".join(w[:n])

def _concept_id(key: str, value: str) -> str:
    return f"concept::{key}::{value.lower()}"

def _add_concept(gs: GraphStore, key: str, value: str):
    cid = _concept_id(key, value)
    if not gs.has_node(cid):
        gs.add_node(cid, type="Concept", key=key, value=value)
    return cid

# ------------- identity text (from wizard) -------------
def build_identity_text(p: Dict[str, Any]) -> str:
    """
    Compose a neutral, factual identity paragraph from wizard persona.
    No tone/emotion. Only uses fields if present.
    """
    ai = p.get("ai_name") or "Assistant"
    role = p.get("role") or "chat_assistant"
    custom_role = p.get("custom_role_desc") or ""
    backstory = _sanitize(p.get("backstory") or "")
    objectives = [o for o in (p.get("objectives") or []) if o]
    skills = [s for s in (p.get("skills") or []) if s]

    parts = [f"{ai} is initialised."]
    if role == "custom" and custom_role:
        parts.append(f"Primary function: {custom_role}.")
    else:
        parts.append(f"Primary function: {role.replace('_',' ')}.")
    if backstory:
        parts.append(f"Background: {backstory}")
    if objectives:
        parts.append("Objectives: " + ", ".join(objectives) + ".")
    if skills:
        parts.append("Skills: " + ", ".join(skills) + ".")

    mode_prefs = p.get("mode_prefs") or {}
    if mode_prefs:
        kv = []
        for k, v in mode_prefs.items():
            if isinstance(v, bool):
                kv.append(f"{k}={str(v).lower()}")
            elif isinstance(v, (int, float, str)):
                kv.append(f"{k}={v}")
        if kv:
            parts.append("Mode preferences: " + ", ".join(kv) + ".")
    return " ".join(parts).strip()

# ------------- seeding -------------
@click.command()
@click.option("--user", "user_id", required=True, help="User id to bind identity under")
def main(user_id):
    # Load stores
    vs = VectorStore(dim=getattr(config, "EMBEDDING_DIM", 1024),
                     path=config.FAISS_PATH,
                     normalize=getattr(config, "EMBEDDING_NORMALIZE", True))
    vs.load()
    gs = GraphStore(config.GRAPH_PATH); gs.load()
    db = TabularStore(config.DB_PATH)
    te = TraitEngine(config.TRAITS_PATH)

    # Wizard data
    persona = load_persona() or {}
    identity = load_identity() or {}
    ai_name = identity.get("ai_name") or persona.get("ai_name") or getattr(config, "ASSISTANT_NAME", "Assistant")
    role = persona.get("role") or "chat_assistant"
    custom_role = persona.get("custom_role_desc") or ""

    # Ensure the user has a name fact
    ensure_user_name_fact(db, user_id)

    # Seed stable identity facts (only what the wizard specified)
    facts: List[Dict[str, Any]] = []
    now = _now()

    facts.append({
        "user_id": user_id, "subject": "assistant", "key": "assistant.name",
        "value": ai_name, "confidence": 0.98, "last_seen": now, "usage_count": 0
    })
    if role:
        facts.append({
            "user_id": user_id, "subject": "assistant", "key": "assistant.role",
            "value": role, "confidence": 0.95, "last_seen": now, "usage_count": 0
        })
    if role == "custom" and custom_role:
        facts.append({
            "user_id": user_id, "subject": "assistant", "key": "assistant.role_desc",
            "value": custom_role, "confidence": 0.92, "last_seen": now, "usage_count": 0
        })

    # Optional: mirror wizard objectives/skills as discrete facts (kept simple, no hard-coded content)
    for o in (persona.get("objectives") or []):
        if o:
            facts.append({
                "user_id": user_id, "subject": "assistant", "key": "objective",
                "value": _sanitize(o), "confidence": 0.9, "last_seen": now, "usage_count": 0
            })
    for s in (persona.get("skills") or []):
        if s:
            facts.append({
                "user_id": user_id, "subject": "assistant", "key": "skill",
                "value": _sanitize(s), "confidence": 0.9, "last_seen": now, "usage_count": 0
            })

    if facts:
        upsert_cards(db, facts)

    # Build identity episode text from wizard persona
    id_text = build_identity_text(persona)
    if not id_text:
        id_text = f"{ai_name} is initialised. Primary function: {role.replace('_',' ')}."

    # Vectorise + insert episode (no emotion)
    qvec = embed_texts([id_text], getattr(config, "EMBEDDING_MODEL", "BAAI/bge-large-en-v1.5"))[0]
    eid = _insert_identity_episode(db, vs, gs, user_id, id_text, qvec, facts)

    # Link episode to current traits (no new traits added here)
    _link_episode_to_traits(gs, eid, te.traits)

    # Persist
    try: vs.save()
    except Exception: pass
    try: gs.save()
    except Exception: pass
    try: db.export_snapshots(config.DATA_DIR)
    except Exception: pass

    print(f"Seeded identity for '{ai_name}' under user '{user_id}'. Episode={eid}")

# ------------- episode + graph wiring -------------
def _insert_identity_episode(db: TabularStore, vs: VectorStore, gs: GraphStore,
                             user_id: str, text: str, qvec, facts: List[Dict[str, Any]]) -> str:
    import uuid
    eid = str(uuid.uuid4())
    ts = _now()
    gist = _first_words(text, 12)

    row = {
        "id": eid,
        "user_id": user_id,
        "ts": float(ts),
        "summary": text[: getattr(config, "EPISODE_SUMMARY_CLAMP", 400)],
        "vector_dim": int(getattr(config, "EMBEDDING_DIM", 1024)),
        "strength": 0.6,
        "usage_count": 0,
        "last_accessed": float(ts),
        "gist": gist,
        "emotion": "",  # intentional: no emotion
    }
    db.insert_episode(row)

    # Vector + episode node
    vs.add(qvec.reshape(1, -1), [eid])
    if not gs.has_node(eid):
        gs.add_node(eid, type="Episode", user_id=user_id, ts=float(ts))

    # Concept nodes for seeded facts (assistant.name/role/etc.)
    _link_episode_to_facts(gs, eid, facts)
    return eid

def _link_episode_to_facts(gs: GraphStore, eid: str, facts: List[Dict[str, Any]]):
    base_w = float(getattr(config, "EP_CONCEPT_EDGE_BASE", 0.4))
    for f in facts or []:
        key = f.get("key"); val = (f.get("value") or "").strip()
        if not key or not val:
            continue
        cid = _add_concept(gs, key, val)
        w = max(0.05, min(1.0, base_w * float(f.get("confidence", 0.9))))
        gs.add_edge(eid, cid, type="MENTIONS", weight=w)

def _link_episode_to_traits(gs: GraphStore, eid: str, traits: List[Dict[str, Any]]):
    """
    Create trait nodes (type='Trait') if missing and link the identity episode to them.
    We do NOT add or modify traits here — only link to whatever currently exists.
    """
    for t in traits or []:
        name = (t.get("name") or "").strip()
        if not name:
            continue
        node_id = f"trait::{name.lower()}"
        if not gs.has_node(node_id):
            gs.add_node(node_id, type="Trait", name=name.lower())
        strength = float(t.get("strength", 0.5))
        gs.add_edge(eid, node_id, type="HAS_TRAIT", weight=max(0.05, min(1.0, strength)))

if __name__ == "__main__":
    main()
