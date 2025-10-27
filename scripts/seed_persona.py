# scripts/seed_persona.py
from __future__ import annotations
import time, uuid, yaml
from pathlib import Path

import config
from core.embeddings import embed_texts
from core.memcards import upsert_cards
from core.traits import TraitEngine
from stores.vector_store import VectorStore
from stores.graph_store import GraphStore
from stores.tabular import TabularStore

def _cards_from_yaml(user_id: str, facts):
    now=time.time()
    out=[]
    for f in facts or []:
        subj=f.get("subject","").strip().lower()
        key =f.get("key","").strip()
        val =str(f.get("value","")).strip()
        conf=float(f.get("confidence",0.8))
        out.append({
            "user_id": user_id,
            "subject": ("assistant" if subj=="assistant" else user_id),
            "key": key, "value": val,
            "confidence": conf, "last_seen": now, "usage_count": 0
        })
    return out

def _insert_episode(db, vs, gs, user_id: str, text: str, emotion: str = "", strength: float = 0.7):
    vec = embed_texts([text], config.EMBEDDING_MODEL)[0]
    eid = str(uuid.uuid4())[:8]
    row = {
        "id": eid, "user_id": user_id, "ts": time.time(),
        "summary": text[:280], "vector_dim": len(vec),
        "strength": strength, "usage_count": 0, "last_accessed": time.time(),
        "gist": text[:200], "emotion": emotion or ""
    }
    db.insert_episode(row)
    vs.add(vec.reshape(1,-1), [eid])
    try:
        if not gs.has_node(eid): gs.add_node(eid, type="Episode")
    except Exception:
        gs.add_node(eid, type="Episode")
    gs.add_edge("seed", eid, type="IDENTITY", weight=strength)
    return eid

def main(yaml_path: str, user_id: str):
    ypath = Path(yaml_path)
    assert ypath.exists(), f"persona file not found: {ypath}"

    data = yaml.safe_load(ypath.read_text(encoding="utf-8")) or {}
    facts      = data.get("facts", [])
    traits_cfg = data.get("traits", [])
    episodes   = data.get("episodes", [])
    ident_pack = (data.get("identity_pack") or "").strip()

    # stores
    vs = VectorStore(dim=config.EMBEDDING_DIM, path=config.FAISS_PATH, normalize=config.EMBEDDING_NORMALIZE); vs.load()
    gs = GraphStore(config.GRAPH_PATH); gs.load()
    db = TabularStore(config.DB_PATH)
    te = TraitEngine(config.TRAITS_PATH)

    # 1) facts
    cards = _cards_from_yaml(user_id, facts)
    upsert_cards(db, cards)

    # 2) traits
    name_to_idx = {t["name"].lower(): i for i,t in enumerate(te.traits)}
    for t in traits_cfg:
        nm  = str(t.get("name","")).strip()
        strn= float(t.get("strength", 0.5))
        if not nm: continue
        k = nm.lower()
        if k in name_to_idx:
            te.traits[name_to_idx[k]]["strength"] = strn
        else:
            te.traits.append({"name": nm, "strength": strn, "usage_count": 0, "last_used": time.time()})
    te.save()

    # 3) identity episodes
    for ep in episodes or []:
        _insert_episode(db, vs, gs, user_id, ep.get("text","").strip(), ep.get("emotion",""), strength=0.8)

    # 4) identity pack as a strong “MemoryPack” episode
    if ident_pack:
        txt = "MemoryPack: " + " ".join(line.strip("• ").strip() for line in ident_pack.splitlines() if line.strip())
        _insert_episode(db, vs, gs, user_id, txt, emotion="neutral", strength=0.9)

    # persist
    try: vs.save()
    except Exception as e: print("[warn] vector save failed:", e)
    gs.save()
    try: db.export_snapshots(config.DATA_DIR)
    except Exception: pass

    print("Persona seeded for user:", user_id)

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("yaml_path")
    ap.add_argument("--user", required=True)
    a = ap.parse_args()
    main(a.yaml_path, a.user)
