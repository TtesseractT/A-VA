# app.py (runtime-only, wizard-driven; no emotion/sys-tone wiring)
# -----------------------------------------------------------------------------
# - No hard-coded emotional tone or latent/thinking hooks.
# - All persona/style/length behaviour comes from the setup wizard (data/*.json).
# - Runtime traits and facts are added through normal use (TraitEngine + memcards).
# - Recent dialogue context includes the last 30 interactions (outside memories).
# -----------------------------------------------------------------------------

import uuid, time, re, numpy as np, random
from typing import Optional, List, Dict, Any, Tuple
import yaml
from collections import deque

import click
from rich import print as rprint
from rich.prompt import Prompt

import config
from core.embeddings import embed_texts
from core.traits import TraitEngine
from core import retrieval
from core.memcards import (
    extract_cards, upsert_cards, recall_cards, mark_cards_used,
    ensure_user_name_fact, get_user_prefs
)
from stores.vector_store import VectorStore
from stores.graph_store import GraphStore
from stores.tabular import TabularStore
from core.llm import get_llm
from core.realtime import EventBus

# Wizard-driven persona helpers (no mood/emotion logic here)
from core.persona import decode_overrides, system_primer
from core.onboarding import load_identity  # for assistant label

# -------------------- working memory (recent dialogue buffer) --------------------
class DialogueBuffer:
    """
    Holds last N user/assistant turns for direct conversational context only.
    This is separate from memory/facts recall and is always included.
    """
    def __init__(self, max_turns: int = 30, max_chars: int = 240):
        self.max_turns = max_turns
        self.max_chars = max_chars
        self.lines: deque[tuple[str, str]] = deque(maxlen=max_turns * 2)  # user/assistant pairs

    def add(self, role: str, text: str):
        if not text:
            return
        t = re.sub(r"\s+", " ", text).strip()
        if len(t) > self.max_chars:
            t = t[: self.max_chars].rstrip() + "…"
        self.lines.append((role, t))

    def render(self, user_label: str, assistant_label: str) -> str:
        if not self.lines:
            return ""
        parts = [" Recent dialogue:"]
        for role, text in list(self.lines)[-self.max_turns * 2:]:
            label = user_label if role == "user" else assistant_label
            parts.append(f" {label}: {text}")
        return " " + " | ".join(parts) + "."

# -------------------- anti-repeat --------------------
def _jaccard(a: str, b: str) -> float:
    tokn = lambda s: set(re.findall(r"[a-z0-9']+", s.lower()))
    A, B = tokn(a), tokn(b)
    if not A or not B:
        return 0.0
    return len(A & B) / max(1, len(A | B))

def too_similar(new_text: str, history: List[str], thr: float = 0.75) -> bool:
    if not new_text or not history:
        return False
    return any(_jaccard(new_text, h) >= thr for h in history[-2:])

# -------------------- memory gating --------------------
def memory_relevance_gate(msg: str, final_pairs: List[Tuple[str, float]], db, *,
                          min_sim=None, min_hits=None, jacc=None, topn=None):
    """Heuristic: when to include memory gists/facts for this turn."""
    min_sim = min_sim if min_sim is not None else getattr(config, "MEMORY_GATE_MIN_SIM", 0.30)
    min_hits = min_hits if min_hits is not None else getattr(config, "MEMORY_GATE_MIN_HITS", 1)
    jacc    = jacc if jacc is not None else getattr(config, "MEMORY_GATE_JACCARD", 0.07)
    topn    = topn if topn is not None else getattr(config, "MEMORY_GATE_GISTS", 8)

    cue_words = getattr(config, "MEMORY_ALWAYS_ON_CUES", ["earlier","before","as we said","again","that thing"])
    mlow = (msg or "").lower()
    if any(c in mlow for c in cue_words):
        ids = [eid for eid, _ in (final_pairs or [])[:topn]]
        gists = []
        for eid in ids:
            row = db.con.execute("SELECT gist, summary FROM episodes WHERE id=?", (eid,)).fetchone()
            if row:
                g = (row[0] or row[1] or "").strip()
                if g:
                    gists.append(g)
        return True, gists[:3]

    smax = max((score for _, score in (final_pairs or [])), default=0.0)
    if smax < float(min_sim):
        return False, []

    ids = [eid for eid, _ in (final_pairs or [])[:topn]]
    hits, matched = 0, []
    for eid in ids:
        row = db.con.execute("SELECT gist, summary FROM episodes WHERE id=?", (eid,)).fetchone()
        if not row:
            continue
        g = (row[0] or row[1] or "").strip()
        if not g:
            continue
        if _jaccard(msg, g) >= float(jacc):
            hits += 1
            matched.append(g)
    return (hits >= int(min_hits)), matched[:3]

# -------------------- retrieval helpers --------------------
def assemble_context(user_id: str, msg: str, vs, gs, db, te: TraitEngine):
    """Return (query_vector, candidate_ids, active_trait_names, final_pairs)."""
    _last = db.list_episodes(user_id=user_id, limit=1)
    _gist = (_last[0][8] if _last else "") or ""  # tolerate schema; gist usually column 8
    _query_text = (msg + " || last: " + _gist) if _gist else msg
    qvec = embed_texts([_query_text], config.EMBEDDING_MODEL)[0]

    final = retrieval.expand_candidates(
        qvec, msg, vs, db,
        k_candidates=getattr(config, "CANDIDATE_K", 120),
        k_final=getattr(config, "FINAL_K", 16)
    )
    seed_ids = [eid for eid, _ in final]
    _ = retrieval.spreading_activation(seed_ids, gs, depth=config.SPREAD_DEPTH, decay=config.SPREAD_DECAY)

    trait_names = te.active_traits(qvec, m=config.TRAIT_BULLETS, alpha=config.ALPHA_USAGE, beta=config.BETA_TIME)
    return qvec, seed_ids, trait_names, final

# -------------------- gist helpers --------------------
def sanitize_gist(text: str) -> str:
    if not text:
        return ""
    t = text.strip().strip('"').strip("'")
    t = re.sub(r"\b(as an ai|assistant|language model|computer program)\b", "", t, flags=re.I)
    t = re.sub(r"^\s*i\s+am\b", "said they are", t, flags=re.I)
    t = re.sub(r"^\s*i'?m\b", "said they are", t, flags=re.I)
    t = re.sub(r"\s{2,}", " ", t).strip()
    words = t.split()
    t = " ".join(words[:12]).rstrip(" ,;:-")
    return t

def make_gist(llm, msg: str) -> str:
    sys = (
        "Rewrite the user's message as a short neutral memory note. "
        "6-12 words. No opinions. No metaphors. "
        "No identity or role talk. "
        "Do not invent details beyond what was just said. Output only the note."
    )
    g = llm.generate_complete(sys, msg, options={"temperature": 0.3, "num_predict": 40}, max_segments=1)
    return sanitize_gist(g)

# -------------------- graph helpers --------------------
def nearest_episodes(vs, vec: np.ndarray, topk: int, exclude_id: str = None):
    if vs.vecs is None or vs.vecs.shape[0] == 0:
        return []
    v = vec.astype(np.float32).reshape(1, -1)
    sims = (vs.vecs @ v.T).ravel()
    ids = vs.ids
    pairs = []
    for i, s in enumerate(sims):
        eid = ids[i]
        if exclude_id and eid == exclude_id:
            continue
        pairs.append((eid, float(s)))
    pairs.sort(key=lambda x: x[1], reverse=True)
    return pairs[:max(0, int(topk))]

def concept_id(key: str, value: str) -> str:
    return f"concept::{key}::{value.lower()}"

def add_concept(gs, key: str, value: str):
    cid = concept_id(key, value)
    if not gs.has_node(cid):
        gs.add_node(cid, type="Concept", key=key, value=value)
    return cid

def connect_episode_graph(eid: str, vec, cards, vs, gs):
    # Similar-episode edges
    nbrs = nearest_episodes(vs, vec, topk=getattr(config, "SIM_EP_EDGE_TOPK", 5), exclude_id=eid)
    for nid, sim in nbrs:
        if sim >= getattr(config, "SIM_EP_EDGE_MIN", 0.28):
            gs.add_edge(eid, nid, type="SIM", weight=float(sim))

    # Concept edges (facts mentioned this turn)
    added_concepts = []
    base = getattr(config, "EP_CONCEPT_EDGE_BASE", 0.4)
    for c in cards or []:
        key = c["key"]; val = (c["value"] or "").strip()
        if not key or not val:
            continue
        cid = add_concept(gs, key, val)
        w = max(0.05, min(1.0, base * float(c.get("confidence", 0.9))))
        gs.add_edge(eid, cid, type="MENTIONS", weight=w)
        added_concepts.append(cid)

    # Implication edges (domain rules)
    for c in cards or []:
        key = c["key"]; val = (c["value"] or "").strip().lower()
        for (src_k, src_v), outs in getattr(config, "IMPLICATIONS", {}).items():
            if key == src_k and val == src_v:
                cid_src = add_concept(gs, key, val)
                for out_k, out_v in outs:
                    cid_out = add_concept(gs, out_k, out_v)
                    gs.add_edge(cid_src, cid_out, type="RELATED", weight=0.4)

    # Co-occur edges between concepts mentioned in the same turn
    co_w = getattr(config, "CO_OCCUR_EDGE_WEIGHT", 0.18)
    for i in range(len(added_concepts)):
        for j in range(i + 1, len(added_concepts)):
            gs.add_edge(added_concepts[i], added_concepts[j], type="CO_OCCUR", weight=co_w)

# -------------------- persistence per turn --------------------
def write_episode(db, vs, gs, user_id: str, msg: str, qvec: np.ndarray,
                  te: TraitEngine, llm, cards: Optional[List[dict]] = None) -> str:
    eid = str(uuid.uuid4())
    ts = time.time()
    gist = make_gist(llm, msg)

    row = {
        "id": eid,
        "user_id": user_id,
        "ts": float(ts),
        "summary": msg.strip()[: getattr(config, "EPISODE_SUMMARY_CLAMP", 400)],
        "vector_dim": int(getattr(config, "EMBEDDING_DIM", 1024)),
        "strength": 0.5,
        "usage_count": 0,
        "last_accessed": float(ts),
        "gist": gist,
        # No emotion detection/writes. If your DB schema has 'emotion', it can be empty:
        "emotion": "",
    }
    db.insert_episode(row)

    # Vector add + graph wiring (no mood links)
    vs.add(qvec.astype(np.float32).reshape(1, -1), [eid])
    if not gs.has_node(eid):
        gs.add_node(eid, type="Episode", user_id=user_id, ts=float(ts))
    connect_episode_graph(eid, qvec, cards or [], vs, gs)
    return eid

# -------------------- name helpers --------------------
def display_name(db, user_id: str) -> str:
    name = user_id.strip().title()
    mem = recall_cards(db, user_id, k=12)
    for m in mem:
        if m["key"] == "user.name" and m["value"].strip():
            return m["value"].strip()
    return name

def fix_addressing(text: str, name: str) -> str:
    if not text:
        return text
    t = text
    t = re.sub(rf"\bUser\s+{re.escape(name)}\s*:\s*", f"{name}: ", t, flags=re.I)
    t = re.sub(r"\bUser\s*:\s*", "", t, flags=re.I)
    t = re.sub(r"(^|\n)\s*User([,\.!?]\s+)", rf"\1{name}\2", t, flags=re.I)
    t = re.sub(rf"\bUser\s+{re.escape(name)}\b", name, t, flags=re.I)
    t = re.sub(r"\bUser\b(?=[,\.!?])", name, t, flags=re.I)
    t = re.sub(r"\s{2,}", " ", t).strip()
    return t

# -------------------- load stores/llm --------------------
def load_all():
    vs = VectorStore(dim=getattr(config, "EMBEDDING_DIM", 1024),
                     path=config.FAISS_PATH,
                     normalize=getattr(config, "EMBEDDING_NORMALIZE", True))
    vs.load()
    gs = GraphStore(config.GRAPH_PATH); gs.load()
    db = TabularStore(config.DB_PATH)
    te = TraitEngine(config.TRAITS_PATH)
    llm = get_llm(getattr(config, "MODEL_BACKEND", "ollama"))
    return vs, gs, db, te, llm

# -------------------- CLI --------------------
@click.group()
def cli():
    """Trait Engine CLI"""
    pass

# -------------------- chat --------------------
@cli.command()
@click.option("--user", "user_id", required=True, help="User id/name for the conversation")
@click.option("--no-realtime", is_flag=True, default=False, help="Disable realtime pub bus")
def chat(user_id, no_realtime):
    rprint(f"[cyan]LLM backend:[/cyan] {getattr(config, 'MODEL_BACKEND', 'ollama')}")
    rprint(f"[cyan]Chat started for user:[/cyan] {user_id}")

    vs, gs, db, te, llm = load_all()

    ident = load_identity() or {}
    assistant_label = ident.get("ai_name") or getattr(config, "ASSISTANT_NAME", "Assistant")

    dmem = DialogueBuffer(max_turns=30, max_chars=240)
    last_assistant_lines: List[str] = []

    bus = None
    if not no_realtime:
        try:
            bus = EventBus(role="pub")
        except Exception as e:
            print(f"[warn] realtime bus unavailable: {e}")

    ensure_user_name_fact(db, user_id)
    name = display_name(db, user_id)

    # -------------------- conversation loop --------------------
    while True:
        try:
            msg = Prompt.ask(f"[bold green]{name}[/bold green]")
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not msg or msg.strip().lower() in {"quit", "exit", ":q"}:
            break

        dmem.add("user", msg)

        # Retrieve preferences (plain/brief etc.), but do not wire any emotion
        prefs = get_user_prefs(db, user_id)

        # 1) retrieval / traits
        qvec, seed_ids, trait_names, final_pairs = assemble_context(user_id, msg, vs, gs, db, te)

        # 2) system instruction: wizard-driven only (no extra tone/emotion text here)
        system_text = system_primer()

        # 2.5) memory (topic-gated) + facts
        use_mem, matched_gists = memory_relevance_gate(
            msg, final_pairs, db,
            min_sim=getattr(config, "MEMORY_GATE_MIN_SIM", 0.30),
            min_hits=getattr(config, "MEMORY_GATE_MIN_HITS", 1),
            jacc=getattr(config, "MEMORY_GATE_JACCARD", 0.07),
            topn=getattr(config, "MEMORY_GATE_GISTS", 8),
        )
        if use_mem:
            mem = recall_cards(db, user_id, k=getattr(config, "MEMORY_MAX_CARDS", 12))
            if mem:
                kv = []
                for m in mem:
                    k, v = m["key"], m["value"]
                    if k == "user.name": k = "user_name"
                    if k == "assistant.name": k = "assistant_name"
                    kv.append(f"{k}={v}")
                system_text += " Facts: " + "; ".join(kv) + "."
            if matched_gists:
                system_text += " Memory focus: " + " | ".join(matched_gists) + "."

        # 2.6) always include recent dialogue (30 interactions)
        system_text += dmem.render(user_label=name, assistant_label=assistant_label)

        # 3) decoding (wizard controls length; unlimited if set)
        decode_opts = decode_overrides()

        # 4) generate (full context)
        reply = llm.generate_complete(
            system_text,
            msg,
            options=decode_opts,
            max_segments=getattr(config, "CONTINUE_MAX_SEGMENTS", 12)
        )

        # 5) cleanup and anti-repeat
        if too_similar(reply, last_assistant_lines, thr=0.80):
            alt_sys = system_text + " Avoid repeating previous phrases. Add a new concrete angle."
            alt = llm.generate_complete(alt_sys, msg, options=decode_opts, max_segments=4)
            if alt and not too_similar(alt, last_assistant_lines, thr=0.80):
                reply = alt

        reply = fix_addressing(reply, name)
        rprint(f"[magenta]{assistant_label}[/magenta]: {reply}")

        # Update short dialogue memory
        dmem.add("assistant", reply)
        last_assistant_lines.append(reply)
        if len(last_assistant_lines) > 8:
            last_assistant_lines = last_assistant_lines[-8:]

        # 6) facts extraction + persistence (runtime trait updates only)
        cards = extract_cards(user_id, msg) if msg else None
        if cards:
            upsert_cards(db, cards)

        eid = write_episode(db, vs, gs, user_id, msg, qvec, te=te, llm=llm, cards=cards)
        te.discover_new_trait(msg, qvec)
        te.reinforce(trait_names)
        te.save()

        mark_cards_used(db, recall_cards(db, user_id, k=4))
        for sid in seed_ids:
            rows = db.con.execute("SELECT usage_count,strength FROM episodes WHERE id=?", (sid,)).fetchone()
            if rows:
                usage, strength = rows
                db.update_episode_usage(
                    sid,
                    usage_count=int(usage) + 1,
                    last_accessed=time.time(),
                    strength=min(1.0, float(strength) + 0.012),
                )

        # snapshots / realtime (no emotion fields published)
        try: vs.save()
        except Exception as e: print(f"[warn] vector snapshot save failed: {e}")
        try: gs.save()
        except Exception as e: print(f"[warn] graph save failed: {e}")
        try: db.export_snapshots(config.DATA_DIR)
        except Exception: pass
        try:
            if bus:
                ep_count = db.con.execute("SELECT COUNT(*) FROM episodes").fetchone()[0]
                vec_count = int(vs.vecs.shape[0]) if vs.vecs is not None else 0
                recent = db.con.execute(
                    "SELECT id,gist,ts FROM episodes ORDER BY ts DESC LIMIT 5"
                ).fetchall()
                bus.publish("metrics", {"episodes": int(ep_count), "vectors": vec_count})
                bus.publish("recent", [{"id": r[0], "gist": r[1], "ts": float(r[2])} for r in recent])
                bus.publish("traits", {"traits": te.traits})
        except Exception as e:
            print(f"[warn] realtime publish failed: {e}")

# -------------------- seed persona (optional YAML importer) --------------------
@cli.command("seed-persona")
@click.option("--path", "path", required=True, help="Path to persona.yaml")
@click.option("--user", "user_id", required=False, help="Override user id (defaults to persona.user_name or 'default')")
def seed_persona(path, user_id):
    """Optional helper to import facts/traits/episodes from a YAML file."""
    vs, gs, db, te, llm = load_all()

    with open(path, "r", encoding="utf-8") as f:
        y = yaml.safe_load(f) or {}

    user_id = user_id or y.get("user_name") or "default"
    assistant_name = y.get("assistant_name") or getattr(config, "ASSISTANT_NAME", "Assistant")
    rprint(f"[cyan]Seeding persona for user:[/cyan] {user_id}")

    now_ts = time.time()

    # facts
    facts_in = y.get("facts", []) or []
    facts = []
    for item in facts_in:
        subj = str(item.get("subject", "assistant")).strip().lower()
        key  = str(item.get("key", "")).strip()
        val  = str(item.get("value", "")).strip()
        if not key or val == "":
            continue
        facts.append({
            "user_id": user_id,
            "subject": subj,
            "key": key,
            "value": val,
            "confidence": float(item.get("confidence", 0.9)),
            "last_seen": now_ts,
            "usage_count": int(item.get("usage_count", 0)),
        })

    if assistant_name:
        facts.append({
            "user_id": user_id, "subject": "assistant", "key": "assistant.name",
            "value": assistant_name, "confidence": 0.98, "last_seen": now_ts, "usage_count": 0
        })
    facts.append({
        "user_id": user_id, "subject": "user", "key": "user.name",
        "value": str(y.get("user_name", user_id)), "confidence": 0.98, "last_seen": now_ts, "usage_count": 0
    })

    if facts:
        upsert_cards(db, facts)
        rprint(f"[green]Seeded facts:[/green] {len(facts)}")

    # traits
    traits_in = y.get("traits", []) or []
    touched = 0
    te_names = {t["name"].lower(): i for i, t in enumerate(te.traits)}
    for t in traits_in:
        nm = str(t.get("name", "")).strip()
        if not nm:
            continue
        strength = float(t.get("strength", 0.5))
        if nm.lower() in te_names:
            idx = te_names[nm.lower()]
            te.traits[idx]["strength"] = max(strength, float(te.traits[idx].get("strength", 0.0)))
        else:
            te.traits.append({"name": nm, "strength": strength, "usage_count": 0})
        touched += 1
    te.save()
    if touched:
        rprint(f"[green]Seeded/updated traits:[/green] {touched}")

    # episodes + identity pack
    episodes_in = list(y.get("episodes", []) or [])
    id_pack = y.get("identity_pack")
    if isinstance(id_pack, str) and id_pack.strip():
        for raw in [ln.strip(" •\t") for ln in id_pack.splitlines() if ln.strip()]:
            episodes_in.append({"text": raw})

    inserted = 0
    for ep in episodes_in:
        text = str(ep.get("text", "")).strip()
        if not text:
            continue
        qvec = embed_texts([text], config.EMBEDDING_MODEL)[0]
        _ = write_episode(db, vs, gs, user_id, text, qvec, te=te, llm=llm, cards=None)
        inserted += 1
    if inserted:
        rprint(f"[green]Seeded episodes:[/green] {inserted}")

    try:
        vs.save(); gs.save(); db.export_snapshots(config.DATA_DIR)
    except Exception:
        pass

    rprint("[bold green]Persona seeding complete.[/bold green]")

# -------------------- utility commands --------------------
@cli.command()
@click.option("--user", "user_id", required=True)
@click.option("--limit", default=12, show_default=True)
def episodes(user_id, limit):
    _, _, db, _, _ = load_all()
    rows = db.list_episodes(user_id=user_id, limit=limit)
    for r in rows:
        # schema tolerant: id, ts, gist, summary (indexes may differ across versions)
        print(f"- [{r[0]}] {time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime(r[2]))} :: gist={r[8] if len(r) > 8 else ''} :: {r[3]}")

@cli.command()
def traits():
    _, _, _, te, _ = load_all()
    for t in te.traits:
        print(f"- {t['name']}: usage={t.get('usage_count',0)} strength={t.get('strength',0.0):.2f}")

@cli.command()
@click.option("--user", "user_id", required=False)
def facts(user_id):
    _, _, db, _, _ = load_all()
    rows = db.list_facts(user_id=user_id)
    for r in rows:
        print(f"- {r}")

@cli.command()
def snapshot():
    vs, gs, db, te, llm = load_all()
    from pathlib import Path
    db.export_snapshots(config.DATA_DIR)
    print(f"Snapshots written to: {Path(config.DATA_DIR).resolve()}")

if __name__ == "__main__":
    cli()
