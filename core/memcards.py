# core/memcards.py
from __future__ import annotations
import json, re, time
from typing import List, Dict, Any, Optional

import config
from core.llm import get_llm

# Prefer wizard identity (ai_name/user_name) if present
try:
    from core.onboarding import load_identity
except Exception:
    def load_identity() -> Dict[str, Any]:
        return {}

# --------------------------- configurable defaults ---------------------------

FACTS_TABLE: str = getattr(config, "FACTS_TABLE", "facts_kv")

# Block phrases that leak AI/assistant boilerplate into identity (configurable)
DEFAULT_IDENTITY_BLOCK = [
    r"\bas an ai\b", r"\bi am an ai\b", r"\bmy role\b", r"\bi'?m here to help\b",
    r"\bassistant\b", r"\blanguage model\b", r"\bi (?:was|am) designed\b",
]
_IDENTITY_BLOCK = getattr(config, "MEMCARDS_IDENTITY_BLOCK", DEFAULT_IDENTITY_BLOCK)
_IDENTITY_RX = [re.compile(p, re.I) for p in _IDENTITY_BLOCK]

# Optional taboo keys/values (user can set in config; empty by default)
_TABOO_KEYS   = set(getattr(config, "MEMCARDS_TABOO_KEYS", []))
_TABOO_VALUES = set(getattr(config, "MEMCARDS_TABOO_VALUES", []))

# Length clamps
_CLEAN_CLAMP  = int(getattr(config, "MEMCARD_CLEAN_CLAMP", 220))   # generic truncation
_VALUE_CLAMP  = int(getattr(config, "MEMCARD_VALUE_CLAMP", 160))   # value-specific

# LLM generation knobs (kept here to avoid hard-coded values)
# We honour persona.decode_overrides() (unlimited if user chose it).
# If unlimited, we still cap to this many tokens for compact JSON.
_MAX_TOKENS_SOFT = int(getattr(config, "MEMCARDS_NUM_PREDICT", 400))
_TEMPERATURE     = float(getattr(config, "MEMCARDS_TEMPERATURE", 0.2))

# Allowed/encouraged key hints to bias extraction (purely hints; not enforced)
_ALLOWED_KEY_HINTS = getattr(
    config,
    "MEMCARDS_ALLOWED_KEY_HINTS",
    [
        "name", "gender", "pronouns", "age",
        "likes:<topic>", "dislikes:<topic>",
        "tone:plain", "tone:concise",
        "prefs:writing_style", "prefs:coding_style",
        "tools:preferred", "topics:interests",
        "availability", "timezone",
        "contact:email", "contact:handle",
    ],
)

# User preference keys checked elsewhere (configurable)
_USER_PREF_KEYS = getattr(
    config,
    "USER_PREF_KEYS",
    ["pref.brief", "pref.low_positive", "pref.plain"],
)

# -----------------------------------------------------------------------------

def _looks_like_ai_identity(s: str) -> bool:
    return bool(s) and any(rx.search(s) for rx in _IDENTITY_RX)

def _ensure_table(con):
    con.execute(f"""
        CREATE TABLE IF NOT EXISTS {FACTS_TABLE} (
            user_id TEXT,
            subject TEXT,
            key TEXT,
            value TEXT,
            confidence DOUBLE,
            last_seen DOUBLE,
            usage_count INTEGER
        )
    """)

def _clean_text(s: str, n: int = _CLEAN_CLAMP) -> str:
    s = (s or "").strip()
    s = re.sub(r"\s{2,}", " ", s)
    return s[: int(n)]

def _canonicalise(subject: str, key: str, value: str, user_id: str) -> tuple[str, str, str]:
    """
    Map pronouns/names to canonical subjects and tidy keys/values.
    - 'assistant' / AI name -> 'assistant'
    - 'I/me/my' -> the concrete user_id passed by the app
    """
    ident = load_identity() or {}
    a_name = (ident.get("ai_name") or getattr(config, "ASSISTANT_NAME", "Assistant")).strip()

    # subject
    subj = (subject or "").lower().strip()
    if subj in {"you", "your", "assistant", a_name.lower()}:
        subj = "assistant"
    elif subj in {"i", "me", "my", "mine"}:
        subj = user_id

    # key
    key = (key or "").strip().lower()
    key = key.replace(" ", "_")
    key = re.sub(r"[^a-z0-9_\-:.]", "", key)

    # Optional namespacing for preferences
    if key.startswith(("like:", "likes:")):
        key = "likes:" + key.split(":", 1)[1]
    if key.startswith(("dislike:", "hates:")):
        key = "dislikes:" + key.split(":", 1)[1]

    # value
    value = _clean_text(value, _VALUE_CLAMP)

    return subj, key, value

def _keep_fact(key: str, value: str) -> bool:
    """Filter out boilerplate / unsafe identity leakage / taboo patterns."""
    if not key or not value:
        return False
    if key in _TABOO_KEYS or value in _TABOO_VALUES:
        return False
    if _looks_like_ai_identity(value) or _looks_like_ai_identity(key):
        return False
    return True

def _parse_llm_json(s: str) -> List[Dict[str, Any]]:
    try:
        obj = json.loads(s)
        if isinstance(obj, dict) and "facts" in obj and isinstance(obj["facts"], list):
            return obj["facts"]
        if isinstance(obj, list):
            return obj
        return []
    except Exception:
        # try to salvage JSON array/object
        m = re.search(r"\{.*\}|\[.*\]", s, flags=re.S)
        if not m:
            return []
        try:
            obj = json.loads(m.group(0))
            if isinstance(obj, dict) and "facts" in obj and isinstance(obj["facts"], list):
                return obj["facts"]
            if isinstance(obj, list):
                return obj
        except Exception:
            return []
    return []

def _llm_extract_facts(user_id: str, text: str) -> List[Dict[str, Any]]:
    """
    Ask the local LLM to return open-ended conversational facts in JSON.
    Each fact: { "subject": "assistant|you|<AI name>|user|I", "key": "<label>", "value": "<value>", "confidence": 0..1 }
    """
    if not text or len(text) < 2:
        return []

    llm = get_llm(getattr(config, "MODEL_BACKEND", "ollama"))
    ident = load_identity() or {}
    a_name = ident.get("ai_name") or getattr(config, "ASSISTANT_NAME", "Assistant")

    # Build a compact, configurable instruction
    hints = ", ".join(_ALLOWED_KEY_HINTS)
    sys = (
        "Extract concrete conversational facts as JSON with this schema:\n"
        "{ \"facts\": [ {\"subject\": \"assistant|you|<AI name>|user|I|me|<user_id>\","
        "\"key\": \"short_label\", \"value\": \"short_value\", \"confidence\": 0.0-1.0 } ] }\n"
        "Only include facts that are explicitly stated or strongly implied in the message.\n"
        f"Prefer small, stable attributes (e.g., {hints}).\n"
        "Do NOT include meta-identity like 'assistant', 'AI', 'language model', 'here to help'.\n"
        "Keep keys compact (e.g., likes:<topic>, dislikes:<topic>, tone:plain, timezone, contact:email).\n"
        "If nothing factual, return {\"facts\":[]}."
    )

    user = f"AIName={a_name}\nUserId={user_id}\nMessage: {text}"

    # Respect persona decode settings, but keep output compact JSON
    try:
        from core.persona import decode_overrides
        opts = decode_overrides()
    except Exception:
        opts = {}

    # We need structured JSON; if persona allows unlimited, apply a soft cap for safety
    num_predict = opts.get("num_predict", _MAX_TOKENS_SOFT)
    if num_predict in (-1, None):
        num_predict = _MAX_TOKENS_SOFT

    options = dict(opts)
    options.update({
        "temperature": _TEMPERATURE,
        "num_predict": int(num_predict),
    })

    out = llm.generate_complete(sys, user, options=options, max_segments=2)
    facts = _parse_llm_json(out)

    rows: List[Dict[str, Any]] = []
    now = time.time()
    for f in facts:
        subj = str(f.get("subject", "")).strip() or "user"
        key  = str(f.get("key", "")).strip()
        val  = str(f.get("value", "")).strip()
        conf = float(f.get("confidence", 0.7))
        subj, key, val = _canonicalise(subj, key, val, user_id=user_id)
        if not _keep_fact(key, val):
            continue
        rows.append({
            "user_id": user_id,
            "subject": subj,
            "key": key,
            "value": val,
            "confidence": max(0.0, min(1.0, conf)),
            "last_seen": now,
            "usage_count": 0
        })
    return rows

def extract_cards(user_id: str, text: str) -> List[Dict]:
    """
    Dynamic: delegate to LLM to extract open-ended facts. Falls back to a very
    small name capture to seed identity if LLM returns nothing.
    """
    cards = _llm_extract_facts(user_id, text)

    if not cards:
        # minimal seed: "my name is X" / "call me X"
        m = re.search(r"\bmy\s+name\s+is\s+([A-Za-z][A-Za-z\-']{1,30})\b", text or "", re.I)
        if not m:
            m = re.search(r"\bcall\s+me\s+([A-Za-z][A-Za-z\-']{1,30})\b", text or "", re.I)
        if m:
            name = m.group(1).strip().title()
            if name:
                now = time.time()
                cards.append({
                    "user_id": user_id, "subject": user_id, "key": "user.name",
                    "value": name, "confidence": 0.85, "last_seen": now, "usage_count": 0
                })
    # dedup (subject,key,value) by highest confidence
    dedup: Dict[tuple, Dict[str, Any]] = {}
    for c in cards:
        k = (c["subject"], c["key"], c["value"])
        if k not in dedup or c["confidence"] > dedup[k]["confidence"]:
            dedup[k] = c
    return list(dedup.values())

# ---------- upsert / recall / mark use ----------
def upsert_cards(db, cards: List[Dict]) -> None:
    if not cards:
        return
    con = db.con; _ensure_table(con)
    for c in cards:
        row = con.execute(
            f"SELECT value,confidence,usage_count FROM {FACTS_TABLE} "
            f"WHERE user_id=? AND subject=? AND key=? AND value=?",
            (c["user_id"], c["subject"], c["key"], c["value"])
        ).fetchone()
        if row:
            new_conf = min(1.0, max(float(c["confidence"]), float(row[1] or 0.7)) + 0.05)
            con.execute(
                f"UPDATE {FACTS_TABLE} SET confidence=?, last_seen=?, usage_count=? "
                f"WHERE user_id=? AND subject=? AND key=? AND value=?",
                (new_conf, c["last_seen"], int(row[2] or 0) + 1,
                 c["user_id"], c["subject"], c["key"], c["value"])
            )
        else:
            # If a new value for the same (subject,key) exists with lower confidence, replace it
            old = con.execute(
                f"SELECT value,confidence FROM {FACTS_TABLE} "
                f"WHERE user_id=? AND subject=? AND key=? ORDER BY confidence DESC LIMIT 1",
                (c["user_id"], c["subject"], c["key"])
            ).fetchone()
            if old and float(c["confidence"]) > float(old[1]) + 0.15:
                con.execute(
                    f"DELETE FROM {FACTS_TABLE} WHERE user_id=? AND subject=? AND key=?",
                    (c["user_id"], c["subject"], c["key"])
                )
            con.execute(
                f"INSERT INTO {FACTS_TABLE} (user_id,subject,key,value,confidence,last_seen,usage_count) "
                f"VALUES (?,?,?,?,?,?,?)",
                (c["user_id"], c["subject"], c["key"], c["value"],
                 c["confidence"], c["last_seen"], c["usage_count"])
            )

def recall_cards(db, user_id: str, k: Optional[int] = None):
    if k is None:
        k = int(getattr(config, "MEMORY_MAX_CARDS", 12))
    bias = float(getattr(config, "MEMORY_RECENCY_BIAS", 1.15))
    con = db.con; _ensure_table(con)
    rows = con.execute(
        f"SELECT user_id,subject,key,value,confidence,last_seen,usage_count FROM {FACTS_TABLE} "
        f"WHERE user_id=? OR subject='assistant'",
        (user_id,)
    ).fetchall()
    if not rows:
        return []
    now = time.time()
    scored = []
    for r in rows:
        conf = float(r[4] or 0.7)
        usage = int(r[6] or 0)
        age = max(1.0, now - float(r[5] or now))
        rec = 1.0 / age
        score = (conf ** 1.1) * (1.0 + 0.15 * usage) * (rec ** bias)
        scored.append((score, r))
    scored.sort(key=lambda x: -x[0])
    rows = [r for _, r in scored[:k]]
    return [{
        "user_id": r[0], "subject": r[1], "key": r[2], "value": r[3],
        "confidence": r[4], "last_seen": r[5], "usage_count": r[6],
    } for r in rows]

def mark_cards_used(db, cards: List[Dict]):
    if not cards:
        return
    con = db.con; _ensure_table(con)
    for c in cards:
        row = con.execute(
            f"SELECT usage_count FROM {FACTS_TABLE} WHERE user_id=? AND subject=? AND key=? AND value=?",
            (c["user_id"], c["subject"], c["key"], c["value"])
        ).fetchone()
        if row is not None:
            con.execute(
                f"UPDATE {FACTS_TABLE} SET usage_count=? WHERE user_id=? AND subject=? AND key=? AND value=?",
                (int(row[0] or 0) + 1, c["user_id"], c["subject"], c["key"], c["value"])
            )

def ensure_user_name_fact(db, user_id: str):
    con = db.con; _ensure_table(con)
    row = con.execute(
        f"SELECT 1 FROM {FACTS_TABLE} WHERE user_id=? AND key='user.name'",
        (user_id,)
    ).fetchone()
    if not row:
        name = user_id.strip().title()
        con.execute(
            f"INSERT INTO {FACTS_TABLE} (user_id,subject,key,value,confidence,last_seen,usage_count) "
            f"VALUES (?,?,?,?,?,?,?)",
            (user_id, user_id, "user.name", name, 0.85, time.time(), 0)
        )

def get_user_prefs(db, user_id: str):
    con = db.con; _ensure_table(con)
    flags = {k: False for k in _USER_PREF_KEYS}
    for k in list(flags.keys()):
        row = con.execute(f"SELECT 1 FROM {FACTS_TABLE} WHERE user_id=? AND key=?", (user_id, k)).fetchone()
        flags[k] = bool(row)
    return flags
