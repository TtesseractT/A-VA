# core/memcards.py
from __future__ import annotations
import json, re, time
from typing import List, Dict, Any, Optional
import config
from core.llm import get_llm

FACTS_TABLE = "facts_kv"

# Block phrases that leak AI/assistant boilerplate into identity
_IDENTITY_BLOCK = [
    r"\bas an ai\b", r"\bi am an ai\b", r"\bmy role\b", r"\bi'?m here to help\b",
    r"\bassistant\b", r"\blanguage model\b", r"\bi (?:was|am) designed\b",
]
_IDENTITY_RX = [re.compile(p, re.I) for p in _IDENTITY_BLOCK]

# Heuristic task/notes catchers
TASK_TRIGGERS = re.compile(
    r"\b(remember|remind(?: me)?|note(?: this)?|add that|make a note|todo|to-do|task[: ]|please remember)\b",
    re.I
)
SENT_SPLIT = re.compile(r"[;\n]|(?<!\b(?:e\.g|i\.e))\.\s+")

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

def _clean_text(s: str, n: int = 220) -> str:
    s = (s or "").strip()
    s = re.sub(r"\s{2,}", " ", s)
    return s[:n]

def _canonicalise(subject: str, key: str, value: str, user_id: str) -> tuple[str,str,str]:
    a_name = getattr(config, "ASSISTANT_NAME", "Ava").strip()
    subj = (subject or "").lower().strip()
    if subj in {"you","your","assistant", a_name.lower()}:
        subj = "assistant"
    elif subj in {"i","me","my","mine"}:
        subj = user_id
    key = (key or "").strip().lower()
    key = key.replace(" ", "_")
    key = re.sub(r"[^a-z0-9_\-:.]", "", key)
    if key.startswith("like:") or key.startswith("likes:"):
        key = "likes:" + key.split(":",1)[1]
    if key.startswith("dislike:") or key.startswith("hates:"):
        key = "dislikes:" + key.split(":",1)[1]
    value = _clean_text(value, 220)
    return subj, key, value

def _keep_fact(key: str, value: str) -> bool:
    if not key or not value:
        return False
    if _looks_like_ai_identity(value) or _looks_like_ai_identity(key):
        return False
    return True

def _parse_llm_json(s: str) -> List[Dict[str,Any]]:
    try:
        obj = json.loads(s)
        if isinstance(obj, dict) and "facts" in obj and isinstance(obj["facts"], list):
            return obj["facts"]
        if isinstance(obj, list):
            return obj
        return []
    except Exception:
        m = re.search(r"\{.*\}|\[.*\]", s, flags=re.S)
        if not m: return []
        try:
            obj = json.loads(m.group(0))
            if isinstance(obj, dict) and "facts" in obj and isinstance(obj["facts"], list):
                return obj["facts"]
            if isinstance(obj, list):
                return obj
        except Exception:
            return []
    return []

def _llm_extract_facts(user_id: str, text: str) -> List[Dict[str,Any]]:
    """
    Ask the local LLM to return open-ended conversational facts AND tasks/notes.
    Each fact:
      { "subject": "assistant|you|user|I|<user_name>",
        "key": "<label>", "value": "<value>", "confidence": 0..1 }
    Examples of keys: user.name, likes:music, dislikes:loud_noises, tone:plain,
                      todo:<slug>, note:<slug>, project:name, project:deadline, etc.
    """
    if not text or len(text) < 2:
        return []
    llm = get_llm(getattr(config, "MODEL_BACKEND", "ollama"))
    a_name = getattr(config, "ASSISTANT_NAME", "Assistant")
    sys = (
      "Extract explicit conversational facts AND any tasks/notes the user asks to remember, as JSON:\n"
      "{ \"facts\": [ {\"subject\": \"assistant|you|user|I|<user_name>\","
      "\"key\": \"short_snake_or_namespace_label\", \"value\": \"short_value\", \"confidence\": 0.0-1.0 } ] }\n"
      "Include tasks/reminders when the user says things like 'remember', 'remind me', 'note this', 'add that', 'todo'.\n"
      "For tasks, use keys like todo:<slug> or note:<slug> and put the natural-language directive in value.\n"
      "Prefer small, stable attributes and clearly asked notes/tasks. If nothing factual, return {\"facts\":[]}."
    )
    user = f"AssistantName={a_name}\nUserId={user_id}\nMessage: {text}"
    out = llm.generate_complete(sys, user, options={"temperature":0.2, "num_predict":220}, max_segments=2)
    facts = _parse_llm_json(out)
    rows = []
    now = time.time()
    for f in facts:
        subj = str(f.get("subject","")).strip() or "user"
        key  = str(f.get("key","")).strip()
        val  = str(f.get("value","")).strip()
        conf = float(f.get("confidence", 0.75))
        subj,key,val = _canonicalise(subj, key, val, user_id=user_id)
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

def _heuristic_task_fallback(user_id: str, text: str) -> List[Dict[str,Any]]:
    """If the LLM misses it, heuristically capture 'remember/note/todo' lines into note:/todo: facts."""
    if not text or not TASK_TRIGGERS.search(text):
        return []
    now = time.time()
    items: List[Dict[str,Any]] = []
    # Split into sentences-ish and pull the segments around the trigger
    for seg in [s.strip() for s in SENT_SPLIT.split(text) if s.strip()]:
        if TASK_TRIGGERS.search(seg):
            # Normalise some common phrasing
            clean = re.sub(r"^\b(please|can you|could you|would you)\b", "", seg, flags=re.I).strip()
            clean = re.sub(r"\b(remember|remind(?: me)?|note(?: this)?|add that|make a note|todo|to-do|task[: ]?)\b[:,-]?\s*", "", clean, flags=re.I).strip()
            if not clean:
                continue
            # Build a slug key
            slug = re.sub(r"[^a-z0-9]+", "_", clean.lower()).strip("_")
            key  = f"todo:{slug}" if "change" in clean.lower() or "need to" in clean.lower() else f"note:{slug}"
            items.append({
                "user_id": user_id,
                "subject": user_id,
                "key": key[:64],                 # keep key tidy
                "value": _clean_text(clean, 220),
                "confidence": 0.80,
                "last_seen": now,
                "usage_count": 0
            })
    # de-dup on (key,value)
    ded = {}
    for it in items:
        k = (it["key"], it["value"])
        if k not in ded:
            ded[k] = it
    return list(ded.values())

def extract_cards(user_id: str, text: str) -> List[Dict]:
    """Dynamic extraction: LLM facts + heuristic task/notes fallback + minimal name capture."""
    cards = _llm_extract_facts(user_id, text)

    # Heuristic notes/tasks if LLM missed them
    tasks = _heuristic_task_fallback(user_id, text)
    cards.extend(tasks)

    # Minimal seed: "my name is X" / "call me X"
    if not cards:
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
    dedup = {}
    for c in cards:
        k = (c.get("subject",""), c.get("key",""), c.get("value",""))
        if not c.get("key") or not c.get("value"):
            continue
        if k not in dedup or float(c.get("confidence",0)) > float(dedup[k].get("confidence",0)):
            dedup[k] = c
    return list(dedup.values())

# ---------- upsert / recall / mark use ----------
def upsert_cards(db, cards: List[Dict]) -> None:
    if not cards: return
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
            # replace weaker prior values on same (subject,key)
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
    if k is None: k = getattr(config, "MEMORY_MAX_CARDS", 12)
    bias = getattr(config, "MEMORY_RECENCY_BIAS", 1.15)
    con=db.con; _ensure_table(con)
    rows=con.execute(
        f"SELECT user_id,subject,key,value,confidence,last_seen,usage_count FROM {FACTS_TABLE} "
        f"WHERE user_id=? OR subject='assistant'",
        (user_id,)
    ).fetchall()
    if not rows: return []
    now=time.time()
    scored=[]
    for r in rows:
        conf=float(r[4] or 0.7); usage=int(r[6] or 0)
        age=max(1.0, now-float(r[5] or now)); rec=1.0/age
        score=(conf**1.1)*(1.0+0.15*usage)*(rec**bias)
        scored.append((score, r))
    scored.sort(key=lambda x:-x[0]); rows=[r for _,r in scored[:k]]
    return [{
        "user_id": r[0], "subject": r[1], "key": r[2], "value": r[3],
        "confidence": r[4], "last_seen": r[5], "usage_count": r[6],
    } for r in rows]

def recall_notes(db, user_id: str, k: int = 10) -> List[Dict[str,Any]]:
    """Return the most recent todo:/note: items for the user."""
    con = db.con; _ensure_table(con)
    rows = con.execute(
        f"""
        SELECT user_id,subject,key,value,confidence,last_seen,usage_count
        FROM {FACTS_TABLE}
        WHERE user_id=? AND (key LIKE 'todo:%' OR key LIKE 'note:%')
        ORDER BY last_seen DESC
        LIMIT ?
        """,
        (user_id, int(k))
    ).fetchall()
    return [{
        "user_id": r[0], "subject": r[1], "key": r[2], "value": r[3],
        "confidence": r[4], "last_seen": r[5], "usage_count": r[6],
    } for r in rows]

def mark_cards_used(db, cards: List[Dict]):
    if not cards: return
    con=db.con; _ensure_table(con)
    for c in cards:
        row=con.execute(
            f"SELECT usage_count FROM {FACTS_TABLE} WHERE user_id=? AND subject=? AND key=? AND value=?",
            (c["user_id"], c["subject"], c["key"], c["value"])
        ).fetchone()
        if row is not None:
            con.execute(
                f"UPDATE {FACTS_TABLE} SET usage_count=? WHERE user_id=? AND subject=? AND key=? AND value=?",
                (int(row[0] or 0)+1, c["user_id"], c["subject"], c["key"], c["value"])
            )

def ensure_user_name_fact(db, user_id: str):
    con=db.con; _ensure_table(con)
    row=con.execute(f"SELECT 1 FROM {FACTS_TABLE} WHERE user_id=? AND key='user.name'", (user_id,)).fetchone()
    if not row:
        name=user_id.strip().title()
        con.execute(
            f"INSERT INTO {FACTS_TABLE} (user_id,subject,key,value,confidence,last_seen,usage_count) VALUES (?,?,?,?,?,?,?)",
            (user_id, user_id, "user.name", name, 0.85, time.time(), 0)
        )

def get_user_prefs(db, user_id: str):
    con=db.con; _ensure_table(con)
    flags={"pref.brief":False,"pref.low_positive":False,"pref.plain":False}
    for k in list(flags.keys()):
        row=con.execute(f"SELECT 1 FROM {FACTS_TABLE} WHERE user_id=? AND key=?", (user_id,k)).fetchone()
        flags[k]=bool(row)
    return flags
