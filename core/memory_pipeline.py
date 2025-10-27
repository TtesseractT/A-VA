# core/memory_pipeline.py
from __future__ import annotations
import json, re
from typing import List, Dict, Any, Optional, Tuple
import config
from core.llm import get_llm

_SPACES = re.compile(r"\s{2,}")

def _clean(s: str, n: int = 220) -> str:
    if not s: return ""
    return _SPACES.sub(" ", s.strip())[:n]

def _json_from(text: str) -> Any:
    # try strict first
    try:
        return json.loads(text)
    except Exception:
        pass
    # try to find first top-level JSON object/array
    m = re.search(r"(\{.*\}|\[.*\])", text, flags=re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(1))
    except Exception:
        return None

def _within_bounds(start: int, end: int, total: int) -> bool:
    return isinstance(start, int) and isinstance(end, int) and 0 <= start < end <= total

# ============= Proposers =============
def propose_gist(user_text: str, role: str) -> Dict[str, Any]:
    """
    Returns an EXTRACTIVE gist proposal with character offsets.
    We force the model to give start/end indices into the *original* text.
    """
    llm = get_llm(getattr(config, "MODEL_BACKEND", "ollama"))
    max_words = int(getattr(config, "GIST_WORDS", 14))
    sys = (
        "Find a short verbatim substring (<= {N} words) from the message that best captures its core idea. "
        "Return JSON: {{\"start\": int, \"end\": int}}. Do not add or change words. "
        "Indices are 0-based inclusive start, exclusive end over the raw message characters."
    ).format(N=max_words)
    out = llm.generate_complete(sys, user_text, options={"temperature": 0.1, "num_predict": 120}, max_segments=1).strip()
    obj = _json_from(out) or {}
    start = obj.get("start", -1)
    end = obj.get("end", -1)
    # normalise to ints if they arrive as strings
    try: start = int(start)
    except Exception: start = -1
    try: end = int(end)
    except Exception: end = -1

    if not _within_bounds(start, end, len(user_text or "")):
        # fallback: first clause up to punctuation
        txt = (user_text or "").strip()
        m = re.match(r"^(.{1,160}?)(?:[.!?]|$)", txt)
        snippet = m.group(1) if m else txt[:160]
        return {"role": role, "start": 0, "end": len(snippet), "text": snippet}
    snippet = (user_text or "")[start:end]
    return {"role": role, "start": start, "end": end, "text": snippet}

def propose_facts(user_id: str, assistant_name: str, text: str, role: str) -> List[Dict[str, Any]]:
    """
    Open-ended factual proposals, but MUST include provenance:
      subject, key, value, confidence, span {start,end}, and role ("user"/"assistant")
    """
    if not text or len(text) < 2: return []
    llm = get_llm(getattr(config, "MODEL_BACKEND", "ollama"))
    sys = (
      "Extract small factual claims explicitly present in the message. "
      "Resolve pronouns: 'you'/'{an}' -> assistant; 'I'/'me' -> user '{uid}'. "
      "Return JSON: {{\"facts\":[{{\"subject\":\"assistant|{uid}\",\"key\":\"…\",\"value\":\"…\","
      "\"confidence\":0.0-1.0,\"span\":{{\"start\":int,\"end\":int}}}}]}}. "
      "Only include facts supported by exact words in the message; choose spans that cover the evidence. "
      "Keep keys compact (e.g., user.name, likes:quiet, assistant.gender, assistant.hair, tone:plain). "
      "Exclude meta-identity like 'AI', 'assistant', 'here to help'."
    ).format(uid=user_id, an=assistant_name)
    prompt = f"assistant_name={assistant_name}\nuser_id={user_id}\nrole={role}\nmessage:\n{text}"
    out = llm.generate_complete(sys, prompt, options={"temperature": 0.2, "num_predict": 260}, max_segments=2).strip()
    obj = _json_from(out) or {}
    facts = obj.get("facts", []) if isinstance(obj, dict) else []
    rows: List[Dict[str, Any]] = []
    for f in facts:
        subj = str(f.get("subject","")).strip().lower()
        key  = str(f.get("key","")).strip().lower().replace(" ", "_")
        val  = _clean(str(f.get("value","")))
        conf = f.get("confidence", 0.6)
        try: conf = float(conf)
        except Exception: conf = 0.6
        span = f.get("span", {}) or {}
        s = span.get("start", -1); e = span.get("end", -1)
        try: s = int(s)
        except Exception: s = -1
        try: e = int(e)
        except Exception: e = -1

        rows.append({
            "role": role,
            "subject": "assistant" if subj in {"assistant", assistant_name.lower(), "you"} else user_id,
            "key": key, "value": val,
            "confidence": max(0.0, min(1.0, conf)),
            "span": {"start": s, "end": e}
        })
    return rows

# ============= Verifiers =============
def verify_against_text(text: str, fact: Dict[str, Any]) -> bool:
    """
    Verifies by substring entailment: if span is valid, the substring must
    literally contain or equal the value (case-insensitive).
    """
    text = text or ""
    span = fact.get("span") or {}
    s = span.get("start", -1); e = span.get("end", -1)
    try:
        s = int(s); e = int(e)
    except Exception:
        s, e = -1, -1

    if _within_bounds(s, e, len(text)):
        sub = text[s:e].lower().strip()
        val = str(fact.get("value","")).lower().strip()
        if val and val in sub:
            return True
        if sub and sub == val:
            return True

    # fallback lexical check: value present anywhere in text
    val = str(fact.get("value","")).strip()
    if val and val.lower() in text.lower():
        return True

    return False

def filter_and_verify_facts(user_id: str, assistant_name: str, text: str, role: str,
                            facts: List[Dict[str,Any]], conf_floor: float = 0.55) -> List[Dict[str,Any]]:
    KEEP: List[Dict[str, Any]] = []
    for f in facts:
        if f.get("confidence", 0.0) < conf_floor:
            continue
        k = f.get("key",""); v = f.get("value","")
        if not k or not v:
            continue
        # filter assistant boilerplate if it sneaks into values
        if re.search(r"\b(as an ai|assistant|language model|here to help|designed)\b", str(v), re.I):
            continue
        if verify_against_text(text, f):
            KEEP.append(f)
    return KEEP

# ============= Public API =============
def extract_and_verify(user_id: str, assistant_name: str, text: str, role: str) -> Tuple[Dict[str,Any], List[Dict[str,Any]]]:
    """
    Returns (gist_proposal, verified_facts)
    gist_proposal: {"role":..., "text":..., "start":int, "end":int}
    facts: list of {subject,key,value,confidence,span:{start,end},role}
    """
    gist = propose_gist(text, role)
    facts = propose_facts(user_id, assistant_name, text, role)
    vfacts = filter_and_verify_facts(user_id, assistant_name, text, role, facts)
    return gist, vfacts
