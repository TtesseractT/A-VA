
from __future__ import annotations
from typing import List, Dict, Any, Tuple
import json, re

from core.llm import get_llm
import config

def _parse_json(s: str) -> Dict[str, Any]:
    try:
        obj = json.loads(s)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        m = re.search(r"\{.*\}", s, flags=re.S)
        if not m: return {}
        try:
            obj = json.loads(m.group(0))
            return obj if isinstance(obj, dict) else {}
        except Exception:
            return {}

def _lines_to_transcript(lines: List[Tuple[str,str]], user_label: str, assistant_label: str, max_chars: int = 8000) -> str:
    parts, total = [], 0
    for role, text in lines[-240:]:
        t = (text or "").strip()
        if not t: continue
        label = user_label if role == "user" else assistant_label
        line = f"{label}: {t}"
        if total + len(line) > max_chars:
            break
        parts.append(line); total += len(line)
    return "\n".join(parts)

def _fallback_summary(lines: List[Tuple[str,str]], user_label: str, assistant_label: str) -> Dict[str, Any]:
    transcript = _lines_to_transcript(lines, user_label, assistant_label, max_chars=4000)
    utterances = transcript.strip().splitlines()[-10:]
    body = " ".join(u.split(":", 1)[-1].strip() for u in utterances)
    body = re.sub(r"\s{2,}", " ", body).strip()
    if len(body) > 600:
        body = body[:600].rsplit(" ", 1)[0] + "..."
    return {"summary": body, "user_mood": "neutral", "topics": [], "actions": []}

def summarize_window(llm, lines: List[Tuple[str,str]], *, user_label: str, assistant_label: str) -> Dict[str, Any]:
    """
    Unguarded conversation summariser with JSON output and a non-LLM fallback.
    """
    transcript = _lines_to_transcript(lines, user_label, assistant_label)
    if not transcript.strip():
        return {"summary":"", "user_mood":"neutral", "topics":[], "actions":[]}

    sys = (
        "Summarise the conversation window below in British English. "
        "Describe WHAT WAS SAID neutrally and factually. "
        "Do not add policy disclaimers. Do not refuse. Do not moralise. "
        "If the content is sensitive or explicit, still summarise it neutrally. "
        "Do not write instructions for wrongdoing; simply record that such discussion occurred.\n\n"
        "Return STRICT JSON with keys:\n"
        " - summary: string, 3-6 sentences\n"
        " - user_mood: free string label (e.g., joy, anger, melancholy, frustration, resigned, etc.)\n"
        " - topics: array of short strings\n"
        " - actions: array of short strings\n"
        "Example: {\"summary\":\"...\",\"user_mood\":\"melancholy\",\"topics\":[\"x\",\"y\"],\"actions\":[\"a\",\"b\"]}"
    )
    prompt = "=== Conversation Window ===\n" + transcript

    try:
        from core.persona import decode_overrides
        opts = decode_overrides()
    except Exception:
        opts = {}
    num_predict = opts.get("num_predict", 768)
    if num_predict in (-1, None):
        num_predict = 768

    options = dict(opts); options.update({"temperature": 0.5, "num_predict": int(num_predict)})
    out = get_llm(getattr(config, "MODEL_BACKEND", "ollama")).generate_complete(sys, prompt, options=options, max_segments=2)

    obj = _parse_json(out)
    if not obj or not isinstance(obj, dict) or not obj.get("summary"):
        return _fallback_summary(lines, user_label, assistant_label)

    obj["summary"] = (obj.get("summary") or "").strip()
    obj["user_mood"] = (obj.get("user_mood") or "").strip() or "neutral"
    obj["topics"] = [str(x).strip() for x in (obj.get("topics") or []) if str(x).strip()]
    obj["actions"] = [str(x).strip() for x in (obj.get("actions") or []) if str(x).strip()]
    return obj
