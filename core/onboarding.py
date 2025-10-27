
"""
Load persona & identity for use in system context composition without altering
the existing retrieval/memory pipeline. Import and use where needed.
"""
import json
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
PERSONA_JSON = DATA_DIR / "persona.json"
IDENTITY_JSON = DATA_DIR / "identity.json"
TRAITS_JSON  = DATA_DIR / "traits.json"

def load_persona():
    if PERSONA_JSON.exists():
        return json.loads(PERSONA_JSON.read_text(encoding="utf-8"))
    return {}

def load_identity():
    if IDENTITY_JSON.exists():
        return json.loads(IDENTITY_JSON.read_text(encoding="utf-8"))
    p = load_persona()
    return {"ai_name": p.get("ai_name"), "user_name": p.get("user_name"), "role": p.get("role")}

def load_traits():
    if TRAITS_JSON.exists():
        return json.loads(TRAITS_JSON.read_text(encoding="utf-8"))
    return {"traits": []}

def derive_style_hint(persona: dict) -> str:
    s = persona.get("style", {})
    knobs = [
        ("formality", s.get("formality")),
        ("warmth", s.get("warmth")),
        ("humour", s.get("humour")),
        ("directness", s.get("directness")),
        ("creativity", s.get("creativity")),
        ("skepticism", s.get("skepticism")),
        ("analytical", s.get("analytical")),
        ("verbosity", s.get("verbosity")),
    ]
    pairs = [f"{k}:{round(v,2)}" for k,v in knobs if isinstance(v,(int,float))]
    return "style(" + ", ".join(pairs) + ")" if pairs else ""
