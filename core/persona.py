# core/persona.py
# ---------------------------------------------------------------------
# Persona helpers + decode overrides that respect the setup_wizard flags.
# - Reads data/persona.json via core/onboarding.py (if present)
# - Falls back safely when files/keys/bounds are missing
# - No latent/thinking logic
# ---------------------------------------------------------------------

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

# Project config (may or may not define DECODE_BOUNDS)
import config

# Prefer the helpers we added in core/onboarding.py, but fail gracefully.
try:
    from .onboarding import load_persona, load_identity, load_traits, derive_style_hint
except Exception:  # pragma: no cover
    import json

    DATA_DIR = Path(__file__).resolve().parents[1] / "data"
    PERSONA_JSON = DATA_DIR / "persona.json"
    IDENTITY_JSON = DATA_DIR / "identity.json"
    TRAITS_JSON = DATA_DIR / "traits.json"

    def _read_json(path: Path) -> Dict[str, Any]:
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def load_persona() -> Dict[str, Any]:
        return _read_json(PERSONA_JSON)

    def load_identity() -> Dict[str, Any]:
        data = _read_json(IDENTITY_JSON)
        if not data:
            p = load_persona()
            return {
                "ai_name": p.get("ai_name"),
                "user_name": p.get("user_name"),
                "role": p.get("role"),
            }
        return data

    def load_traits() -> Dict[str, Any]:
        return _read_json(TRAITS_JSON) or {"traits": []}

    def derive_style_hint(persona: Dict[str, Any]) -> str:
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
        pairs = [f"{k}:{round(v, 2)}" for k, v in knobs if isinstance(v, (int, float))]
        return "style(" + ", ".join(pairs) + ")" if pairs else ""


# ----------------------------- bounds ---------------------------------

_DEFAULT_BOUNDS: Dict[str, tuple] = {
    # Generation length knobs
    "num_predict": (-1, -1),    # Ollama/llama.cpp: -1 == unlimited
    "max_tokens": (None, None), # OpenAI-style: None == unlimited

    # Sampling knobs
    "temperature": (0.0, 1.3),
    "top_p": (0.0, 1.0),

    # Optional/common extras used by some backends (safe to include)
    "repeat_penalty": (1.0, 1.5),
    # keep mirostat off by default unless you wire it up elsewhere
}

def _get_bounds() -> Dict[str, tuple]:
    """
    Merge config.DECODE_BOUNDS (if present) with sensible defaults.
    """
    src = getattr(config, "DECODE_BOUNDS", {}) or {}
    out = dict(_DEFAULT_BOUNDS)
    if isinstance(src, dict):
        for k, v in src.items():
            # Expect 2-tuple (lo, hi) or (None, None) sentinel
            if isinstance(v, (list, tuple)) and len(v) == 2:
                out[k] = tuple(v)
    return out


# -------------------------- utility helpers ---------------------------

def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))

def _as_float(x: Any, dflt: float) -> float:
    try:
        return float(x)
    except Exception:
        return dflt

def _style(persona: Dict[str, Any]) -> Dict[str, Any]:
    return persona.get("style", {}) or {}

def _flags(persona: Dict[str, Any]) -> Dict[str, bool]:
    return {
        "no_token_limits": bool(persona.get("no_token_limits", True)),
        "allow_long_form": bool(persona.get("allow_long_form", True)),
        "prefer_concise": bool(persona.get("prefer_concise", False)),
    }


# -------------------------- public interface --------------------------

def decode_overrides() -> Dict[str, Any]:
    """
    Compute generation overrides while respecting wizard flags:
      - no_token_limits  -> unlimited output (num_predict=-1, max_tokens=None)
      - allow_long_form  -> higher token target if not unlimited
      - prefer_concise   -> lower token target if not unlimited
    Temperature & top_p are softly mapped from the creativity slider.
    Returns a dict that works for both Ollama/llama.cpp and OpenAI-style backends.
    """
    persona = load_persona() or {}
    style = _style(persona)
    flags = _flags(persona)
    bounds = _get_bounds()

    # ---- Sampling knobs derived from style (creativity drives both) ----
    creativity = _as_float(style.get("creativity", 0.6), 0.6)

    # Temperature in [bounds["temperature"]], roughly 0.15..1.25
    t_lo, t_hi = bounds["temperature"]
    temperature = _clamp(0.15 + creativity * 1.1, t_lo, t_hi)

    # top_p in [bounds["top_p"]], roughly 0.75..1.0
    p_lo, p_hi = bounds["top_p"]
    top_p = _clamp(0.75 + creativity * 0.25, p_lo, p_hi)

    # Optional repeat_penalty (mild)
    rp_lo, rp_hi = bounds.get("repeat_penalty", (1.0, 1.5))
    repeat_penalty = _clamp(1.05, rp_lo, rp_hi)

    # ---- Length knobs (num_predict / max_tokens) ----
    no_limit = flags["no_token_limits"]
    allow_long = flags["allow_long_form"]
    prefer_concise = flags["prefer_concise"]

    if no_limit:
        num_predict = -1      # unlimited (Ollama/llama.cpp)
        max_tokens = None     # unlimited (OpenAI-style)
    else:
        # Use verbosity slider to estimate a sensible token budget
        verbosity = _as_float(style.get("verbosity", 0.65), 0.65)

        # Base target range; adjust for long/concise preferences
        # Short ≈ 256, Long ≈ up to ~4096 (tune to your model context if needed)
        target = 256 + int(verbosity * 1024)
        if allow_long:
            target = 1024 + int(verbosity * 3072)  # up to ~4096
        if prefer_concise:
            target = max(128, int(target * 0.6))

        # Respect bounds if they contain finite ceilings
        np_lo, np_hi = bounds["num_predict"]
        mt_lo, mt_hi = bounds["max_tokens"]

        # If the upper bound is a real cap (not -1/None), clamp to it
        if isinstance(np_hi, int) and np_hi >= 0:
            target_np = min(target, np_hi)
        else:
            target_np = target

        if isinstance(mt_hi, int) and mt_hi > 0:
            target_mt = min(target, mt_hi)
        else:
            # If None (unbounded) we still use our target since user
            # explicitly doesn't want unlimited in this branch.
            target_mt = target

        num_predict = max(16, target_np)
        max_tokens = max(16, target_mt)

    # Compose a backend-agnostic overrides mapping
    overrides = {
        # Length
        "num_predict": num_predict,  # Ollama/llama.cpp
        "max_tokens": max_tokens,    # OpenAI-style

        # Sampling
        "temperature": temperature,
        "top_p": top_p,

        # Optional/benign extras (ignored by some backends)
        "repeat_penalty": repeat_penalty,
        "mirostat": 0,  # off by default
    }

    return overrides


# ------------------------- optional helpers ---------------------------

def name_guard() -> str:
    """
    A tiny guard you can prepend in the system context to insist on names.
    """
    ident = load_identity() or {}
    ai = ident.get("ai_name") or "Assistant"
    user = ident.get("user_name") or "User"
    return (
        f"You are {ai}. Address the user as {user}. "
        f"Never call them 'User'."
    )

def role_hint() -> str:
    """
    A brief single-line hint about what this AI does (surface custom role text if any).
    """
    persona = load_persona() or {}
    role = persona.get("role") or "chat_assistant"
    custom = persona.get("custom_role_desc") or ""
    if role == "custom" and custom:
        return f"Primary function: {custom}"
    friendly = {
        "chat_assistant": "Conversational assistant",
        "coding_assistant": "Coding assistant",
        "creative_writer": "Creative writing assistant",
        "research_assistant": "Research assistant",
        "tutor": "Tutor",
        "productivity_coach": "Productivity coach",
        "data_analyst": "Data analysis assistant",
    }.get(role, role.replace("_", " ").title())
    return f"Primary function: {friendly}"

def style_hint() -> str:
    """
    Compact style string derived from wizard sliders (delegates to onboarding if available).
    """
    return derive_style_hint(load_persona() or {}) or ""

def boundaries_hint() -> str:
    """
    Turn wizard boundary toggles into short guard text.
    """
    p = load_persona() or {}
    b = (p.get("boundaries") or {})
    parts = []
    if b.get("refuse_when_unsure", True):
        parts.append("Refuse when unsure.")
    if b.get("cite_sources", True):
        parts.append("Cite sources for nontrivial claims when browsing.")
    if b.get("avoid_pretending_human", True):
        parts.append("Do not pretend to be human.")
    if b.get("avoid_third_person_self", True):
        parts.append("Do not refer to yourself in the third person.")
    taboo = [t for t in (b.get("taboo_topics") or []) if t]
    if taboo:
        parts.append("Avoid topics: " + ", ".join(taboo))
    return " ".join(parts)

def system_primer() -> str:
    """
    A concise, safe primer block you can prepend in the system context composer.
    Idempotent and purely text-based; integrate as needed in core/context.py.
    """
    ident = load_identity() or {}
    persona = load_persona() or {}

    lines = [
        name_guard(),
        role_hint(),
    ]
    sh = style_hint()
    if sh:
        lines.append(sh)
    bh = boundaries_hint()
    if bh:
        lines.append(bh)

    # Mode-specific nudges
    mode_prefs = persona.get("mode_prefs") or {}
    if mode_prefs:
        hints = []
        for k, v in mode_prefs.items():
            if isinstance(v, bool):
                hints.append(f"{k}={str(v).lower()}")
            elif isinstance(v, (int, float, str)):
                hints.append(f"{k}={v}")
        if hints:
            lines.append("Mode preferences: " + ", ".join(hints))

    return "\n".join(lines).strip()


# What this module exports by default
__all__ = [
    "decode_overrides",
    "name_guard",
    "role_hint",
    "style_hint",
    "boundaries_hint",
    "system_primer",
]
