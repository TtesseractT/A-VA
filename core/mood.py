
from __future__ import annotations
from typing import Tuple, Dict, Set
from pathlib import Path
import json

import config

# Optional classifier; if missing, we rely on user-provided lexicon or neutral fallback
try:
    from transformers import pipeline
    _EMO_PIPE = pipeline(
        "text-classification",
        model="j-hartmann/emotion-english-distilroberta-base",
        return_all_scores=True,
    )
except Exception:
    _EMO_PIPE = None

# Paths from config
_REG_PATH = Path(getattr(config, "EMOTION_REGISTRY_PATH", Path(config.DATA_DIR) / "emotions.json"))
_LEX_PATH = Path(getattr(config, "EMOTION_LEXICON_PATH", Path(config.DATA_DIR) / "emotion_lexicon.json"))

def _load_registry() -> Set[str]:
    try:
        if _REG_PATH.exists():
            return set(json.loads(_REG_PATH.read_text(encoding="utf-8")))
    except Exception:
        pass
    return set()

def register_emotion(label: str) -> None:
    lab = (label or "").strip().lower()
    if not lab:
        lab = "neutral"
    seen = _load_registry()
    if lab in seen:
        return
    seen.add(lab)
    try:
        _REG_PATH.parent.mkdir(parents=True, exist_ok=True)
        _REG_PATH.write_text(json.dumps(sorted(seen), ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass  # non-fatal

def load_lexicon() -> Dict[str, Set[str]]:
    """Load a user-provided lexicon file mapping labels -> list of tokens. No hard-coded defaults."""
    try:
        if _LEX_PATH.exists():
            raw = json.loads(_LEX_PATH.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                return {str(k).lower(): set(str(x).lower() for x in v) for k, v in raw.items() if isinstance(v, list)}
    except Exception:
        pass
    return {}

def save_lexicon(lex: Dict[str, Set[str]]) -> None:
    """Optional helper to persist a learned lexicon elsewhere in the codebase."""
    try:
        _LEX_PATH.parent.mkdir(parents=True, exist_ok=True)
        ser = {k: sorted(list(v)) for k, v in lex.items()}
        _LEX_PATH.write_text(json.dumps(ser, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass

def _lexicon_score(text: str, lex: Dict[str, Set[str]]) -> Tuple[str, float]:
    """Simple bag-of-words scorer using user-provided lexicon. Returns ('', 0.0) if no signals."""
    if not text or not lex:
        return ("", 0.0)
    t = (text or "").lower()
    best_label, best_score = "", 0.0
    for label, words in lex.items():
        # sum of binary matches; advanced weighting can be added by user in lexicon if desired
        score = float(sum(1 for w in words if w and w in t))
        if score > best_score:
            best_label, best_score = label, score
    if best_score <= 0.0:
        return ("", 0.0)
    # naive confidence: squash into (0.55..0.95)
    conf = max(0.55, min(0.95, 0.55 + 0.1 * best_score))
    return (best_label, conf)

def detect_mood_label(text: str) -> Tuple[str, float]:
    """
    Returns (label, confidence). Priority:
      1) Classifier (if installed) -> free-form label
      2) User-provided lexicon (config.EMOTION_LEXICON_PATH)
      3) Neutral fallback
    No hard-coded labels or word lists here.
    """
    # 1) Classifier
    if _EMO_PIPE is not None:
        try:
            res = _EMO_PIPE((text or "")[:1500])
            if res and isinstance(res, list) and res and isinstance(res[0], list) and res[0]:
                best = max(res[0], key=lambda x: x.get("score", 0.0))
                label = (best.get("label") or "neutral").strip().lower()
                register_emotion(label)
                return (label, float(best.get("score", 0.0)))
        except Exception:
            pass

    # 2) User lexicon
    lex = load_lexicon()
    lab, conf = _lexicon_score(text or "", lex)
    if lab:
        register_emotion(lab)
        return (lab, conf)

    # 3) Neutral
    register_emotion("neutral")
    return ("neutral", 0.5)
