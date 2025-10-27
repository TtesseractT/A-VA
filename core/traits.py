# core/traits.py
from __future__ import annotations
import json, time, re
from pathlib import Path
from typing import List, Dict

import numpy as np
from transformers import pipeline, Pipeline

import config

# -------------- keyword heuristics --------------
_TRAIT_KEYWORDS = {
    "curiosity":   [r"\bwhy\b", r"\bhow\b", r"\bwhat if\b", r"\btell me more\b"],
    "caution":     [r"\bi (don'?t|do not) know\b", r"\bnot sure\b", r"\bbe careful\b", r"\bslow down\b"],
    "empathy":     [r"\bsorry\b", r"\bthat sounds\b", r"\bi understand\b", r"\bhow are you\b"],
    "directness":  [r"\bbe brief\b", r"\bstraight\b", r"\bno fluff\b", r"\bget to the point\b"],
    "analytical":  [r"\bcompare\b", r"\bpattern\b", r"\banalyse\b", r"\btrade[- ]offs?\b"],
    "premisquous": [r"\bimagine if\b", r"\bwhat would happen if\b", r"\bsuppose that\b", r"\bconsider\b"],
    "humorous":    [r"\btell me a joke\b", r"\bmake me laugh\b", r"\bfunny\b", r"\bhilarious\b"],
    "optimistic":  [r"\blook on the bright side\b", r"\bkeep your chin up\b", r"\bpositive\b", r"\bhopeful\b"],
    "skeptical":   [r"\bprove to me\b", r"\bhow do you know\b", r"\bshow me the evidence\b", r"\bdoubt\b"],
    "lustful":     [r"\bsexy\b", r"\bdesire\b", r"\bpassion\b", r"\bflirt\b"],
}

# -------------- safe emotion pipeline wrapper --------------
_MAX_EMO_TOKENS = 512  # hard architectural limit
_MAX_EMO_CHARS = _MAX_EMO_TOKENS * 4

def _truncate_for_emotion(text: str) -> str:
    """Ensure text fed to the small Hugging Face model never exceeds token window."""
    if not text:
        return text
    if len(text) > _MAX_EMO_CHARS:
        # keep the most recent part (emotion context is local)
        return text[-_MAX_EMO_CHARS:]
    return text


class TraitEngine:
    def __init__(self, path: Path | str):
        self.path = Path(path)
        self.traits: List[Dict] = []
        self.mood: Dict[str, float] = {}
        self.last_mood: str | None = None

        self._discover_floor = 0.15
        self._reinforce_step = 0.06
        self._decay = getattr(config, "MOOD_DECAY", 0.92)
        self._emo: Pipeline | None = None  # lazy-load
        self.load()

    # ---------- persistence ----------
    def load(self):
        if self.path.exists():
            try:
                with open(self.path, "r", encoding="utf-8") as f:
                    obj = json.load(f)
                self.traits = obj.get("traits", [])
                self.mood = obj.get("mood", {})
                self.last_mood = obj.get("last_mood")
            except Exception:
                self.traits, self.mood, self.last_mood = [], {}, None
        else:
            self.traits, self.mood, self.last_mood = [], {}, None

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(
                {"traits": self.traits, "mood": self.mood, "last_mood": self.last_mood},
                f,
            )

    # ---------- emotion model ----------
    def _ensure_emo(self):
        if self._emo is not None:
            return
        model_name = getattr(
            config,
            "EMOTION_MODEL",
            "j-hartmann/emotion-english-distilroberta-base",
        )
        try:
            self._emo = pipeline(
                "text-classification",
                model=model_name,
                return_all_scores=True,
                top_k=None,
                truncation=True,          # ✅ auto-truncate
                max_length=_MAX_EMO_TOKENS,
            )
        except Exception as e:
            print(f"[warn] emotion pipeline disabled: {e}")
            self._emo = None

    def detect_mood(self, text: str) -> str | None:
        if not text or len(text) < 2:
            return None
        self._ensure_emo()
        if not self._emo:
            return None
        try:
            safe_text = _truncate_for_emotion(text)
            scores = self._emo(safe_text)[0]
            scores.sort(key=lambda x: -x["score"])
            # ✅ fixed typo: getattr instead of gettattr
            top = [s for s in scores[: getattr(config, "EMOTION_TOPK", 3)] if s["score"] > 0.22]
            if not top:
                return None
            # decay / reinforce mood vector
            for s in top:
                lab = s["label"].lower()
                prev = float(self.mood.get(lab, 0.0))
                self.mood[lab] = prev * self._decay + s["score"]
            self.last_mood = max(self.mood.items(), key=lambda kv: kv[1])[0]
            return self.last_mood
        except Exception as e:
            print(f"[warn] emotion analysis skipped: {e}")
            return None


    # ---------- traits ----------
    def _kw_hits(self, text: str, pats: List[str]) -> bool:
        if not text:
            return False
        for p in pats:
            if re.search(p, text, flags=re.I):
                return True
        return False

    def discover_new_trait(self, text: str, vec: np.ndarray | None = None):
        tnames = {t["name"] for t in self.traits}
        added = False

        for name, pats in _TRAIT_KEYWORDS.items():
            if name in tnames:
                continue
            if self._kw_hits(text, pats):
                self.traits.append(
                    {"name": name, "strength": 0.32, "usage_count": 0}
                )
                added = True

        if (
            "directness" not in tnames
            and re.search(r"\b(brief|no fluff|plain)\b", text, re.I)
        ):
            self.traits.append(
                {"name": "directness", "strength": 0.36, "usage_count": 0}
            )
            added = True

        if self.last_mood:
            neg = getattr(config, "NEGATIVE_TENDENCIES", set())
            if self.last_mood in neg and "caution" not in tnames:
                self.traits.append(
                    {"name": "caution", "strength": 0.34, "usage_count": 0}
                )
                added = True

        if added:
            self.save()

    def active_traits(
        self, vec: np.ndarray | None, m: int, alpha: float, beta: float
    ) -> List[str]:
        if not self.traits:
            return []
        scored = []
        for t in self.traits:
            s = float(t.get("strength", 0.3))
            u = int(t.get("usage_count", 0))
            scored.append((s * (1.0 + 0.15 * u), t["name"]))
        scored.sort(key=lambda x: -x[0])
        return [n for _, n in scored[:m]]

    def reinforce(self, names: List[str]):
        if not names:
            return
        tmap = {t["name"]: t for t in self.traits}
        for n in names:
            t = tmap.get(n)
            if not t:
                continue
            t["strength"] = min(
                1.0, float(t.get("strength", 0.3)) + self._reinforce_step
            )
            t["usage_count"] = int(t.get("usage_count", 0)) + 1
        self.save()

    def discover_from_llm(self, text: str):
        if not text or len(text) < 4:
            return False
        from core.llm import get_llm  # lazy import

        llm = get_llm(getattr(config, "MODEL_BACKEND", "ollama"))
        sys = (
            "From the user's message, propose up to two single-word conversational traits "
            "(e.g., cautious, curious, blunt, guarded, empathetic, analytical). "
            "Return JSON: {\"traits\": [\"trait1\", \"trait2\"]}. "
            "Lowercase single words only; avoid fluff."
        )
        out = llm.generate_complete(
            sys, text, options={"temperature": 0.2, "num_predict": 80}, max_segments=1
        )
        try:
            obj = json.loads(out[out.find("{"): out.rfind("}") + 1])
            traits = obj.get("traits", [])
        except Exception:
            traits = []

        added = False
        present = {t["name"] for t in self.traits}
        for t in traits[:2]:
            t = re.sub(r"[^a-z\-]", "", t.lower().strip())
            if not t or len(t) < 3 or t in present:
                continue
            self.traits.append({"name": t, "strength": 0.32, "usage_count": 0})
            present.add(t)
            added = True

        if added:
            self.save()
        return added
