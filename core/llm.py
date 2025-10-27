# core/llm.py
from __future__ import annotations
import os, json, re
from typing import Optional, Dict, Any, Tuple
import requests
import config

_SENT_DONE_RX = re.compile(r'[.!?]["”\']?\s*$')

def _clamp(v, lo, hi): 
    return max(lo, min(hi, v))

class LLMBase:
    def generate_with_system(self, system: str, user: str,
                             options_override: Optional[Dict[str, Any]] = None) -> Tuple[str, Dict]:
        text = self.generate_complete(system, user, options_override or {}, max_segments=1)
        return text, {}

    def generate_complete(self, system: str, user: str,
                          options: Optional[Dict[str, Any]] = None,
                          max_segments: int = 1) -> str:
        raise NotImplementedError

class OllamaLLM(LLMBase):
    def __init__(self):
        self.model = getattr(config, "OLLAMA_MODEL", "llama3.1:8b-instruct-q8_0")
        self.host  = getattr(config, "OLLAMA_HOST", "http://localhost:11434")

    def _opts(self, opts: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        base = dict(getattr(config, "OLLAMA_OPTIONS", {}))
        opts = opts or {}
        for k, v in opts.items():
            if k in ("temperature", "top_p", "repeat_penalty"):
                lo, hi = config.DECODE_BOUNDS.get(k, (0, 2))
                base[k] = _clamp(float(v), lo, hi)
            elif k == "num_predict":
                lo, hi = config.DECODE_BOUNDS.get("num_predict", (32, 1024))
                base[k] = int(_clamp(int(v), lo, hi))
            else:
                base[k] = v
        return base

    def _call(self, system: str, user: str, opts: Dict[str, Any]) -> str:
        url = f"{self.host}/api/generate"
        prompt = f"<|system|>\n{system}\n<|user|>\n{user}\n<|assistant|>\n"
        payload = {"model": self.model, "prompt": prompt, "stream": False, "options": opts}
        r = requests.post(url, data=json.dumps(payload), timeout=120)
        r.raise_for_status()
        data = r.json()
        return data.get("response", "") or ""

    def _should_continue(self, chunk: str, assembled: str, budget_np: int) -> bool:
        if not chunk:
            return False
        t = chunk.strip()
        # classic cutoff signals
        if len(t) >= 3 and t.endswith("..."):
            return True
        # if last sentence not closed and we already produced a decent amount
        if getattr(config, "CONTINUE_IF_UNCLOSED", True):
            long_enough = len(assembled) >= getattr(config, "CONTINUE_MIN_CHARS", 220)
            if long_enough and not _SENT_DONE_RX.search(t):
                return True
        return False

    def generate_complete(self, system: str, user: str,
                          options: Optional[Dict[str, Any]] = None,
                          max_segments: int = 1) -> str:
        """
        Stitches multiple /api/generate calls to finish the thought:
        - First call uses (system, user)
        - Continuations add a tiny 'continue' nudge
        - Stops when sentence closes or segment cap is hit
        """
        opts = self._opts(options)
        assembled = ""
        seg_cap = max(1, int(max_segments))
        # remember initial predict budget
        budget = int(opts.get("num_predict", getattr(config, "OLLAMA_OPTIONS", {}).get("num_predict", 256)))

        for i in range(seg_cap):
            sys_txt = system if i == 0 else (system + "\nContinue the same reply. Do not restart, recap, or change style.")
            usr_txt = user if i == 0 else ""
            chunk = self._call(sys_txt, usr_txt, opts).strip()
            if not chunk:
                break
            assembled = (assembled + " " + chunk).strip()

            # after first hop, cool temperature slightly and raise budget for safety
            if i == 0 and "temperature" in opts:
                opts["temperature"] = max(0.1, float(opts["temperature"]) - 0.05)

            # raise the budget gradually so the next hop can complete
            np_next = min(config.DECODE_BOUNDS["num_predict"][1], budget + 256)
            opts["num_predict"] = np_next

            if not self._should_continue(chunk, assembled, budget):
                break
            budget = np_next

        return assembled

def get_llm(backend: str = "ollama") -> LLMBase:
    if backend.lower() == "ollama":
        return OllamaLLM()
    # future: add other backends here
    return OllamaLLM()
