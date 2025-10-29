from pathlib import Path

# ---------- Paths ----------
DATA_DIR = Path(__file__).parent / "data"
STORE_DIR = Path(__file__).parent
DB_PATH = STORE_DIR / "memory.duckdb"
FAISS_PATH = STORE_DIR / "vectors.faiss"
VECTORS_NPY = STORE_DIR / "vectors.npy"
GRAPH_PATH = STORE_DIR / "graph.json"
TRAITS_PATH = DATA_DIR / "traits.json"

# ---------- Emotion / Traits ----------
EMOTION_REGISTRY_PATH = Path(DATA_DIR) / 'emotions.json'
EMOTION_LEXICON_PATH  = Path(DATA_DIR) / 'emotion_lexicon.json'
AUTO_ADD_EMOTION_TRAITS = True  # traits managed by wizard/runtime only

# Embeddings
# May want to change this to something specific to the language
EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"  # 1024-d
EMBEDDING_DIM = 1024
EMBEDDING_NORMALIZE = True
# Resume / recap behaviour
RESUME_HINT_TURNS = 30     # add hints for first N assistant turns in a new process
RESUME_HINT_GISTS = 30     # how many recent gists to surface as hints

# Retrieval/scoring
CANDIDATE_K = 120
FINAL_K = 16
SPREAD_DEPTH = 2
SPREAD_DECAY = 0.85
ALPHA_USAGE = 0.25
BETA_TIME = 8e-6
MMR_LAMBDA = 0.33
USE_CROSS_ENCODER = True
CE_MODEL = "BAAI/bge-reranker-large"
CE_TOPN = 40
PERSONA_PATH = None
CONTEXT_EPISODES = FINAL_K
CONTEXT_FACTS = 12
TRAIT_BULLETS = 4
SIM_THRESHOLD_LOW = 0.35
EDGE_ETA = 0.08
TRAIT_GAMMA = 0.06

# Dense memory pack
DENSE_PACK = True
DENSE_TOPK = 10
GIST_WORDS = 14
PACK_BULLETS = 5
PACK_MAXTOK = 120
REALTIME_EVENTS = True
EVENTS_ADDR = "tcp://127.0.0.1:5557"

# Memory gating thresholds
MEMORY_GATE_MIN_SIM = 0.30
MEMORY_GATE_JACCARD = 0.07
MEMORY_GATE_MIN_HITS = 1
MEMORY_GATE_GISTS = 8

# LLM backend
MODEL_BACKEND = "ollama"
OLLAMA_MODEL = "llama3.1:8b-instruct-q8_0"
OLLAMA_HOST = "http://localhost:11434"

# Ollama decoding options
OLLAMA_OPTIONS = {
    "temperature": 0.65,
    "top_p": 0.9,
    "repeat_penalty": 1.2,
}

# Global decode bounds
DECODE_BOUNDS = {
    "num_predict": (-1, -1),   # (-1 = unlimited)
    "max_tokens":  (None, None),
    "temperature": (0.0, 1.3),
    "top_p":       (0.0, 1.0),
    "repeat_penalty": (1.0, 1.6),
}

# Reply length presets
LENGTH_PRESETS = {
    "tiny": 90,
    "short": 220,
    "medium": 420,
    "long": 800,
}

# Auto-continue controls
CONTINUE_MIN_CHARS = 1024
CONTINUE_MAX_SEGMENTS = 32

# Identity & tone
ASSISTANT_NAME = "<assistant_name>"
SYSTEM_PRIMER = (
    "You are {assistant_name}. You are not the user {user_name}. "
)
REFUSAL_HINT = (
    "You may say 'no' or 'I don't know yet' for unknowns or unsuitable requests. "
    "Be brief and neutral; offer one clarifying question or a safe alternative."
)

# Memory snapshot
ENABLE_MEMORY_SNAPSHOT = True
MEMORY_MAX_CARDS = 12

# Style/voice guards
STYLE_REWRITE_SYSTEM = (
    "Rewrite the reply as a first-person ('I') utterance in plain, everyday British English. "
    "Remove any assistant disclaimers (e.g., 'as an AI', 'my role', 'I'm here to help', 'assistant'). "
    "Output only the rewritten reply."
)

# Emotion/traits
EMOTION_MODEL = "j-hartmann/emotion-english-distilroberta-base"
EMOTION_TOPK = 3

# Optional: identity filters used by memcards
IDENTITY_BLOCKLIST = [
    r"\bassistant\b", r"\blanguage model\b", r"\bas an ai\b", r"\bmy role\b", r"\bi'?m here to help\b",
]
