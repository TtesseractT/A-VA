# Trait Engine (POC)

A local-first conversational memory system that lets an AI assistant develop and use long-term memories, traits, and a configurable persona. The system is wizard-driven at startup (you choose the AI's name, your name, role, boundaries, style, and initial traits) and then learns organically from normal use-adding facts and evolving traits at runtime.

> Highlights
- Wizard onboarding (names, role, style, boundaries, traits; unlimited tokens toggle)
- Long-term memory via DuckDB (episodes + facts)
- Fast recall with embeddings + graph spreading activation
- Trait engine that updates from real usage
- Local LLM friendly (e.g., Ollama); backends abstracted in `core/llm.py`

---

## Table of Contents

- [Quick Start](#quick-start)
- [How It Works](#how-it-works)
- [Repository Structure](#repository-structure)
- [Onboarding Wizard](#onboarding-wizard)
- [Using the App](#using-the-app)
- [Configuration](#configuration)
- [Data & Storage](#data--storage)
- [Dashboard (Optional)](#dashboard-optional)
- [Troubleshooting](#troubleshooting)
- [Roadmap](#roadmap)
- [License](#license)

---

## Quick Start

> Requires Python 3.10+. On Windows, recommend a virtualenv or Conda.

```bash
# 1) Create and activate a virtual environment
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

# 2) Install deps (minimal core)
pip install duckdb numpy rich click pyyaml

# If your embedding pipeline uses HF models:
pip install torch transformers   # (or sentence-transformers if your core/embeddings needs it)
# If you use Ollama locally, install Ollama separately (outside Python)

# 3) Run the onboarding wizard
python -m scripts.setup_wizard

# 4) Start chatting
python app.py chat --user "<YourName>"
```

> Tip: the wizard writes `data/persona.json`, `data/identity.json`, and `data/traits.json`. You can re-run it any time.

---

## How It Works

Architecture:

```text
[User] -> Conversation Loop (app.py)
       |- Fact Extractor -> DuckDB (facts_kv)
       |- Embedding + Vector Store (vectors.npy)
       |- Rerank & Graph Spread -> Graph Store (graph.json)
       |- Snapshot & Memory Pack Composer
       \- LLM Decode (unlimited output if enabled in wizard)
         v
       Reply + Write-back:
       - Store episode (summary + gist + vector)
       - Upsert facts (if any)
       - Update traits from use (runtime only)
       - Persist snapshots for dashboard
```

- The onboarding wizard lets you set everything about the persona (names, role, style sliders, boundaries, skills) and optionally force unlimited tokens.
- Every turn:
  - Your input is embedded -> candidates retrieved -> reranked & expanded via graph.
  - The system includes the last 30 interactions in context (separate from memory recall).
  - Relevant facts/gists are added if they pass gates.
  - The LLM generates a reply using wizard-driven decode overrides.
  - The turn is written as an episode; any detected facts are upserted; traits reinforce from actual usage.

---

## Repository Structure

```
TRAIT_ENGINE_POC/
├── core/
│   ├── context.py
│   ├── embeddings.py
│   ├── llm.py
│   ├── memcards.py
│   ├── memory_pipeline.py
│   ├── persona.py        
│   ├── onboarding.py    
│   ├── realtime.py
│   ├── retrieval.py
│   ├── traits.py
│   └── utils.py
├── scripts/
│   ├── init.py
│   ├── seed_identity.py
│   ├── seed_persona.py
│   └── setup_wizard.py   # interactive onboarding
├── snapshot-view/
│   ├── dashboard.py
│   └── memories_only.py
├── stores/
│   ├── graph_store.py
│   ├── tabular.py        
│   └── vector_store.py
├── data/                 # runtime data (created by wizard/app)
│   ├── traits.json       # added after -> Init setup
│   ├── persona.json      # added after -> Init setup
│   ├── identity.json     # added after -> Init setup
│   ├── episodes.parquet  # added after -> Init setup
│   └── facts.parquet     # added after -> Init setup
├── app.py
├── config.py             # models, paths, retrieval settings
└── README.md
```

---

## Onboarding Wizard

Run:

```bash
python -m scripts.setup_wizard
```

You'll be asked to set:
- AI name and your name
- Primary role (chat, coding, creative writer, research assistant, tutor, productivity coach, data analyst, or custom)
- Unlimited tokens (on/off), long-form vs concise preferences
- Style sliders (formality, warmth, humour, directness, creativity, skepticism, analytical, verbosity)
- Boundaries (refuse when unsure, cite sources if browsing, avoid pretending human, avoid third-person self)
- Taboo topics (optional)
- Traits: add any number as `name=weight` (stored as `strength`)
- Skills/mode preferences for selected role (optional)

Outputs:
- `data/persona.json`
- `data/identity.json`
- `data/traits.json` (only what you define-no defaults)

---

## Using the App

Start the chat loop:

```bash
python app.py chat --user "<YourName>"
```

What happens:
- The app builds a system prompt via `core/persona.system_primer()` (wizard-driven).
- It includes the last 30 interactions as rolling context.
- It gates in facts/gists when relevant.
- It generates a reply using `decode_overrides()` which honours your `no_token_limits` setting.
- It writes the turn to DuckDB and updates traits based on use.

Utilities:

```bash
# Export read-only Parquet snapshots for analytics/dashboard
python app.py snapshot

# Inspect episodes (latest first)
python app.py episodes --user "<YourName>" --limit 12

# List current traits
python app.py traits

# List facts (optionally scoped by user)
python app.py facts --user "<YourName>"

# (Optional) Import persona/traits/facts/episodes from YAML
python app.py seed-persona --path path/to/persona.yaml
```

---

## Configuration

Most common knobs live in `config.py`. Key items:

```python
# Paths
DATA_DIR, STORE_DIR, DB_PATH, GRAPH_PATH, TRAITS_PATH
VECTORS_NPY

# Embeddings
EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"
EMBEDDING_DIM = 1024
EMBEDDING_NORMALIZE = True

# Retrieval / rerank / graph
CANDIDATE_K = 120
FINAL_K = 16
SPREAD_DEPTH = 2
SPREAD_DECAY = 0.85

# Decode / generation
MODEL_BACKEND = "ollama"     # see core/llm.py
CONTINUE_MAX_SEGMENTS = 12   # segment-stitching upper bound

# Memory gates (tune when recall feels noisy or too quiet)
MEMORY_GATE_MIN_SIM = 0.30
MEMORY_GATE_MIN_HITS = 1
MEMORY_GATE_JACCARD = 0.07
MEMORY_GATE_GISTS = 8
```

> Token limits are controlled in the wizard (and mapped by `core/persona.decode_overrides()`); if "unlimited" is on, we set `num_predict=-1` (llama.cpp/Ollama) and `max_tokens=None` (OpenAI-style).

---

## Data & Storage

- DuckDB at `memory.duckdb`:
  - `episodes` - one row per turn (with `summary`, `gist`, `ts`, `emotion=''`)
  - `facts_kv` - key/value facts (assistant/user)
- Vectors at `vectors.npy` (+ id list) for quick similarity search
- Graph at `graph.json` for episode/concept edges
- Parquet snapshots at `data/episodes.parquet` & `data/facts.parquet` (read-only; safe for dashboards)

Backups:
```bash
python app.py snapshot
# Copy files from /data and {memory.duckdb, graph.json, vectors.npy} for a full backup
```

Privacy:
- Everything is local by default. No external writes unless your `core/llm.py` backend calls an external API you configure.

---

## Dashboard (Optional)

If you use the included Streamlit dashboard:

```bash
pip install streamlit
streamlit run dashboard.py
```

It reads snapshots only (`data/*.parquet`)-never the live DB.

---

## Troubleshooting

LLM KeyError: `'num_predict'` in persona decode  
Ensure `config.py` defines `DECODE_BOUNDS` or use the updated `core/persona.py` which sets safe defaults and reads wizard flags.

Memories labelled "neutral"  
This version writes `emotion=''`. If you see "neutral", it's old rows or a previous default.
- Update schema default in `stores/tabular.py` (already done here).
- One-off clean:
  ```sql
  UPDATE episodes SET emotion = '' WHERE emotion IS NULL OR LOWER(emotion)='neutral';
  ```

Vector search returns nothing on a fresh DB  
That's normal until several messages are stored. Talk to the assistant a few turns, then try again.

Ollama truncating outputs  
Re-run the wizard and ensure "Disable output token limits" is enabled.
`core/persona.decode_overrides()` sets `num_predict=-1` for llama.cpp/Ollama.

---

## Roadmap

- Optional dialogue history size exposed in the wizard (currently fixed at 30).
- Pluggable rerankers and CE models.
- Simple import/export for memories and traits.
- More backends in `core/llm.py` + per-backend decode mapping.

---

## License

MIT `LICENSE`

---

### Credits

Built by Uhmbrella (POC). Uses DuckDB for storage and a simple numpy vector index; LLM backend is pluggable (Ollama-friendly).
