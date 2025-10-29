# dashboard.py
from __future__ import annotations

import os, json, glob, time, math
from pathlib import Path
from typing import Tuple, Optional

import duckdb
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import networkx as nx

try:
    import umap
except Exception:
    umap = None  # optional; we'll guard

# -------- local project config --------
import config

DB_PATH: Path = Path(config.DB_PATH)
GRAPH_PATH: Path = Path(config.GRAPH_PATH)
DATA_DIR: Path = Path(getattr(config, "DATA_DIR", Path(__file__).parent / "data"))
VEC_GLOB = str((Path(__file__).parent) / "vectors*.npy")

# ---------- Page setup ----------
st.set_page_config(page_title="Trait Engine Dashboard", layout="wide")
st.title("🧠 Trait Engine — Live Memory & Graph")

# ---------- Sidebar controls ----------
col1, col2 = st.sidebar.columns(2)
# Autorefresh disabled (Streamlit has no reliable built-in timer without external components)
refresh_note = col1.text_input("Refresh hint (optional)", value="")
max_rows = col2.number_input("Max episodes", min_value=50, max_value=5000, value=400, step=50)
st.sidebar.caption("Use the toolbar's 'Rerun' button to refresh, or change any control above.")

# ---------- Helpers ----------
def _duck_df(sql: str, params: Tuple = ()) -> pd.DataFrame:
    """Short-lived, read-only DuckDB connection to avoid file locks."""
    try:
        con = duckdb.connect(str(DB_PATH), read_only=True)
        try:
            return con.execute(sql, params).fetch_df()
        finally:
            con.close()
    except Exception as e:
        st.error(f"DuckDB read failed: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=2.0, show_spinner=False)
def load_episodes(limit: int) -> pd.DataFrame:
    if not DB_PATH.exists():
        return pd.DataFrame()
    # DuckDB-friendly: epoch seconds -> TIMESTAMP and age in minutes
    sql = f"""
        SELECT
            id,
            user_id,
            ts,
            to_timestamp(ts) AS ts_dt,
            gist,
            emotion,
            summary,
            usage_count,
            CAST(datediff('minute', to_timestamp(ts), now()) AS DOUBLE) AS age_min
        FROM episodes
        ORDER BY ts DESC
        LIMIT {int(limit)}
    """
    return _duck_df(sql)

@st.cache_data(ttl=2.0, show_spinner=False)
def load_facts() -> pd.DataFrame:
    if not DB_PATH.exists():
        return pd.DataFrame()
    sql = """
        SELECT user_id, subject, key, value, confidence, last_seen, usage_count
        FROM facts_kv
        ORDER BY last_seen DESC
        LIMIT 500
    """
    return _duck_df(sql)

@st.cache_data(ttl=3.0, show_spinner=False)
def load_traits() -> dict:
    p = Path(getattr(config, "TRAITS_PATH", DATA_DIR / "traits.json"))
    try:
        if p.exists():
            with open(p, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception as e:
        st.warning(f"Traits read failed: {e}")
    return {"traits": [], "last_mood": None}

@st.cache_data(ttl=3.0, show_spinner=False)
def load_graph() -> nx.Graph:
    g = nx.Graph()
    if GRAPH_PATH.exists():
        try:
            with open(GRAPH_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            g = nx.node_link_graph(data, edges="links")
        except Exception as e:
            st.warning(f"Graph read failed: {e}")
    return g

def _find_latest_vectors() -> Tuple[Optional[Path], Optional[Path]]:
    # Look for vectors.*.npy (timestamped) or fallback to vectors.npy
    npy_candidates = sorted(glob.glob(VEC_GLOB), key=os.path.getmtime, reverse=True)
    npy_path = Path(npy_candidates[0]) if npy_candidates else None

    # Try to guess matching ids file (ids.*.txt) or fallback to vectors.ids.txt
    ids_candidates = []
    if npy_path is not None:
        stem = npy_path.stem  # e.g., vectors.1761438071137
        ts_part = ".".join(stem.split(".")[1:])
        if ts_part:
            ids_candidates = sorted(
                glob.glob(str((Path(__file__).parent) / f"*{ts_part}*.txt")),
                key=os.path.getmtime, reverse=True
            )
    if not ids_candidates:
        ids_candidates = sorted(
            glob.glob(str((Path(__file__).parent) / "vectors*.ids*.txt")) +
            glob.glob(str((Path(__file__).parent) / "vectors.ids.txt")),
            key=os.path.getmtime, reverse=True
        )
    ids_path = Path(ids_candidates[0]) if ids_candidates else None
    return npy_path, ids_path

@st.cache_data(ttl=5.0, show_spinner=False)
def load_vectors() -> Tuple[np.ndarray, list[str]]:
    npy_path, ids_path = _find_latest_vectors()
    if not npy_path or not npy_path.exists():
        return np.empty((0, getattr(config, "EMBEDDING_DIM", 1024))), []
    try:
        vecs = np.load(npy_path)
        if ids_path and ids_path.exists():
            with open(ids_path, "r", encoding="utf-8") as f:
                ids = [ln.strip() for ln in f if ln.strip()]
        else:
            # fallback to synthetic ids
            ids = [f"v{i:06d}" for i in range(vecs.shape[0])]
        if vecs.ndim != 2:
            vecs = np.atleast_2d(vecs)
        return vecs, ids
    except Exception as e:
        st.warning(f"Vector load failed: {e}")
        return np.empty((0, getattr(config, "EMBEDDING_DIM", 1024))), []

def _umap_project(vecs: np.ndarray, n_neighbors: int = 25, min_dist: float = 0.2, metric: str = "cosine") -> np.ndarray:
    if umap is None or vecs is None or len(vecs) == 0:
        return np.empty((0, 2))
    n = vecs.shape[0]
    if n < 3:
        return np.empty((0, 2))
    nn = max(2, min(n - 1, int(n_neighbors)))
    reducer = umap.UMAP(n_neighbors=nn, min_dist=min_dist, n_components=2, metric=metric, random_state=42)
    return reducer.fit_transform(vecs)

def _plot_umap(coords: np.ndarray, ids: list[str]) -> go.Figure:
    if coords.size == 0:
        return go.Figure()
    fig = go.Figure(
        data=[
            go.Scatter(
                x=coords[:, 0],
                y=coords[:, 1],
                mode="markers",
                marker=dict(size=6, opacity=0.75),
                text=ids,
                hovertemplate="<b>%{text}</b><extra></extra>",
            )
        ]
    )
    fig.update_layout(
        dragmode="pan",
        template="plotly_white",
        height=600,
        margin=dict(l=10, r=10, t=30, b=10),
        title="Vector Space (UMAP)"
    )
    return fig

def _plot_graph(g: nx.Graph) -> go.Figure:
    if g is None or g.number_of_nodes() == 0:
        return go.Figure()
    # layout
    pos = nx.kamada_kawai_layout(g) if g.number_of_nodes() < 400 else nx.spring_layout(g, seed=42, k=1 / math.sqrt(g.number_of_nodes()))
    # edges
    edge_x, edge_y = [], []
    for u, v in g.edges():
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]
    edge_trace = go.Scatter(x=edge_x, y=edge_y, mode="lines", line=dict(width=0.6, color="#bbb"), hoverinfo="none")
    # nodes
    node_x, node_y, texts, colors, sizes = [], [], [], [], []
    for n, attrs in g.nodes(data=True):
        x, y = pos[n]
        node_x.append(x); node_y.append(y)
        t = attrs.get("type", "node")
        texts.append(f"{n} — {t}")
        if t == "Episode":
            colors.append("#1f77b4"); sizes.append(8)
        elif t == "Init":
            colors.append("#2ca02c"); sizes.append(10)
        else:
            colors.append("#ff7f0e"); sizes.append(9)
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode="markers",
        marker=dict(size=sizes, color=colors, line=dict(width=0.5, color="#444")),
        text=texts,
        hovertemplate="%{text}<extra></extra>",
    )
    fig = go.Figure(data=[edge_trace, node_trace])
    fig.update_layout(template="plotly_white", height=600, margin=dict(l=10, r=10, t=30, b=10), title="Conversation Graph")
    return fig

# ---------- Data loads ----------
episodes_df = load_episodes(max_rows)
facts_df = load_facts()
traits_state = load_traits()
graph_nx = load_graph()
vecs, vec_ids = load_vectors()

# ---------- Top KPI row ----------
k1, k2, k3, k4 = st.columns(4)
k1.metric("Episodes", int(episodes_df.shape[0]) if not episodes_df.empty else 0)
k2.metric("Facts", int(facts_df.shape[0]) if not facts_df.empty else 0)
k3.metric("Graph nodes", int(graph_nx.number_of_nodes()) if graph_nx else 0)
k4.metric("Vectors", int(vecs.shape[0]) if vecs is not None else 0)

# ---------- Tabs ----------
tab_overview, tab_mem, tab_traits, tab_graph, tab_vec = st.tabs(
    ["Overview", "Memories", "Traits", "Graph", "Vectors"]
)

with tab_overview:
    c1, c2 = st.columns([2, 1])
    # recent episodes
    with c1:
        st.subheader("Recent episodes")
        if episodes_df.empty:
            st.info("No episodes yet.")
        else:
            show_cols = ["ts_dt", "user_id", "gist", "emotion", "usage_count", "age_min", "summary"]
            to_show = episodes_df[show_cols].rename(
                columns={"ts_dt":"time", "age_min":"age (min)"}
            )
            st.dataframe(to_show, use_container_width=True, height=420)
    with c2:
        st.subheader("Latest facts")
        if facts_df.empty:
            st.info("No facts yet.")
        else:
            show_cols = ["subject", "key", "value", "confidence", "usage_count"]
            st.dataframe(facts_df[show_cols].head(20), use_container_width=True, height=420)

with tab_mem:
    st.subheader("All facts (recent first)")
    if facts_df.empty:
        st.info("No facts found.")
    else:
        st.dataframe(facts_df, use_container_width=True, height=520)
    if not episodes_df.empty:
        st.subheader("Episode summaries")
        st.dataframe(
            episodes_df[["id","ts_dt","gist","emotion","summary","usage_count"]],
            use_container_width=True, height=420
        )

with tab_traits:
    st.subheader("Trait state")
    traits = traits_state.get("traits", [])
    last_mood = traits_state.get("last_mood", None)
    st.write(f"Last detected mood: **{last_mood}**" if last_mood else "Last detected mood: (none)")

    if traits:
        tdf = pd.DataFrame(traits)
        for col in ["usage_count","strength"]:
            if col not in tdf.columns:
                tdf[col] = 0
        st.dataframe(tdf.sort_values(["usage_count","strength"], ascending=False), use_container_width=True, height=420)
        fig = go.Figure(data=[go.Bar(x=tdf["name"], y=tdf["strength"])])
        fig.update_layout(template="plotly_white", height=300, title="Trait strengths")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No traits captured yet.")

with tab_graph:
    st.subheader("Relationship graph")
    if graph_nx is None or graph_nx.number_of_nodes() == 0:
        st.info("Graph is empty.")
    else:
        fig = _plot_graph(graph_nx)
        st.plotly_chart(fig, use_container_width=True)

with tab_vec:
    st.subheader("Vector projection (UMAP)")
    if vecs is None or vecs.shape[0] == 0:
        st.info("No vectors available yet.")
    else:
        n_neighbors = st.slider("n_neighbors", min_value=2, max_value=80, value=25, step=1)
        min_dist = st.slider("min_dist", min_value=0.0, max_value=1.0, value=0.2, step=0.05)
        metric = st.selectbox("metric", ["cosine","euclidean"], index=0)

        if umap is None:
            st.error("umap-learn is not installed. Run: pip install umap-learn")
        else:
            coords = _umap_project(vecs, n_neighbors=n_neighbors, min_dist=min_dist, metric=metric)
            fig = _plot_umap(coords, vec_ids if coords.size else [])
            st.plotly_chart(fig, use_container_width=True)

st.caption("Reads are short-lived and read-only to avoid DB locks.")
