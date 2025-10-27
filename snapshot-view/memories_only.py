# memories_only.py
import json, time, glob
from pathlib import Path
import pandas as pd
import streamlit as st
import config

st.set_page_config(page_title="Memories Viewer", layout="wide")

DATA_DIR = Path(config.DATA_DIR)

# ---- Sidebar / controls ----
st.sidebar.header("Memories")
autorefresh = st.sidebar.toggle("Auto-refresh", value=True)
refresh_sec = st.sidebar.slider("Refresh every (sec)", 2, 15, 4)
user_filter = st.sidebar.text_input("User (blank = all)", "")

if autorefresh:
    st.sidebar.caption("Auto-refresh is on.")
    # force re-run by changing the query param (replaces deprecated experimental API)
    st.query_params.update({"t": str(int(time.time()))})

# ---- Helpers: load latest snapshot files (JSON or CSV) ----
def _latest(globpat: str):
    files = sorted(glob.glob(globpat), key=lambda p: Path(p).stat().st_mtime if Path(p).exists() else 0, reverse=True)
    return files[0] if files else None

def load_snapshot_df(kind: str) -> pd.DataFrame:
    """
    kind = 'episodes' or 'facts'
    Accepts any of:
      data/episodes*.json  or  data/episodes*.csv
      data/facts*.json     or  data/facts*.csv
    """
    if kind == "episodes":
        j = _latest(str(DATA_DIR / "episodes*.json"))
        c = _latest(str(DATA_DIR / "episodes*.csv"))
        cols = ["id","user_id","ts","summary","vector_dim","strength","usage_count","last_accessed","gist","emotion"]
    else:
        j = _latest(str(DATA_DIR / "facts*.json"))
        c = _latest(str(DATA_DIR / "facts*.csv"))
        cols = ["user_id","subject","key","value","confidence","last_seen","usage_count"]

    if j and Path(j).exists():
        try:
            df = pd.read_json(j, orient="records", lines=False)
            return df
        except Exception:
            pass
    if c and Path(c).exists():
        try:
            df = pd.read_csv(c)
            return df
        except Exception:
            pass

    # graceful empty
    return pd.DataFrame(columns=cols)

# ---- Load data (snapshots written by app.py -> TabularStore.export_snapshots) ----
ep_df = load_snapshot_df("episodes")
fa_df = load_snapshot_df("facts")

# ---- Filter by user ----
if user_filter.strip():
    ep_df = ep_df[ep_df["user_id"].astype(str).str.contains(user_filter.strip(), case=False, na=False)]
    fa_df = fa_df[fa_df["user_id"].astype(str).str.contains(user_filter.strip(), case=False, na=False)]

# ---- Display Facts ----
st.subheader("Facts / Memory Cards")
if fa_df.empty:
    st.info("No facts snapshot found yet. Start chatting so the app writes snapshots, or run:  `python app.py snapshot`")
else:
    # nicer columns
    show_cols = ["user_id","subject","key","value","confidence","usage_count","last_seen"]
    show_cols = [c for c in show_cols if c in fa_df.columns]
    # sort by recency/confidence if present
    sort_cols = [c for c in ["last_seen","confidence","usage_count"] if c in fa_df.columns]
    if sort_cols:
        fa_df = fa_df.sort_values(sort_cols, ascending=[False, False, False])
    st.dataframe(fa_df[show_cols], use_container_width=True, height=320)

# ---- Display Recent Episodes ----
st.subheader("Recent Episodes")
if ep_df.empty:
    st.info("No episodes snapshot found yet.")
else:
    # compute age_min if ts present
    if "ts" in ep_df.columns:
        now = time.time()
        ep_df["age_min"] = (now - ep_df["ts"].astype(float)) / 60.0
    show_cols = [c for c in ["id","user_id","gist","emotion","summary","usage_count","age_min"] if c in ep_df.columns]
    ep_df = ep_df.sort_values("ts", ascending=False) if "ts" in ep_df.columns else ep_df
    st.dataframe(ep_df[show_cols], use_container_width=True, height=360)

# ---- Footer hint ----
st.caption(
    "This page reads read-only JSON/CSV snapshots from config.DATA_DIR. "
    "The chat app writes them automatically after each turn; you can also run `python app.py snapshot` to dump them."
)
