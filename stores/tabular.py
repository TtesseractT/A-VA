# stores/tabular.py
from __future__ import annotations
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
import duckdb
import time

class TabularStore:
    def __init__(self, db_path: Path):
        self.path = Path(db_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.con = duckdb.connect(str(self.path))
        self._ensure_schema()

    # -------------------- Core schema --------------------
    def _ensure_schema(self):
        """
        Minimal schema: episodes + facts_kv (+ view facts).
        No persona-axis tables; all persona comes from wizard JSON.
        'emotion' is kept for compatibility but defaults to '' (blank).
        Column order is stable so gist is index 8 and emotion index 9.
        """
        # Episodes table (core turn storage)
        self.con.execute("""
            CREATE TABLE IF NOT EXISTS episodes (
                id TEXT PRIMARY KEY,
                user_id TEXT,
                ts DOUBLE,
                summary TEXT,
                vector_dim INTEGER,
                strength DOUBLE,
                usage_count INTEGER,
                last_accessed DOUBLE,
                gist TEXT DEFAULT '',
                emotion TEXT DEFAULT '')
        """)

        # Back-compat for older DBs that predate gist/emotion columns or defaults
        self.con.execute("ALTER TABLE episodes ADD COLUMN IF NOT EXISTS gist TEXT")
        self.con.execute("ALTER TABLE episodes ADD COLUMN IF NOT EXISTS emotion TEXT DEFAULT ''")

        # Key/Value facts
        self.con.execute("""
            CREATE TABLE IF NOT EXISTS facts_kv (
                user_id TEXT,
                subject TEXT,
                key TEXT,
                value TEXT,
                confidence DOUBLE,
                last_seen DOUBLE,
                usage_count INTEGER
            )
        """)

        # Simple compatibility view
        self.con.execute("CREATE OR REPLACE VIEW facts AS SELECT * FROM facts_kv")

        # Optional: small indices to speed common lookups (DuckDB supports CREATE INDEX)
        try:
            self.con.execute("CREATE INDEX IF NOT EXISTS idx_episodes_user_ts ON episodes(user_id, ts)")
            self.con.execute("CREATE INDEX IF NOT EXISTS idx_facts_user_last ON facts_kv(user_id, last_seen)")
        except Exception:
            # Older DuckDB versions may not support CREATE INDEX; safe to ignore
            pass

    # -------------------- Episodes --------------------
    def insert_episode(self, row: Dict[str, Any]):
        """
        Insert an episode row. 'emotion' is optional and defaults to '' (blank).
        """
        self.con.execute("""
            INSERT INTO episodes (
                id, user_id, ts, summary, vector_dim, strength, usage_count, last_accessed, gist, emotion
            ) VALUES (?,?,?,?,?,?,?,?,?,?)
        """, (
            row["id"],
            row["user_id"],
            float(row["ts"]),
            row.get("summary", ""),
            int(row.get("vector_dim", 0)),
            float(row.get("strength", 0.0)),
            int(row.get("usage_count", 0)),
            float(row.get("last_accessed", row["ts"])),
            row.get("gist", ""),
            row.get("emotion", "") or ""
        ))

    def update_episode_usage(self, eid: str, *, usage_count: int, last_accessed: float, strength: float):
        self.con.execute("""
            UPDATE episodes
            SET usage_count=?, last_accessed=?, strength=?
            WHERE id=?
        """, (int(usage_count), float(last_accessed), float(strength), eid))

    def list_episodes(self, user_id: Optional[str], limit: int = 50) -> List[Tuple]:
        if user_id:
            cur = self.con.execute(
                "SELECT * FROM episodes WHERE user_id=? ORDER BY ts DESC LIMIT ?",
                (user_id, int(limit))
            )
        else:
            cur = self.con.execute(
                "SELECT * FROM episodes ORDER BY ts DESC LIMIT ?",
                (int(limit),)
            )
        return cur.fetchall()

    # -------------------- Facts --------------------
    def list_facts(self, user_id: Optional[str] = None) -> List[Tuple]:
        if user_id:
            cur = self.con.execute(
                "SELECT * FROM facts WHERE user_id=? OR subject='assistant' ORDER BY last_seen DESC",
                (user_id,)
            )
        else:
            cur = self.con.execute("SELECT * FROM facts ORDER BY last_seen DESC")
        return cur.fetchall()

    # -------------------- Snapshots (dashboard) --------------------
    def export_snapshots(self, out_dir: Path):
        """
        Write read-only Parquet snapshots so the dashboard never opens the live DB.
        """
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        ep_path = (out_dir / "episodes.parquet").as_posix()
        fc_path = (out_dir / "facts.parquet").as_posix()

        # Episodes snapshot
        self.con.execute(f"COPY (SELECT * FROM episodes) TO '{ep_path}' (FORMAT PARQUET)")

        # Facts snapshot (view may not exist on very old schemas)
        try:
            self.con.execute(f"COPY (SELECT * FROM facts) TO '{fc_path}' (FORMAT PARQUET)")
        except Exception:
            self.con.execute(f"COPY (SELECT * FROM facts_kv) TO '{fc_path}' (FORMAT PARQUET)")
