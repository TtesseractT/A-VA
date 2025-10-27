# stores/graph_store.py
from __future__ import annotations

import json, os, time
from pathlib import Path
import networkx as nx


class GraphStore:
    """
    Thin wrapper around a NetworkX graph with JSON (node_link) persistence.
    - Always guarantees the presence of a bootstrap node 'ep_init'
    - Provides has_node, add_node, add_edge convenience
    - Atomic save to avoid partial writes on Windows
    """

    def __init__(self, path: Path | str):
        self.path = Path(path)
        self.g: nx.Graph = nx.Graph()

    # -------- basics --------
    def has_node(self, node_id: str) -> bool:
        return self.g.has_node(node_id)

    def add_node(self, node_id: str, **attrs):
        # idempotent add/update
        if not self.g.has_node(node_id):
            self.g.add_node(node_id, **attrs)
        else:
            # merge/overwrite attrs
            for k, v in (attrs or {}).items():
                self.g.nodes[node_id][k] = v

    def add_edge(self, u: str, v: str, **attrs):
        """Accumulate edge weight if it already exists, then update other attrs."""
        if self.g.has_edge(u, v):
            # accumulate weight if provided
            w_new = float(attrs.get("weight", 0.0))
            w_old = float(self.g[u][v].get("weight", 0.0))
            if w_new:
                self.g[u][v]["weight"] = w_old + w_new
            # merge any other attributes
            for k, val in attrs.items():
                if k != "weight":
                    self.g[u][v][k] = val
        else:
            self.g.add_edge(u, v, **attrs)

    # -------- persistence --------
    def _ensure_bootstrap(self):
        if not self.g.has_node("ep_init"):
            # A light bootstrap node so the graph never visualises as empty
            self.g.add_node("episode_init", type="Init", created=time.time())

    def load(self):
        """Load graph from disk if present; otherwise start fresh with ep_init."""
        if self.path.exists():
            try:
                with open(self.path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                # Use explicit edges="links" to silence NetworkX future warnings
                self.g = nx.node_link_graph(data, edges="links")
            except Exception:
                # fall back to empty if the file is corrupt
                self.g = nx.Graph()
        else:
            self.g = nx.Graph()
        self._ensure_bootstrap()

    def save(self):
        """Atomic save using a temporary file then os.replace (Windows-safe)."""
        self._ensure_bootstrap()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        data = nx.node_link_data(self.g, edges="links")
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
        os.replace(tmp, self.path)

    # -------- convenience (optional) --------
    def neighbors(self, node_id: str):
        if not self.g.has_node(node_id):
            return []
        return list(self.g.neighbors(node_id))

    def nodes(self, data: bool = False):
        return self.g.nodes(data=data)

    def edges(self, data: bool = False):
        return self.g.edges(data=data)
