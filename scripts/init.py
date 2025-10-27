# scripts/init.py
from __future__ import annotations
from pathlib import Path
import json

import config
from stores.vector_store import VectorStore
from stores.graph_store import GraphStore
from stores.tabular import TabularStore

def main():
    # ensure directories exist
    Path(config.DATA_DIR).mkdir(parents=True, exist_ok=True)
    Path(config.DB_PATH).parent.mkdir(parents=True, exist_ok=True)
    Path(config.FAISS_PATH).parent.mkdir(parents=True, exist_ok=True)
    Path(config.GRAPH_PATH).parent.mkdir(parents=True, exist_ok=True)
    Path(config.TRAITS_PATH).parent.mkdir(parents=True, exist_ok=True)

    # empty vector store (writes first versioned snapshot)
    vs = VectorStore(dim=getattr(config, "EMBEDDING_DIM", 1024),
                     path=config.FAISS_PATH,
                     normalize=getattr(config, "EMBEDDING_NORMALIZE", True))
    vs.load()   # harmless if nothing there
    vs.save()   # creates vectors.<ts>.npy + ids + meta

    # fresh graph
    gs = GraphStore(config.GRAPH_PATH)
    gs.g.clear()
    gs.save()

    # fresh duckdb schema
    db = TabularStore(config.DB_PATH)
    # nothing to insert, schema is ensured in constructor
    # write snapshots so dashboard can open immediately
    db.export_snapshots(config.DATA_DIR)

    # minimal traits file
    traits_path = Path(config.TRAITS_PATH)
    if not traits_path.exists():
        traits_path.write_text(json.dumps({"traits": [], "mood": {}, "last_mood": None}, ensure_ascii=False), encoding="utf-8")

    print("Initialisation complete.")

if __name__ == "__main__":
    main()



'''# scripts/init.py
from __future__ import annotations
from pathlib import Path
import json, time
import config
from stores.tabular import TabularStore
from stores.vector_store import VectorStore
from stores.graph_store import GraphStore

def main():
    Path(config.DATA_DIR).mkdir(parents=True, exist_ok=True)

    # Tabular (DuckDB)
    db = TabularStore(config.DB_PATH)

    # Vector store (FAISS)
    vs = VectorStore(dim=config.EMBEDDING_DIM, path=config.FAISS_PATH, normalize=config.EMBEDDING_NORMALIZE)
    vs.save()  # create empty index & ids list

    # Graph
    gs = GraphStore(config.GRAPH_PATH)
    gs.save()

    # Traits json (created by TraitEngine on first load)
    traits_path = Path(config.TRAITS_PATH)
    if not traits_path.exists():
        traits_path.parent.mkdir(parents=True, exist_ok=True)
        traits_path.write_text(json.dumps([], ensure_ascii=False), encoding="utf-8")

    print("Initialisation complete.")

if __name__ == "__main__":
    main()
'''