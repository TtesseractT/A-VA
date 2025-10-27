# stores/vector_store.py
from __future__ import annotations
import os, time, json
from pathlib import Path
from typing import List, Sequence, Optional, Tuple
import numpy as np
import config

class VectorStore:
    """
    Simple numpy-based vector store with atomic saves and cosine search.

    Files (fixed names):
      - vectors.npy          (float32 [n, dim])
      - vectors.ids.txt      (one id per line)
      - vectors.meta.json    (dim, normalize, count)

    Optional bounded rotating backups if config.VSTORE_ROTATE_KEEP > 0:
      - vectors.YYYYMMDDhhmmss.npy
      - vectors.YYYYMMDDhhmmss.ids.txt
    """

    def __init__(self, dim: int, path: Path, normalize: bool = True):
        self.dim = int(dim)
        self.dir = Path(path).parent if Path(path).suffix else Path(path)
        self.dir.mkdir(parents=True, exist_ok=True)

        self.vec_path = self.dir / "vectors.npy"
        self.ids_path = self.dir / "vectors.ids.txt"
        self.meta_path = self.dir / "vectors.meta.json"

        self.normalize = bool(normalize)
        self.ids: List[str] = []
        self.vecs: np.ndarray = np.zeros((0, self.dim), dtype=np.float32)

    # ---------------- public API ----------------
    def load(self) -> None:
        """Load fixed files if present; else start empty."""
        if self.vec_path.exists() and self.ids_path.exists():
            self._load_pair(self.vec_path, self.ids_path)
        else:
            # start empty
            self.ids = []
            self.vecs = np.zeros((0, self.dim), dtype=np.float32)
            self._write_meta()

    def add(self, vecs: np.ndarray, ids: Sequence[str]) -> None:
        vecs = np.asarray(vecs, dtype=np.float32)
        if vecs.ndim != 2 or vecs.shape[1] != self.dim:
            raise ValueError(f"Vector dims mismatch: got {vecs.shape}, expected (*,{self.dim})")
        if self.normalize and vecs.size:
            vecs = _l2norm(vecs)
        self.vecs = np.vstack([self.vecs, vecs]) if self.vecs.size else vecs.copy()
        self.ids.extend([str(x) for x in ids])

    def get_vectors(self, ids: Sequence[str]) -> np.ndarray:
        if not self.ids or not ids:
            return np.zeros((0, self.dim), dtype=np.float32)
        # map ids to indices (skip missing)
        idx = [self.ids.index(i) for i in ids if i in self.ids]
        if not idx:
            return np.zeros((0, self.dim), dtype=np.float32)
        return self.vecs[np.asarray(idx, dtype=np.int64)]

    def search(self, qvec: np.ndarray, k: int = 50) -> List[Tuple[str, float]]:
        """
        Cosine search (dot product on L2-normalized vectors).
        Returns list of (id, score) sorted desc.
        """
        if self.vecs.shape[0] == 0:
            return []
        q = np.asarray(qvec, dtype=np.float32).reshape(-1)
        if q.shape[0] != self.dim:
            raise ValueError(f"Query dim mismatch: got {q.shape[0]}, expected {self.dim}")
        if self.normalize:
            q = q / (np.linalg.norm(q) + 1e-12)
            sims = self.vecs @ q
        else:
            # compute cosine explicitly if store not normalized
            qn = np.linalg.norm(q) + 1e-12
            vn = np.linalg.norm(self.vecs, axis=1) + 1e-12
            sims = (self.vecs @ q) / (vn * qn)
        # top-k
        k = int(max(1, min(k, sims.shape[0])))
        idx = np.argpartition(-sims, k - 1)[:k]
        # sort exact top-k
        idx = idx[np.argsort(-sims[idx])]
        return [(self.ids[i], float(sims[i])) for i in idx]

    def save(self) -> None:
        """
        Atomic save to fixed files. If VSTORE_ROTATE_KEEP > 0, also keep at most N timestamped backups.
        """
        self._write_meta()

        vec_path = self.vec_path
        ids_path = self.ids_path
        vec_path.parent.mkdir(parents=True, exist_ok=True)

        # Use explicit temp names (NumPy adds .npy if given a suffix it doesn't like).
        tmp_vec = vec_path.with_name(vec_path.name + ".tmp")
        tmp_ids = ids_path.with_name(ids_path.name + ".tmp")

        # Write ids temp
        with open(tmp_ids, "w", encoding="utf-8") as f:
            f.write("\n".join(self.ids))

        # Write vectors temp using a file handle to keep exact filename
        with open(tmp_vec, "wb") as f:
            np.save(f, self.vecs, allow_pickle=False)

        # Atomic replace
        os.replace(tmp_vec, vec_path)
        os.replace(tmp_ids, ids_path)

        # Optional rotating backups
        keep = int(getattr(config, "VSTORE_ROTATE_KEEP", 0) or 0)
        if keep > 0:
            stamp = _timestamp()
            b_vec = self.dir / f"vectors.{stamp}.npy"
            b_ids = self.dir / f"vectors.{stamp}.ids.txt"
            self._atomic_copy(vec_path, b_vec)
            self._atomic_copy(ids_path, b_ids)
            self._prune_backups(keep)

    # ---------------- internal ----------------
    def _load_pair(self, vec_p: Path, ids_p: Path) -> None:
        ids = []
        if ids_p.exists():
            with open(ids_p, "r", encoding="utf-8") as f:
                ids = [ln.strip() for ln in f if ln.strip()]
        vecs = np.zeros((0, self.dim), dtype=np.float32)
        if vec_p.exists():
            v = np.load(vec_p, allow_pickle=False)
            if v.ndim == 1 and v.size == 0:
                v = np.zeros((0, self.dim), dtype=np.float32)
            if v.ndim != 2 or v.shape[1] != self.dim:
                if v.size == 0:
                    v = np.zeros((0, self.dim), dtype=np.float32)
                else:
                    raise ValueError(f"Bad vector shape {v.shape}, expected (*,{self.dim})")
            vecs = v.astype(np.float32, copy=False)
        if self.normalize and vecs.size:
            vecs = _l2norm(vecs)
        if vecs.shape[0] != len(ids):
            ids = ids[:vecs.shape[0]]
        self.ids = ids
        self.vecs = vecs
        self._write_meta()

    def _write_meta(self) -> None:
        meta = {"dim": self.dim, "normalize": self.normalize, "count": int(self.vecs.shape[0])}
        try:
            with open(self.meta_path, "w", encoding="utf-8") as f:
                json.dump(meta, f)
        except Exception:
            pass

    def _atomic_copy(self, src: Path, dst: Path) -> None:
        tmp = dst.with_name(dst.name + ".tmp")
        with open(src, "rb") as s, open(tmp, "wb") as d:
            d.write(s.read())
        os.replace(tmp, dst)

    def _prune_backups(self, keep: int) -> None:
        bv = sorted(self.dir.glob("vectors.*.npy"), key=lambda p: p.stat().st_mtime, reverse=True)
        bi = sorted(self.dir.glob("vectors.*.ids.txt"), key=lambda p: p.stat().st_mtime, reverse=True)
        for p in bv[keep:]:
            try: p.unlink()
            except Exception: pass
        for p in bi[keep:]:
            try: p.unlink()
            except Exception: pass


# ---------- helpers ----------
def _l2norm(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
    return x / n

def _timestamp() -> str:
    return time.strftime("%Y%m%d%H%M%S", time.localtime())
