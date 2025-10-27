# core/retrieval.py
from typing import List, Tuple, Dict, Optional
import math, time, re
import numpy as np

try:
    import config
    MMR_LAMBDA    = float(getattr(config, "MMR_LAMBDA", 0.33))
    USE_CE        = bool(getattr(config, "USE_CROSS_ENCODER", True))
    CE_MODEL_NAME = getattr(config, "CE_MODEL", "BAAI/bge-reranker-large")
    CE_TOPN       = int(getattr(config, "CE_TOPN", 40))
    CANDIDATE_K   = int(getattr(config, "CANDIDATE_K", 120))
    FINAL_K       = int(getattr(config, "FINAL_K", 16))
    ALPHA_USAGE   = float(getattr(config, "ALPHA_USAGE", 0.25))
    BETA_TIME     = float(getattr(config, "BETA_TIME", 8e-6))
except Exception:
    MMR_LAMBDA=0.33; USE_CE=True; CE_MODEL_NAME="BAAI/bge-reranker-large"
    CE_TOPN=40; CANDIDATE_K=120; FINAL_K=16; ALPHA_USAGE=0.25; BETA_TIME=8e-6

STOP = set(["the","a","an","and","or","to","of","for","in","on","with","by","is","it","that","this","as","be","are","was","were","at","from","about","into","over","than","then","so","but","if","do","did","does","i","you"])

_CE = None
def _get_ce():
    global _CE
    if _CE is not None: return _CE
    if not USE_CE: return None
    try:
        from sentence_transformers import CrossEncoder
        _CE = CrossEncoder(CE_MODEL_NAME)
    except Exception:
        _CE = None
    return _CE

def _cos(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))

def _tokens(t: str):
    return [w for w in re.findall(r"[a-z0-9']+", (t or "").lower()) if w not in STOP]

def rank_with_bias(raw: List[Tuple[str, float]], meta: Dict[str, Dict],
                   alpha: float=ALPHA_USAGE, beta: float=BETA_TIME):
    out=[]
    now=time.time()
    for eid, sim in raw:
        m = meta.get(eid, {})
        usage = float(m.get("usage_count",0))
        last  = float(m.get("last_accessed", now))
        age = max(1.0, now-last)
        rec = 1.0/age
        score = sim * (1.0 + alpha*math.log1p(usage)) * (1.0 + beta*rec*1e6)
        out.append((eid, score))
    out.sort(key=lambda x:-x[1])
    return out

def mmr_diversify(qvec, cand: List[Tuple[str,float]], id2vec: Dict[str,np.ndarray], k:int=16, lamb:float=MMR_LAMBDA):
    if not cand: return []
    cand = [(eid,s) for eid,s in cand if eid in id2vec]
    if not cand: return []
    chosen=[]
    chosen_ids=set()
    sims={eid:score for eid,score in cand}
    while len(chosen) < min(k,len(cand)):
        best=None; best_sc=-1e9
        for eid,_ in cand:
            if eid in chosen_ids: continue
            rel = sims.get(eid,0.0)
            red = 0.0
            if chosen:
                red = max(_cos(id2vec[eid], id2vec[cid]) for cid,_ in chosen)
            sc = lamb*rel - (1.0-lamb)*red
            if sc>best_sc:
                best_sc=sc; best=eid
        if best is None: break
        chosen.append((best, sims.get(best,0.0))); chosen_ids.add(best)
    return chosen

def _fetch_texts_for_ids(db, ids: List[str]) -> Dict[str,str]:
    out={}
    con=db.con
    for eid in ids:
        row=con.execute("SELECT id, COALESCE(gist, summary) FROM episodes WHERE id=?", (eid,)).fetchone()
        if row and row[0]:
            out[row[0]]=(row[1] or "").strip()
    return out

def ce_rerank(query_text: str, id2text: Dict[str,str], cand: List[Tuple[str,float]], topn:int=CE_TOPN):
    ce=_get_ce()
    if ce is None or not cand: return cand
    subset=cand[:min(topn,len(cand))]
    pairs=[]; keep=[]
    for eid,_ in subset:
        t=id2text.get(eid,"")
        if t:
            pairs.append((query_text,t)); keep.append(eid)
    if not pairs: return subset
    try:
        scores=ce.predict(pairs)
    except Exception:
        return subset
    rer=list(zip(keep,[float(s) for s in scores]))
    rer.sort(key=lambda x:-x[1])
    missing=[(eid,score) for eid,score in subset if eid not in keep]
    return rer+missing

def expand_candidates(qvec: np.ndarray, qtext: str, vs, db,
                      k_candidates:int=CANDIDATE_K, k_final:int=FINAL_K):
    raw = vs.search(qvec, k=k_candidates)  # [(eid,sim)]
    if not raw: return []
    ids=[eid for eid,_ in raw]

    # usage/recency meta
    meta={}
    con=db.con
    for eid in ids:
        m=con.execute("SELECT usage_count,last_accessed FROM episodes WHERE id=?", (eid,)).fetchone()
        if m: meta[eid]={"usage_count": m[0], "last_accessed": m[1]}
    biased=rank_with_bias(raw, meta)

    # hybrid keyword boost
    shortlist=[eid for eid,_ in biased[:max(k_final*4,60)]]
    id2text=_fetch_texts_for_ids(db, shortlist)
    qtok=set(_tokens(qtext))
    boosted=[]
    for eid, sc in biased:
        if eid not in id2text: 
            boosted.append((eid, sc)); continue
        etok=set(_tokens(id2text[eid]))
        overlap=len(qtok & etok)
        boost = 1.0 + 0.07*min(6, overlap)   # up to ~1.42x
        boosted.append((eid, sc*boost))
    boosted.sort(key=lambda x:-x[1])

    # MMR + CE
    shortlist=[eid for eid,_ in boosted[:max(k_final*4,60)]]
    id2vec=vs.get_vectors(shortlist)
    mmr=mmr_diversify(qvec, [(eid,s) for eid,s in boosted if eid in id2vec], id2vec, k=min(k_final*2,len(id2vec)))
    rer=ce_rerank(qtext, id2text, mmr, topn=min(CE_TOPN, len(mmr)))
    return rer[:k_final]

def spreading_activation(seed_ids: List[str], gs, depth:int=1, decay:float=0.85):
    if not seed_ids: return []
    act={sid:1.0 for sid in seed_ids}
    frontier=list(seed_ids)
    for _ in range(max(0, depth)):
        nxt=[]
        for nid in frontier:
            try:
                succ = gs.successors(nid)
            except Exception:
                succ = []
            for s in succ:
                score = act.get(nid,1.0) * decay
                if score > act.get(s, 0.0):
                    act[s] = score
                    nxt.append(s)
        frontier = nxt
        if not frontier: break
    return [k for k,_ in sorted(act.items(), key=lambda x:-x[1])]


'''# core/retrieval.py
from typing import List, Tuple, Dict, Optional
import math, time
import numpy as np

try:
    import config
    MMR_LAMBDA    = float(getattr(config, "MMR_LAMBDA", 0.35))
    USE_CE        = bool(getattr(config, "USE_CROSS_ENCODER", True))
    CE_MODEL_NAME = getattr(config, "CE_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
    CE_TOPN       = int(getattr(config, "CE_TOPN", 20))
    CANDIDATE_K   = int(getattr(config, "CANDIDATE_K", 40))
    FINAL_K       = int(getattr(config, "FINAL_K", 8))
    ALPHA_USAGE   = float(getattr(config, "ALPHA_USAGE", 0.20))
    BETA_TIME     = float(getattr(config, "BETA_TIME", 5e-6))
except Exception:
    MMR_LAMBDA = 0.35; USE_CE=True; CE_MODEL_NAME="cross-encoder/ms-marco-MiniLM-L-6-v2"
    CE_TOPN=20; CANDIDATE_K=40; FINAL_K=8; ALPHA_USAGE=0.20; BETA_TIME=5e-6

_CE = None
def _get_ce():
    global _CE
    if _CE is not None: return _CE
    if not USE_CE: return None
    try:
        from sentence_transformers import CrossEncoder
        _CE = CrossEncoder(CE_MODEL_NAME)
    except Exception:
        _CE = None
    return _CE

def _cos(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))

def rank_with_bias(raw: List[Tuple[str, float]], meta: Dict[str, Dict],
                   alpha: float=ALPHA_USAGE, beta: float=BETA_TIME):
    out=[]
    now=time.time()
    for eid, sim in raw:
        m = meta.get(eid, {})
        usage = float(m.get("usage_count",0))
        last  = float(m.get("last_accessed", now))
        age = max(1.0, now-last)
        rec = 1.0/age
        score = sim * (1.0 + alpha*math.log1p(usage)) * (1.0 + beta*rec*1e6)
        out.append((eid, score))
    out.sort(key=lambda x:-x[1])
    return out

def mmr_diversify(qvec, cand: List[Tuple[str,float]], id2vec: Dict[str,np.ndarray], k:int=8, lamb:float=MMR_LAMBDA):
    if not cand: return []
    cand = [(eid,s) for eid,s in cand if eid in id2vec]
    if not cand: return []
    chosen=[]
    chosen_ids=set()
    sims={eid:score for eid,score in cand}
    while len(chosen) < min(k,len(cand)):
        best=None; best_sc=-1e9
        for eid,_ in cand:
            if eid in chosen_ids: continue
            rel = sims.get(eid,0.0)
            red = 0.0
            if chosen:
                red = max(_cos(id2vec[eid], id2vec[cid]) for cid,_ in chosen)
            sc = lamb*rel - (1.0-lamb)*red
            if sc>best_sc:
                best_sc=sc; best=eid
        if best is None: break
        chosen.append((best, sims.get(best,0.0))); chosen_ids.add(best)
    return chosen

def _fetch_texts_for_ids(db, ids: List[str]) -> Dict[str,str]:
    out={}
    con=db.con
    for eid in ids:
        row=con.execute("SELECT id, summary FROM episodes WHERE id=?", (eid,)).fetchone()
        if row and row[0]:
            out[row[0]]=(row[1] or "").strip()
    return out

def ce_rerank(query_text: str, id2text: Dict[str,str], cand: List[Tuple[str,float]], topn:int=CE_TOPN):
    ce=_get_ce()
    if ce is None or not cand: return cand
    subset=cand[:min(topn,len(cand))]
    pairs=[]; keep=[]
    for eid,_ in subset:
        t=id2text.get(eid,"")
        if t:
            pairs.append((query_text,t)); keep.append(eid)
    if not pairs: return subset
    scores=ce.predict(pairs)
    rer=list(zip(keep,[float(s) for s in scores]))
    rer.sort(key=lambda x:-x[1])
    missing=[(eid,score) for eid,score in subset if eid not in keep]
    return rer+missing

def expand_candidates(qvec: np.ndarray, qtext: str, vs, db,
                      k_candidates:int=CANDIDATE_K, k_final:int=FINAL_K):
    raw = vs.search(qvec, k=k_candidates)  # [(eid,sim)]
    if not raw: return []
    ids=[eid for eid,_ in raw]
    meta={}
    con=db.con
    for eid in ids:
        m=con.execute("SELECT usage_count,last_accessed FROM episodes WHERE id=?", (eid,)).fetchone()
        if m: meta[eid]={"usage_count": m[0], "last_accessed": m[1]}
    biased=rank_with_bias(raw, meta)
    shortlist=[eid for eid,_ in biased[:max(k_final*3,30)]]
    id2vec=vs.get_vectors(shortlist)
    id2text=_fetch_texts_for_ids(db, shortlist)
    mmr=mmr_diversify(qvec, [(eid,s) for eid,s in biased if eid in id2vec], id2vec, k=min(k_final*2,len(id2vec)))
    rer=ce_rerank(qtext, id2text, mmr, topn=min(CE_TOPN, len(mmr)))
    return rer[:k_final]

def spreading_activation(seed_ids: List[str], gs, depth:int=1, decay:float=0.85):
    seen=set(seed_ids); frontier=list(seed_ids)
    for _ in range(max(0,depth)):
        nxt=[]
        for nid in frontier:
            try: succ=gs.successors(nid)
            except Exception: succ=[]
            for s in succ:
                if s not in seen:
                    seen.add(s); nxt.append(s)
        frontier=nxt
        if not frontier: break
    return list(seen)
'''