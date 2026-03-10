from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple
import math

def min_max_norm(scores:Dict[str,float])-> Dict[str,float]:
    if not scores:
        return {}

    vals=list(scores.values())
    lo, hi=min(vals), max(vals)
    if hi - lo <1e-12:
        return {k: 1.0 for k in scores.keys()}
    return {k:(v-lo)/(hi-lo) for k,v in scores.items() }

def blend_scores(
        dense_scores:Dict[str,float],
        bm25_scores:Dict[str,float],
        *,
        alpha:float
)->Dict[str,float]:

    dn=min_max_norm(dense_scores)
    bn=min_max_norm(bm25_scores)

    all_ids=set(dn.keys()) | set(bn.keys())
    out:Dict[str,float]={}
    
    for cid in all_ids:
        out[cid]= alpha * dn.get(cid,0.0) + (1.0 - alpha)*bn.get(cid,0.0)
    
    return out 