from __future__ import annotations


def truncate_text(text:str, max_chars:int) ->str:
    if max_chars <=0:
        return ""
    if len(text)<=max_chars:
        return text
    return text[:max_chars]

def batch_iter(xs:list,batch_size:int):
    if batch_size <=0:
        raise ValueError("batch size must be > 0")
    for i in range(0,len(xs),batch_size):
        yield xs[i:i+batch_size]

def min_max(xs: list[float]) -> list[float]:
    if not xs:
        return xs
    lo, hi = min(xs), max(xs)
    if hi - lo < 1e-12:
        return [1.0 for _ in xs]
    return [(x - lo) / (hi - lo) for x in xs]
