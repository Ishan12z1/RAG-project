from __future__ import annotations
from typing import Set, Sequence, Dict, Any
from rag.utils.contracts import RetrievedChunk,AbstainConfig,AbstainDecision
import re

# Minimal stopword set what should not count as keywords
_STOPWORDS: Set[str] = {
    "a","an","the","and","or","but","if","then","else","when","while",
    "is","are","was","were","be","been","being",
    "to","of","in","on","for","from","with","as","at","by","into","about",
    "it","this","that","these","those",
    "i","you","we","they","he","she","them","his","her","our","your",
    "what","why","how","who","whom","which",
    "do","does","did","doing",
    "can","could","should","would","may","might","must",
}

_TOKEN_RE = re.compile(r"[a-z0-9]+")

def _tokenize(text):
    return _TOKEN_RE.findall(text.lower())

def _keywords(query):
    toks=[t for t in _tokenize(query) if t not in _STOPWORDS]
    toks=[t for t in toks if len(t)>=3]
    return toks

def _overlap_fraction(query_keywords, context_text):
    ctx=set(_tokenize(context_text))
    hits=sum(1 for t in query_keywords if t in ctx)
    return hits/max(1,len(query_keywords))


def should_abstain(
        query:str,
        retrieved:Sequence[RetrievedChunk],
        cfg:AbstainConfig=AbstainConfig(),
)->AbstainDecision:
    reasons=[]
    signals:Dict[str,Any]={}

    if not retrieved:
        return AbstainDecision(
            abstain=True,
            reasons=["no_retrieved_chunks"],
            signals={"top1": None, "topk": None, "gap": None, "overlap": 0.0},
        )
    
    chunks=sorted(retrieved,key=lambda c:c.score,reverse=True)

    # Calcuating the score diff between top1 and gap_k query (or last query if len(chunks)<gap_k)
    top1= chunks[0].score
    k_indx=min(max(cfg.gap_k,1)-1,len(chunks)-1)
    topk=chunks[k_indx].score
    gap=top1-topk
    signals.update({"top1": top1, "topk": topk, "gap": gap, "gap_k_used": k_indx + 1})

    if top1 < cfg.min_top1:
        reasons.append("top1_below_threshold")

    if gap < cfg.min_gap:
        reasons.append("score_gap_too_small")
    
    # Calcuate overlapping words between query and context
    qk=_keywords(query)
    context="/n".join(c.text for c in chunks)
    context=context[:cfg.max_context_chars]
    overlap=_overlap_fraction(query,context)

    signals.update({
    "query_keywords": qk[:50],
    "overlap": overlap,
    "num_keywords": len(qk),
    })
    if len(qk) >= 2 and overlap < cfg.min_overlap:
        reasons.append("low_query_context_overlap")

    abstain = len(reasons) > 0
    return AbstainDecision(abstain=abstain, reasons=reasons, signals=signals)