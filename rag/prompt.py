from __future__ import annotations

from dataclasses import dataclass
from typing import Any,Dict,Sequence,Tuple, List
from rag.retrieval.retrieve import RetrievedChunk, Citation


CITATION_TAG_PREFIX = "C"  # [C1], [C2], ...

@dataclass(frozen=True)
class EvidenceItem:
    citation_tag:str
    citation:Any
    text:str
    score:float
    metadata:Dict[str,Any]


# we are converting each chunk into a Evidence item format 
def assign_citation_tags(chunks: Sequence[RetrievedChunk]) -> List[EvidenceItem]:
    """
    Deterministically assigns [C1]..[Ck] in the given order.
    Caller should already ensure 'chunks' are sorted deterministically.
    """
    items: List[EvidenceItem] = []
    for i, ch in enumerate(chunks, start=1):
        tag = f"[{CITATION_TAG_PREFIX}{i}]"
        items.append(
            EvidenceItem(
                citation_tag=tag,
                citation=ch.citation,
                text=ch.text,
                score=float(ch.score),
                metadata=dict(ch.metadata) if ch.metadata is not None else {},
            )
        )
        if i==1:
            print(ch.metadata)
    return items

def _truncate(text:str,max_chars:int)->str:
    if max_chars <=0:
        return ""
    t=(text or "").strip()
    if len(t) < max_chars:
        return t
    return t[:max_chars-1].rstrip()+"..."

def build_evidence_block(
        chunks:Sequence[RetrievedChunk],
        *,
        max_chunks:int=8,
        max_chars_per_chunk:int=1200
    )-> Tuple[str,List[EvidenceItem]]:
    
    use=list(chunks)[:max_chunks] # safe even if len(chunks)< max_chunks 
    items=assign_citation_tags(use)

    lines:List[str]=[]
    lines.append("EVIDENCE (use only this evidence for factual claims):")
    for it in items:
        c= it.citation
        header=(
            f"{it.citation_tag} "
            f"doc_id={getattr(c, 'doc_id', '')} | "
            f"title={getattr(c, 'title', '')} | "
            f"section={getattr(c, 'section', '')} | "
            f"chunk_id={getattr(c, 'chunk_id', '')} | "
            f"source={getattr(c, 'source', '')} | "
            f"url={getattr(c, 'url', '')}"
        )
        lines.append(header)
        lines.append(_truncate(it.text,max_chars_per_chunk))
        lines.append("") # blank line betwee evidence items 

    
    return "\n".join(lines).rstrip() +"\n", items


SYSTEM_GROUNDED_QA = """You are a grounded assistant.
Use ONLY the EVIDENCE provided. Do not use outside knowledge.
If the evidence is insufficient, say you don’t have enough information.

Citation rules:
- Every factual claim must cite at least one evidence tag like [C1].
- If a sentence contains multiple claims, cite all relevant tags.
- Do NOT invent citations. Only use tags that appear in the evidence block.
- Do NOT cite if you are stating a limitation (e.g., "the evidence doesn't say").
"""

USER_TEMPLATE = """Question:
{question}

{evidence_block}

Answer:
"""


def build_prompt(question: str, chunks: Sequence[RetrievedChunk]) -> Dict[str, str]:
    """
    Returns a dict suitable for chat APIs:
      {"system": ..., "user": ...}

    Keep this function pure/deterministic: same inputs -> same prompt.
    """
    evidence_block, _items = build_evidence_block(chunks)
    user = USER_TEMPLATE.format(question=question.strip(), evidence_block=evidence_block)
    return {"system": SYSTEM_GROUNDED_QA, "user": user}