from __future__ import annotations

# Prompt version v1
from dataclasses import dataclass
from typing import Any,Dict,Sequence,Tuple, List
from rag.retrieval.retrieve import RetrievedChunk, Citation
from rag.utils.contracts import EvidenceItem

CITATION_TAG_PREFIX = "C"  # [C1], [C2], ...
PROMPT_VERSION="v2"


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
    for it in items:
        c= it.citation
        header=(
            f"{it.citation_tag} "
            f"doc_id={getattr(c, 'doc_id', 'unknown')} | "
            f"title={getattr(c, 'title', 'unknown')} | "
            f"section={getattr(c, 'section', 'unknown')} | "
            f"chunk_id={getattr(c, 'chunk_id', 'unknown')} | "
            f"source={getattr(c, 'source', 'unknown')} | "
            f"url={getattr(c, 'url', 'none')}"
        )
        lines.append(header)
        lines.append(_truncate(it.text,max_chars_per_chunk))
        lines.append("") # blank line betwee evidence items 

    
    return "\n".join(lines).rstrip() +"\n", items


SYSTEM_GROUNDED_QA = (
    "You are a grounded medical QA assistant.\n"
    "You must follow these rules:\n"
    "1) Use ONLY the EVIDENCE provided. Do not use outside knowledge.\n"
    "2) Every bullet MUST end with citations in square brackets using the evidence tags, "
    "like [C1] or [C1, C2].\n"
    "3) If the EVIDENCE is insufficient to answer, respond with:\n"
    "   ABSTAIN: <one sentence>\n"
    "   NEED: <up to 3 clarifying items>\n"
    "4) Do not cite sources not in EVIDENCE.\n"
    # f"PromptVersion: {PROMPT_VERSION}\n"
)

USER_TEMPLATE = (
    "QUESTION:\n{query}\n\n"
    "EVIDENCE:\n"
    "{evidence}\n\n"
    "OUTPUT FORMAT (choose exactly one):\n"
    "A) Answer (2–5 bullets):\n"
    "- <claim>. [C1]\n"
    "- <claim>. [C1, C2]\n\n"
    "B) Abstain:\n"
    "ABSTAIN: <one sentence>\n"
    "NEED: (1) ... (2) ... (3) ...\n"
)


def build_prompt(question: str, evidence_block: str) -> Dict[str, str]:
    """
    Returns a dict suitable for chat APIs:
      {"system": ..., "user": ...}

    Keep this function pure/deterministic: same inputs -> same prompt.
    """
    user = USER_TEMPLATE.format(query=question.strip(), evidence=evidence_block)
    return {"system": SYSTEM_GROUNDED_QA, "user": user}

