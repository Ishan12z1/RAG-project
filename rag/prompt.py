from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

from rag.retrieval.retrieve import RetrievedChunk
from rag.utils.contracts import EvidenceItem


CITATION_TAG_PREFIX = "C"
PROMPT_VERSION = "v3"


def assign_citation_tags(chunks: Sequence[RetrievedChunk]) -> List[EvidenceItem]:
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


def _truncate(text: str, max_chars: int) -> str:
    if max_chars <= 0:
        return ""
    t = (text or "").strip()
    if len(t) <= max_chars:
        return t
    return t[: max_chars - 1].rstrip() + "..."


def build_evidence_block(
    chunks: Sequence[RetrievedChunk],
    *,
    max_chunks: int = 8,
    max_chars_per_chunk: int = 1200,
) -> Tuple[str, List[EvidenceItem]]:
    use = list(chunks)[:max_chunks]
    items = assign_citation_tags(use)

    lines: List[str] = []
    for it in items:
        c = it.citation
        header = (
            f"{it.citation_tag} "
            f"doc_id={getattr(c, 'doc_id', 'unknown')} | "
            f"title={getattr(c, 'title', 'unknown')} | "
            f"section={getattr(c, 'section', 'unknown')} | "
            f"chunk_id={getattr(c, 'chunk_id', 'unknown')} | "
            f"source={getattr(c, 'source', 'unknown')} | "
            f"url={getattr(c, 'url', 'none')}"
        )
        lines.append(header)
        lines.append(_truncate(it.text, max_chars_per_chunk))
        lines.append("")

    return "\n".join(lines).rstrip() + "\n", items


SYSTEM_GROUNDED_QA = (
    "You are a grounded conversational assistant.\n"
    "You must follow these rules:\n"
    "1) Use ONLY the EVIDENCE provided. Do not use outside knowledge.\n"
    "2) Be helpful and conversational, but stay strictly grounded in the evidence.\n"
    "3) If the question is unrelated to the available evidence or the evidence is too weak, abstain.\n"
    "4) Every answer bullet MUST end with citations using the provided evidence tags, like [C1] or [C1, C2].\n"
    "5) Do not cite anything that is not in the EVIDENCE block.\n"
)


USER_TEMPLATE = (
    "RECENT CONVERSATION:\n{conversation}\n\n"
    "QUESTION:\n{query}\n\n"
    "EVIDENCE:\n"
    "{evidence}\n\n"
    "OUTPUT FORMAT (choose exactly one):\n"
    "A) Answer (2-4 short bullets in a natural chat tone):\n"
    "- <helpful grounded point>. [C1]\n"
    "- <helpful grounded point>. [C1, C2]\n\n"
    "B) Abstain:\n"
    "ABSTAIN: I can't answer that from the available source material.\n"
    "NEED: (1) ... (2) ... (3) ...\n"
)


def build_conversation_context(
    history: Sequence[dict[str, str]] | None,
    *,
    max_turns: int = 4,
    max_chars_per_turn: int = 240,
) -> str:
    if not history:
        return "None."

    recent_turns = list(history)[-max_turns:]
    lines: List[str] = []
    for turn in recent_turns:
        role = str(turn.get("role", "user")).strip().lower()
        speaker = "User" if role == "user" else "Assistant"
        content = _truncate(str(turn.get("content", "")), max_chars_per_turn)
        if content:
            lines.append(f"{speaker}: {content}")
    return "\n".join(lines) if lines else "None."


def build_contextual_query(
    question: str,
    history: Sequence[dict[str, str]] | None,
    *,
    max_turns: int = 2,
    max_chars_per_turn: int = 120,
) -> str:
    if not history:
        return question.strip()

    recent_turns = list(history)[-max_turns:]
    history_parts: List[str] = []
    for turn in recent_turns:
        content = _truncate(str(turn.get("content", "")), max_chars_per_turn)
        if content:
            history_parts.append(content)

    if not history_parts:
        return question.strip()
    return " ".join(history_parts + [question.strip()])


def build_prompt(question: str, evidence_block: str, conversation_context: str | None = None) -> Dict[str, str]:
    user = USER_TEMPLATE.format(
        conversation=(conversation_context or "None.").strip(),
        query=question.strip(),
        evidence=evidence_block,
    )
    return {"system": SYSTEM_GROUNDED_QA, "user": user}
