from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

from rag.retrieval.retrieve import RetrievedChunk
from rag.utils.contracts import EvidenceItem

CITATION_TAG_PREFIX = "C"
PROMPT_VERSION = "v5-structured-json"


def assign_citation_tags(chunks: Sequence[RetrievedChunk]) -> List[EvidenceItem]:
    items: List[EvidenceItem] = []
    for i, ch in enumerate(chunks, start=1):
        tag = f"{CITATION_TAG_PREFIX}{i}"
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
    max_chunks: int = 5,
    max_chars_per_chunk: int = 900,
) -> Tuple[str, List[EvidenceItem]]:
    use = list(chunks)[:max_chunks]
    items = assign_citation_tags(use)

    lines: List[str] = []
    for it in items:
        c = it.citation
        header = (
            f"{it.citation_tag} "
            f"title={getattr(c, 'title', 'unknown')} | "
            f"section={getattr(c, 'section', 'unknown')} | "
            f"url={getattr(c, 'url', 'none')}"
        )
        lines.append(header)
        lines.append(_truncate(it.text, max_chars_per_chunk))
        lines.append("")

    return "\n".join(lines).rstrip() + "\n", items


SYSTEM_GROUNDED_QA = (
    "You are a grounded assistant.\n"
    "Use only the provided evidence.\n"
    "Return only valid JSON with no markdown fences and no extra text.\n"
    "The JSON must match exactly one of these shapes.\n"
    'Answer mode: {"mode":"answer","segments":[{"text":"...","citations":["C1","C2"]}]}\n'
    'Abstain mode: {"mode":"abstain","needs":["...","..."]}\n'
    "Use the recent conversation summary only to resolve follow-up context.\n"
    "Every answer segment must be supported by the provided evidence tags.\n"
    "Use only citation tags that appear in the evidence pack.\n"
    "Do not put citation tags inside segment text.\n"
    "Do not output bullets, headings, labels, or free-form sections.\n"
    "If the evidence is insufficient or the request is unsupported by the evidence, choose abstain mode.\n"
)

USER_TEMPLATE = (
    "CONVERSATION SUMMARY:\n{conversation}\n\n"
    "USER QUESTION:\n{query}\n\n"
    "EVIDENCE PACK:\n{evidence}\n\n"
    "Return strict JSON only.\n"
    'Every segment object in answer mode must include both "text" and "citations".\n'
    'Example valid answer JSON: {{"mode":"answer","segments":[{{"text":"...","citations":["C1"]}}]}}\n'
    "For answer mode, write 1-2 short paragraph-like segments in natural chat language.\n"
    "Keep the full answer concise, about 2-4 sentences total.\n"
    "For abstain mode, return 1-3 concrete missing-information needs.\n"
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


def build_repair_prompt(
    *,
    question: str,
    conversation_context: str,
    evidence_block: str,
    previous_output: str,
    parse_warnings: Sequence[str],
) -> Dict[str, str]:
    warning_text = "\n".join(f"- {warning}" for warning in parse_warnings) if parse_warnings else "- unknown_schema_error"
    user = (
        "Your previous response was invalid for the required JSON schema.\n\n"
        f"CONVERSATION SUMMARY:\n{(conversation_context or 'None.').strip()}\n\n"
        f"USER QUESTION:\n{question.strip()}\n\n"
        f"EVIDENCE PACK:\n{evidence_block}\n\n"
        f"PREVIOUS INVALID OUTPUT:\n{previous_output.strip()}\n\n"
        f"VALIDATION ERRORS:\n{warning_text}\n\n"
        "Repair the response.\n"
        "Return strict JSON only.\n"
        'If mode is "answer", every segment must include both "text" and "citations".\n'
        'Use only evidence tags from the evidence pack, such as "C1".\n'
        'Do not place citation tags inside the visible text.\n'
        'If you cannot support the answer with citations, return {"mode":"abstain","needs":[...]}.'
    )
    return {"system": SYSTEM_GROUNDED_QA, "user": user}
