from __future__ import annotations

import re
from typing import Dict, List, Optional, Sequence, Tuple

from rag.utils.contracts import EvidenceItem, ParsedAnswer

# Accept [C1] or [C1, C2] with optional spaces
_CITATION_BRACKET_RE = re.compile(r"\[([^\]]+)\]\s*$")
_CITATION_TAG_RE = re.compile(r"\bC\d+\b")

def _extract_bullets(text: str) -> List[str]:
    lines = [ln.rstrip() for ln in (text or "").splitlines()]
    bullets: List[str] = []
    for ln in lines:
        if ln.startswith("- "):
            bullets.append(ln[2:].strip())
    return bullets

def _parse_citations_from_bullet(bullet: str) -> Tuple[str, List[str], Optional[str]]:
    """
    Returns: (bullet_without_citations, tags, warning)
    tags are like ["C1", "C2"] (no brackets).
    """
    m = _CITATION_BRACKET_RE.search(bullet)
    if not m:
        return bullet, [], "missing_citation_brackets"
    inside = m.group(1)
    tags = _CITATION_TAG_RE.findall(inside)
    if not tags:
        return bullet[: m.start()].rstrip(), [], "no_valid_citation_tags"
    # remove trailing bracket section from bullet
    clean = bullet[: m.start()].rstrip()
    return clean, tags, None

def _parse_abstain(text: str) -> Tuple[Optional[str], List[str]]:
    """
    Expected:
      ABSTAIN: ...
      NEED: (1) ... (2) ...
    """
    reason = None
    needs: List[str] = []
    lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
    for ln in lines:
        if ln.upper().startswith("ABSTAIN:"):
            reason = ln.split(":", 1)[1].strip() if ":" in ln else ln.strip()
        elif ln.upper().startswith("NEED:"):
            rest = ln.split(":", 1)[1].strip() if ":" in ln else ""
            # naive split on (n)
            parts = re.split(r"\(\d+\)", rest)
            needs = [p.strip(" .;-") for p in parts if p.strip()]
    return reason, needs

def parse_model_output(
    raw_text: str,
    evidence_items: Sequence[EvidenceItem],
    *,
    min_bullets: int = 2,
    max_bullets: int = 5,
) -> ParsedAnswer:
    parse_warnings: List[str] = []
    tag_to_chunk_id: Dict[str, str] = {}
    allowed_tags = set()

    for it in evidence_items:
        # it.citation_tag like "[C1]" -> normalize to "C1"
        tag = it.citation_tag.strip()[1:-1] if it.citation_tag.startswith("[") else it.citation_tag
        allowed_tags.add(tag)
        # Prefer the RetrievedChunk's chunk_id via citation
        chunk_id = getattr(it.citation, "chunk_id", None) or getattr(it.citation, "chunk_id", "")
        tag_to_chunk_id[tag] = chunk_id

    text = (raw_text or "").strip()

    # Abstain mode check first
    if text.upper().startswith("ABSTAIN:"):
        reason, needs = _parse_abstain(text)
        return ParsedAnswer(
            mode="abstain",
            bullets=[],
            citation_by_bullet=[],
            resolved_chunk_ids_by_bullet=[],
            abstain_reason=reason,
            needs=needs,
            raw_text=raw_text,
            parse_warning=[],
        )

    bullets_raw = _extract_bullets(text)
    if not bullets_raw:
        return ParsedAnswer(
            mode="answer",
            bullets=[],
            citation_by_bullet=[],
            resolved_chunk_ids_by_bullet=[],
            abstain_reason=None,
            needs=[],
            raw_text=raw_text,
            parse_warning=["no_bullets_found"],
        )

    if not (min_bullets <= len(bullets_raw) <= max_bullets):
        parse_warnings.append(f"bullet_count_out_of_range:{len(bullets_raw)}")

    bullets: List[str] = []
    citations_by_bullet: List[List[str]] = []
    resolved_chunk_ids_by_bullet: List[List[str]] = []

    for b in bullets_raw:
        clean, tags, warn = _parse_citations_from_bullet(b)
        if warn:
            parse_warnings.append(warn)
        # validate tags against allowed
        valid_tags = [t for t in tags if t in allowed_tags]
        invalid_tags = [t for t in tags if t not in allowed_tags]
        if invalid_tags:
            parse_warnings.append(f"invalid_tags:{','.join(invalid_tags)}")
        if not valid_tags:
            parse_warnings.append("no_valid_tags_after_validation")

        bullets.append(clean)
        citations_by_bullet.append(valid_tags)
        resolved_chunk_ids_by_bullet.append([tag_to_chunk_id[t] for t in valid_tags if tag_to_chunk_id.get(t)])

    return ParsedAnswer(
        mode="answer",
        bullets=bullets,
        citation_by_bullet=citations_by_bullet,
        resolved_chunk_ids_by_bullet=resolved_chunk_ids_by_bullet,
        abstain_reason=None,
        needs=[],
        raw_text=raw_text,
        parse_warning=parse_warnings,
    )
