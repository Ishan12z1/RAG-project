from __future__ import annotations

import re
from typing import Dict, List, Optional, Sequence, Tuple

from rag.utils.contracts import EvidenceItem, ParsedAnswer


# Accept trailing citation block like:
#   - some claim [C1]
#   - some claim [C1, C2]
# Spaces optional around commas / before bracket.
_CITATION_BRACKET_RE = re.compile(r"\[([^\]]+)\]\s*$")
_CITATION_TAG_RE = re.compile(r"\bC\d+\b", flags=re.IGNORECASE)

# More tolerant abstain detection
_ABSTAIN_START_RE = re.compile(r"^\s*ABSTAIN\s*:", flags=re.IGNORECASE)
_NEED_LINE_RE = re.compile(r"^\s*NEED\s*:\s*(.*)$", flags=re.IGNORECASE)


def _extract_bullets(text: str) -> List[str]:
    """
    Extract bullets from lines beginning with '- '.
    """
    lines = [ln.rstrip() for ln in (text or "").splitlines()]
    bullets: List[str] = []

    for ln in lines:
        stripped = ln.lstrip()
        if stripped.startswith("- "):
            bullets.append(stripped[2:].strip())

    return bullets


def _parse_citations_from_bullet(bullet: str) -> Tuple[str, List[str], Optional[str]]:
    """
    Returns:
        (bullet_without_citations, tags, warning)

    tags are normalized like ["C1", "C2"] with no brackets.
    """
    m = _CITATION_BRACKET_RE.search(bullet)
    if not m:
        return bullet.strip(), [], "missing_citation_brackets"

    inside = m.group(1)
    tags = [t.upper() for t in _CITATION_TAG_RE.findall(inside)]

    if not tags:
        clean = bullet[: m.start()].rstrip()
        return clean, [], "no_valid_citation_tags"

    clean = bullet[: m.start()].rstrip()
    return clean, tags, None


def _parse_needs_text(needs_text: str) -> List[str]:
    """
    Parses NEED content in formats like:
      NEED: (1) age (2) meds
      NEED: 1. age 2. meds
      NEED: age; meds
      NEED: age, meds

    Keeps this reasonably tolerant without becoming too magical.
    """
    text = (needs_text or "").strip()
    if not text:
        return []

    # Try numbered patterns first: (1) ..., (2) ...  OR 1. ..., 2. ...
    numbered_matches = re.findall(
        r"(?:\(\d+\)|\b\d+\.)\s*([^()]+?)(?=(?:\(\d+\)|\b\d+\.|$))",
        text,
        flags=re.IGNORECASE,
    )
    if numbered_matches:
        return [m.strip(" .;-") for m in numbered_matches if m.strip(" .;-")]

    # Fallback: split on semicolons first, then commas if needed.
    if ";" in text:
        parts = [p.strip(" .;-") for p in text.split(";")]
        return [p for p in parts if p]

    if "," in text:
        parts = [p.strip(" .;-") for p in text.split(",")]
        return [p for p in parts if p]

    return [text.strip(" .;-")] if text.strip(" .;-") else []


def _parse_abstain(text: str) -> Tuple[Optional[str], List[str], List[str]]:
    """
    Expected main patterns:
      ABSTAIN: ...
      NEED: ...

    Returns:
        (reason, needs, warnings)
    """
    reason: Optional[str] = None
    needs: List[str] = []
    warnings: List[str] = []

    lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
    need_lines: List[str] = []

    for ln in lines:
        if _ABSTAIN_START_RE.match(ln):
            reason = re.split(r"^\s*ABSTAIN\s*:\s*", ln, maxsplit=1, flags=re.IGNORECASE)[1].strip()
        else:
            m = _NEED_LINE_RE.match(ln)
            if m:
                need_lines.append(m.group(1).strip())

    if reason is None:
        warnings.append("abstain_reason_missing")

    if need_lines:
        combined_needs = " ".join(need_lines).strip()
        needs = _parse_needs_text(combined_needs)

    return reason, needs, warnings


def _normalize_citation_tag(raw_tag: str) -> str:
    raw = (raw_tag or "").strip()
    if raw.startswith("[") and raw.endswith("]") and len(raw) >= 3:
        raw = raw[1:-1].strip()
    return raw.upper()


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
        tag = _normalize_citation_tag(getattr(it, "citation_tag", ""))
        if not tag:
            continue

        allowed_tags.add(tag)

        citation_obj = getattr(it, "citation", None)
        chunk_id = getattr(citation_obj, "chunk_id", "") if citation_obj is not None else ""
        tag_to_chunk_id[tag] = chunk_id

    text = (raw_text or "").strip()

    if not text:
        return ParsedAnswer(
            mode="parse_error",
            bullets=[],
            citation_by_bullets=[],
            resolved_chunk_ids_by_bullet=[],
            abstain_reason=None,
            needs=[],
            raw_text=raw_text,
            parse_warnings=["empty_model_output"],
        )

    # Abstain mode first
    if _ABSTAIN_START_RE.match(text):
        reason, needs, abstain_warnings = _parse_abstain(text)
        return ParsedAnswer(
            mode="abstain",
            bullets=[],
            citation_by_bullets=[],
            resolved_chunk_ids_by_bullet=[],
            abstain_reason=reason,
            needs=needs,
            raw_text=raw_text,
            parse_warnings=abstain_warnings,
        )

    bullets_raw = _extract_bullets(text)
    if not bullets_raw:
        return ParsedAnswer(
            mode="parse_error",
            bullets=[],
            citation_by_bullets=[],
            resolved_chunk_ids_by_bullet=[],
            abstain_reason=None,
            needs=[],
            raw_text=raw_text,
            parse_warnings=["no_bullets_found"],
        )

    if not (min_bullets <= len(bullets_raw) <= max_bullets):
        parse_warnings.append(f"bullet_count_out_of_range:{len(bullets_raw)}")

    bullets: List[str] = []
    citation_by_bullets: List[List[str]] = []
    resolved_chunk_ids_by_bullet: List[List[str]] = []

    for idx, bullet in enumerate(bullets_raw, start=1):
        clean_bullet, tags, warn = _parse_citations_from_bullet(bullet)

        if warn:
            parse_warnings.append(f"bullet_{idx}_{warn}")

        valid_tags: List[str] = []
        invalid_tags: List[str] = []

        for tag in tags:
            if tag in allowed_tags:
                valid_tags.append(tag)
            else:
                invalid_tags.append(tag)

        if invalid_tags:
            parse_warnings.append(f"bullet_{idx}_invalid_tags:{','.join(invalid_tags)}")

        if not valid_tags:
            parse_warnings.append(f"bullet_{idx}_no_valid_tags_after_validation")

        resolved_chunk_ids = [
            tag_to_chunk_id[tag]
            for tag in valid_tags
            if tag_to_chunk_id.get(tag)
        ]

        if valid_tags and not resolved_chunk_ids:
            parse_warnings.append(f"bullet_{idx}_valid_tags_but_no_resolved_chunk_ids")

        bullets.append(clean_bullet)
        citation_by_bullets.append(valid_tags)
        resolved_chunk_ids_by_bullet.append(resolved_chunk_ids)

    return ParsedAnswer(
        mode="answer",
        bullets=bullets,
        citation_by_bullets=citation_by_bullets,
        resolved_chunk_ids_by_bullet=resolved_chunk_ids_by_bullet,
        abstain_reason=None,
        needs=[],
        raw_text=raw_text,
        parse_warnings=parse_warnings,
    )