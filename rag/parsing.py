from __future__ import annotations

import json
import re
from typing import Dict, List, Sequence, Tuple

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from rag.utils.contracts import AnswerSegment, EvidenceItem, ParsedAnswer

_VISIBLE_CITATION_RE = re.compile(r"\[(C\d+(?:\s*,\s*C\d+)*)\]", flags=re.IGNORECASE)


class _AnswerSegmentSchema(BaseModel):
    text: str = Field(min_length=1)
    citations: List[str] = Field(min_length=1)

    model_config = ConfigDict(extra="forbid")


class _StructuredOutputSchema(BaseModel):
    mode: str
    segments: List[_AnswerSegmentSchema] = Field(default_factory=list)
    needs: List[str] = Field(default_factory=list)

    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="after")
    def validate_mode_shape(self) -> "_StructuredOutputSchema":
        if self.mode == "answer":
            if not self.segments:
                raise ValueError("answer mode requires segments")
            if self.needs:
                raise ValueError("answer mode cannot include needs")
            return self
        if self.mode == "abstain":
            if self.segments:
                raise ValueError("abstain mode cannot include segments")
            if not self.needs:
                raise ValueError("abstain mode requires needs")
            return self
        raise ValueError("mode must be answer or abstain")


def _normalize_citation_tag(raw_tag: str) -> str:
    raw = (raw_tag or "").strip()
    if raw.startswith("[") and raw.endswith("]") and len(raw) >= 3:
        raw = raw[1:-1].strip()
    return raw.upper()


def _strip_visible_citations(text: str) -> Tuple[str, bool]:
    cleaned = _VISIBLE_CITATION_RE.sub("", text or "")
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned, cleaned != (text or "").strip()


def _extract_json_text(raw_text: str) -> Tuple[str, List[str]]:
    text = (raw_text or "").strip()
    warnings: List[str] = []
    if not text:
        return "", warnings

    if text.startswith("```"):
        lines = text.splitlines()
        if len(lines) >= 3 and lines[-1].strip() == "```":
            text = "\n".join(lines[1:-1]).strip()
            warnings.append("json_wrapped_in_code_fence")

    start = text.find("{")
    end = text.rfind("}")
    if start > 0 or end != len(text) - 1:
        if start >= 0 and end > start:
            text = text[start : end + 1]
            warnings.append("json_extracted_from_surrounding_text")

    return text, warnings


def _build_tag_maps(evidence_items: Sequence[EvidenceItem]) -> Tuple[set[str], Dict[str, str]]:
    allowed_tags: set[str] = set()
    tag_to_chunk_id: Dict[str, str] = {}

    for it in evidence_items:
        tag = _normalize_citation_tag(getattr(it, "citation_tag", ""))
        if not tag:
            continue
        allowed_tags.add(tag)
        citation_obj = getattr(it, "citation", None)
        chunk_id = getattr(citation_obj, "chunk_id", "") if citation_obj is not None else ""
        tag_to_chunk_id[tag] = chunk_id

    return allowed_tags, tag_to_chunk_id


def _parse_structured_json(raw_text: str) -> Tuple[_StructuredOutputSchema | None, List[str]]:
    warnings: List[str] = []
    json_text, extraction_warnings = _extract_json_text(raw_text)
    warnings.extend(extraction_warnings)

    if not json_text:
        warnings.append("empty_model_output")
        return None, warnings

    try:
        payload = json.loads(json_text)
    except json.JSONDecodeError as exc:
        warnings.append(f"json_decode_error:{exc.msg}")
        return None, warnings

    try:
        return _StructuredOutputSchema.model_validate(payload), warnings
    except ValidationError as exc:
        for err in exc.errors():
            location = ".".join(str(part) for part in err.get("loc", ())) or "root"
            warnings.append(f"schema_validation_error:{location}:{err.get('msg', 'invalid')}")
        return None, warnings


def parse_model_output(raw_text: str, evidence_items: Sequence[EvidenceItem]) -> ParsedAnswer:
    structured, warnings = _parse_structured_json(raw_text)
    if structured is None:
        return ParsedAnswer(
            mode="parse_error",
            segments=[],
            needs=[],
            raw_text=raw_text,
            parse_warnings=warnings,
            schema_valid=False,
        )

    allowed_tags, tag_to_chunk_id = _build_tag_maps(evidence_items)

    if structured.mode == "abstain":
        cleaned_needs = [need.strip() for need in structured.needs if need and need.strip()]
        return ParsedAnswer(
            mode="abstain",
            segments=[],
            needs=cleaned_needs,
            raw_text=raw_text,
            parse_warnings=warnings,
            schema_valid=True,
        )

    parse_warnings = list(warnings)
    segments: List[AnswerSegment] = []

    for idx, item in enumerate(structured.segments, start=1):
        normalized_tags = [_normalize_citation_tag(tag) for tag in item.citations]
        invalid_tags = [tag for tag in normalized_tags if tag not in allowed_tags]
        if invalid_tags:
            parse_warnings.append(f"segment_{idx}_invalid_tags:{','.join(invalid_tags)}")
            return ParsedAnswer(
                mode="parse_error",
                segments=[],
                needs=[],
                raw_text=raw_text,
                parse_warnings=parse_warnings,
                schema_valid=False,
            )

        deduped_tags: List[str] = []
        for tag in normalized_tags:
            if tag not in deduped_tags:
                deduped_tags.append(tag)

        resolved_chunk_ids = [tag_to_chunk_id[tag] for tag in deduped_tags if tag_to_chunk_id.get(tag)]
        if len(resolved_chunk_ids) != len(deduped_tags):
            parse_warnings.append(f"segment_{idx}_unresolved_citation_tag")
            return ParsedAnswer(
                mode="parse_error",
                segments=[],
                needs=[],
                raw_text=raw_text,
                parse_warnings=parse_warnings,
                schema_valid=False,
            )

        clean_text, removed_visible_tags = _strip_visible_citations(item.text)
        if removed_visible_tags:
            parse_warnings.append(f"segment_{idx}_visible_citation_tags_removed")
        if not clean_text:
            parse_warnings.append(f"segment_{idx}_text_empty_after_cleanup")
            return ParsedAnswer(
                mode="parse_error",
                segments=[],
                needs=[],
                raw_text=raw_text,
                parse_warnings=parse_warnings,
                schema_valid=False,
            )

        segments.append(
            AnswerSegment(
                text=clean_text,
                citations=deduped_tags,
                resolved_chunk_ids=resolved_chunk_ids,
            )
        )

    return ParsedAnswer(
        mode="answer",
        segments=segments,
        needs=[],
        raw_text=raw_text,
        parse_warnings=parse_warnings,
        schema_valid=True,
    )
