from typing import Any, Dict
from rag.utils.contracts import Citation, RetrievedChunk


REQUIRED_CITATION_KEYS = ("source", "title", "section", "chunk_id","doc_id","url")

def _normalize_citation_fields(row: Dict[str, Any]) -> tuple[Citation, Dict[str, Any]]:
    """
    Ensures citation has source/title/section/chunk_id.
    If parquet uses different keys, map them here.
    """

    # Aliases you might have in your parquet
    key_map = {
        "source_url": "url",
        "doc_url": "source",
        "document_url": "source",
        "heading": "title",
        "section_path":"section",
        "header": "section",
        "source_url": "url",
        "document_url": "url",
        "doc_url": "url",
    }

    norm = dict(row)
    for k_from, k_to in key_map.items():
        if k_to not in norm and k_from in norm:
            norm[k_to] = norm[k_from]

    missing = [k for k in ("source", "title", "section", "chunk_id","doc_id","url") if k not in norm or norm[k] is None]
    if missing:
        raise ValueError(
            f"Missing required citation fields {missing}. "
            f"Ensure processed_chunks.parquet includes {REQUIRED_CITATION_KEYS} (or add alias mapping)."
        )

    citation = Citation(
        source=str(norm["source"]),
        title=str(norm["title"]),
        section=str(norm["section"]),
        chunk_id=str(norm["chunk_id"]),
        doc_id=str(norm["doc_id"]),
        url=str(norm["url"])
    )

    # metadata = everything else except required fields + chunk_text
    for k in ("source", "title", "section","doc_id","url"):
        norm.pop(k, None)
    # chunk_id stays in metadata? keep it only once
    norm.pop("chunk_id", None)
    norm.pop("chunk_text", None)

    return citation, norm


def build_retrieved_chunk_from_row(
    row: Dict[str, Any],
    *,
    score: float = 0.0,
    rank: int | None = None,
) -> RetrievedChunk:
    """
    Normalize a parquet row into a RetrievedChunk with a proper Citation object.
    """
    row = dict(row)

    # ensure chunk_id is string
    if "chunk_id" in row and row["chunk_id"] is not None:
        row["chunk_id"] = str(row["chunk_id"])

    text = str(row.get("chunk_text", ""))

    citation, extra_md = _normalize_citation_fields(row)

    return RetrievedChunk(
        chunk_id=citation.chunk_id,
        score=float(score),
        text=text,
        citation=citation,
        metadata=extra_md,
        rank=rank,
    )