from rag.utils.contracts import Citation, ParsedAnswer, PipelineResult

ABSTAIN_MESSAGE = "I can't answer that from the available source material."
PARSE_ERROR_MESSAGE = "I couldn't produce a properly grounded answer from the retrieved evidence."


def _render_chat_style_answer(blocks: list[str]) -> str:
    cleaned = [block.strip() for block in blocks if block and block.strip()]
    if not cleaned:
        return ""
    return "\n\n".join(cleaned)


def format_answer(parsed: ParsedAnswer) -> str:
    if parsed.mode == "answer":
        return _render_chat_style_answer([segment.text for segment in parsed.segments])
    if parsed.mode == "abstain":
        return ABSTAIN_MESSAGE
    return PARSE_ERROR_MESSAGE

def get_used_citations(result: PipelineResult) -> list[Citation]:
    chunk_map = {chunk.chunk_id: chunk for chunk in result.retrieved_chunks}
    seen = set()
    citations: list[Citation] = []

    for segment in result.parsed_output.segments:
        for chunk_id in segment.resolved_chunk_ids:
            if not chunk_id or chunk_id in seen:
                continue

            chunk = chunk_map.get(chunk_id)
            if chunk is None or chunk.citation is None:
                continue

            seen.add(chunk_id)
            citations.append(chunk.citation)

    return citations
