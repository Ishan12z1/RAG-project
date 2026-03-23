from rag.utils.contracts import Citation, ParsedAnswer, PipelineResult


def _render_chat_style_answer(bullets: list[str]) -> str:
    cleaned = [bullet.strip() for bullet in bullets if bullet and bullet.strip()]
    if not cleaned:
        return ""
    return "\n\n".join(cleaned)


def format_answer(parsed: ParsedAnswer) -> str:
    if parsed.mode == "answer":
        return _render_chat_style_answer(parsed.bullets)
    if parsed.mode == "abstain":
        return parsed.abstain_reason or "I can't answer that from the available source material."
    return "I couldn't produce a properly grounded answer from the retrieved evidence."


def get_used_citations(result: PipelineResult) -> list[Citation]:
    chunk_map = {chunk.chunk_id: chunk for chunk in result.retrieved_chunks}
    seen = set()
    citations: list[Citation] = []

    for chunk_ids in result.parsed_output.resolved_chunk_ids_by_bullet:
        for chunk_id in chunk_ids:
            if not chunk_id or chunk_id in seen:
                continue

            chunk = chunk_map.get(chunk_id)
            if chunk is None or chunk.citation is None:
                continue

            seen.add(chunk_id)
            citations.append(chunk.citation)

    return citations
