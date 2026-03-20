from rag.utils.contracts import ParsedAnswer, Citation, PipelineResult


def format_answer(parsed: ParsedAnswer) -> str:
    if parsed.mode == "answer":
        return "\n".join(f"- {b}" for b in parsed.bullets)
    if parsed.mode == "abstain":
        return parsed.abstain_reason or "I don’t have enough evidence to answer safely."
    return "I couldn’t produce a properly grounded answer from the retrieved evidence."


def get_used_citations(result: PipelineResult) -> list[Citation]:
    chunk_map = {chunk.chunk_id: chunk for chunk in result.retrieved_chunks}
    seen = set()
    citations: list[Citation] = []

    for chunk_ids in result.parsed_output.resolved_chunk_ids_by_bullet:
        for chunk_id in chunk_ids:
            if not chunk_id or chunk_id in seen:
                continue

            chunk = chunk_map.get(chunk_id)
            if chunk is None:
                continue

            if chunk.citation is None:
                continue

            seen.add(chunk_id)
            citations.append(chunk.citation)

    return citations