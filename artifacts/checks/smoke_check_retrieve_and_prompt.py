# smoke_check_retrieve_and_prompt.py
#
# Run:
#   python smoke_check_retrieve_and_prompt.py
#
# Optional env vars:
#   RAG_INDEX_DIR, RAG_EMB_DIR, RAG_CHUNKS_PATH, RAG_SMOKE_QUERY

import os
import re
from pprint import pprint

from rag.retrieval.retrieve import Retriever
from rag.prompt import build_evidence_block, build_prompt

INDEX_DIR = os.getenv("RAG_INDEX_DIR", "artifacts/index")
EMB_DIR = os.getenv("RAG_EMB_DIR", "artifacts/embeddings")
CHUNKS_PATH = os.getenv("RAG_CHUNKS_PATH", "data/processed_chunks.parquet")
SMOKE_QUERY = os.getenv("RAG_SMOKE_QUERY", "What is diabetes?")

HEADER_RE = re.compile(r"\[C\d+\]\s+doc_id=.*\|\s+title=.*\|\s+section=.*\|\s+chunk_id=.*\|\s+source=.*")


def check(cond: bool, msg_ok: str, msg_fail: str):
    if cond:
        print(f"[OK]  {msg_ok}")
        return True
    print(f"[FAIL] {msg_fail}")
    return False


def main():
    print("=== CONFIG ===")
    print("INDEX_DIR   =", INDEX_DIR)
    print("EMB_DIR     =", EMB_DIR)
    print("CHUNKS_PATH =", CHUNKS_PATH)
    print("QUERY       =", SMOKE_QUERY)
    print()

    # -------------------------
    # Slice 3.1: retrieve smoke
    # -------------------------
    r = Retriever(index_dir=INDEX_DIR, embeddings_dir=EMB_DIR, chunks_path=CHUNKS_PATH)
    out = r.retrieve(SMOKE_QUERY, top_k=5)

    check(isinstance(out, list), "Retriever returned a list", f"Retriever returned {type(out)}")
    check(len(out) > 0, f"Retriever returned {len(out)} results", "Retriever returned 0 results (check paths/index/embeddings/chunks)")

    if not out:
        return

    first = out[0]
    print("\n=== FIRST RESULT (SUMMARY) ===")
    print("chunk_id :", getattr(first, "chunk_id", None))
    print("score    :", getattr(first, "score", None))
    print("text[:200]:", (getattr(first, "text", "") or "")[:200].replace("\n", " "))
    print("\n=== FIRST RESULT (CITATION) ===")
    c = getattr(first, "citation", None)
    pprint({
        "doc_id": getattr(c, "doc_id", None),
        "title": getattr(c, "title", None),
        "section": getattr(c, "section", None),
        "chunk_id": getattr(c, "chunk_id", None),
        "source": getattr(c, "source", None),
    })

    # Citation field checks
    required = ("source", "title", "section", "chunk_id", "doc_id")
    for attr in required:
        ok = hasattr(c, attr) and isinstance(getattr(c, attr), str) and bool(getattr(c, attr))
        check(ok, f"Citation field '{attr}' present and non-empty", f"Citation field '{attr}' missing/empty")

    # ------------------------------------
    # Slice 3.1: filtering by doc_id smoke
    # ------------------------------------
    target_doc_id = c.doc_id
    filtered = r.retrieve(SMOKE_QUERY, top_k=5, filters={"doc_id": target_doc_id}, oversample=10)

    check(len(filtered) > 0, f"Filter returned {len(filtered)} results", "Filter returned 0 results (try oversample bigger, or doc_id not filterable)")
    if filtered:
        all_match = all(x.citation.doc_id == target_doc_id for x in filtered)
        check(all_match, "All filtered results match doc_id", "Some filtered results do not match doc_id")

    # -----------------------------
    # Slice 3.1: determinism check
    # -----------------------------
    a = r.retrieve(SMOKE_QUERY, top_k=5)
    b = r.retrieve(SMOKE_QUERY, top_k=5)
    same_ids = [x.chunk_id for x in a] == [x.chunk_id for x in b]
    check(same_ids, "Determinism: top-k chunk_ids match across two runs", "Determinism failed: top-k chunk_ids changed")

    print("\nTop-k chunk_ids run A:", [x.chunk_id for x in a])
    print("Top-k chunk_ids run B:", [x.chunk_id for x in b])

    # -----------------------------------------
    # Slice 3.2: evidence block + prompt smoke
    # -----------------------------------------
    evidence_block, items = build_evidence_block(out, max_chunks=5, max_chars_per_chunk=300)

    print("\n=== EVIDENCE BLOCK (PRINT) ===")
    print(evidence_block)

    check("EVIDENCE" in evidence_block, "Evidence block contains 'EVIDENCE'", "Evidence block missing 'EVIDENCE'")
    check(len(items) == len(out[:5]), f"Evidence items count = {len(items)}", f"Evidence items count mismatch: got {len(items)} expected {len(out[:5])}")

    expected_tags = [f"[C{i}]" for i in range(1, len(items) + 1)]
    got_tags = [it.citation_tag for it in items]
    check(got_tags == expected_tags, "Evidence tags are [C1]..[Ck] in order", f"Tag mismatch: got {got_tags} expected {expected_tags}")

    check(bool(HEADER_RE.search(evidence_block)), "Evidence header regex matches", "Evidence header format does not match expected contract")

    prompt = build_prompt("Explain diabetes briefly.", out[:3])
    print("\n=== PROMPT (SYSTEM) ===")
    print(prompt["system"])
    print("\n=== PROMPT (USER) ===")
    print(prompt["user"])

    check("Citation rules" in prompt["system"], "Prompt system includes citation rules", "Prompt system missing 'Citation rules'")
    check("Question:" in prompt["user"], "Prompt user includes 'Question:'", "Prompt user missing 'Question:'")
    check("EVIDENCE" in prompt["user"], "Prompt user includes 'EVIDENCE'", "Prompt user missing 'EVIDENCE'")
    check("[C1]" in prompt["user"], "Prompt user includes [C1]", "Prompt user missing [C1]")


if __name__ == "__main__":
    main()