from __future__ import annotations

import hashlib
from typing import  Tuple
from rag.retrieval.bm25_index import BM25Index, BM25Config
import pandas as pd


def iter_chunks(path: str = "data/run_2/processed_chunks.parquet") -> list[Tuple[str, str]]:
    df = pd.read_parquet(path)
    chunks = []

    for _, row in df.iterrows():
        chunks.append((row["chunk_id"], row["chunk_text"]))

    return chunks


def compute_corpus_fingerprint(all_chunks: list[Tuple[str, str]]) -> Tuple[str, list[Tuple[str, str]]]:
    """
    Deterministic fingerprint: hash of (chunk_id + '\0' + text) in chunk_id order.
    """
    h = hashlib.sha256()

    all_chunks.sort(key=lambda x: x[0])

    for cid, txt in all_chunks:
        h.update(cid.encode("utf-8"))
        h.update(b"\x00")
        h.update(txt.encode("utf-8"))
        h.update(b"\x00")

    return h.hexdigest(), all_chunks


def main():
    chunks = iter_chunks()
    fp, all_chunks = compute_corpus_fingerprint(chunks)

    idx = BM25Index.build(
        all_chunks,
        config=BM25Config(k1=1.5, b=0.75),
        corpus_fingerprint=fp,
    )
    idx.save("artifacts/bm25")


if __name__ == "__main__":
    main()