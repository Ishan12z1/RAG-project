# scripts/smoke_check_faiss_index.py
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import faiss


def main() -> None:
    # choose latest index dir
    root = Path("data/index")
    idx_dirs = sorted([p for p in root.iterdir() if p.is_dir()], key=lambda p: p.name)
    if not idx_dirs:
        raise FileNotFoundError("No index dirs under data/index")
    idx_dir = idx_dirs[-1]

    meta = json.loads((idx_dir / "meta.json").read_text(encoding="utf-8"))
    index = faiss.read_index(str(idx_dir / "faiss.index"))

    # load embeddings used by this index
    emb_dir = Path("data/embeddings") / meta["emb_hash"]
    vecs = np.load(emb_dir / "vectors.npy").astype(np.float32)
    vecs = np.ascontiguousarray(vecs)

    # query with an existing vector; top-1 should usually be itself
    i = min(3, vecs.shape[0] - 1)
    q = vecs[i : i + 1]
    D, I = index.search(q, 5)

    print("query_row:", i)
    print("top5 indices:", I[0].tolist())
    print("top5 scores:", D[0].tolist())

    assert 0 <= I[0][0] < vecs.shape[0]
    # If vectors are normalized and using IP, self score should be close to 1
    if meta["metric"] == "ip":
        assert D[0][0] > 0.90

    print("OK")


if __name__ == "__main__":
    main()