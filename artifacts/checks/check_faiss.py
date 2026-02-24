# eval/check_faiss_index.py
import argparse
from pathlib import Path
import numpy as np
import faiss

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index_path", required=True, help="Path to saved faiss index file")
    ap.add_argument("--dim", type=int, default=0, help="Optional: expected dimension (0 = skip check)")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--n_queries", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--ef_search", type=int, default=64, help="Only used if index is HNSW")
    args = ap.parse_args()

    index_path = Path(args.index_path)
    assert index_path.exists(), f"Index not found: {index_path}"

    index = faiss.read_index(str(index_path))

    print("=== FAISS INDEX INFO ===")
    print("class:", type(index))
    # metric_type is not available on all wrappers the same way; this is best-effort
    if hasattr(index, "metric_type"):
        print("metric_type:", index.metric_type, "(0=L2, 1=IP)")
    print("ntotal:", index.ntotal)

    # Dim check (works for most indexes)
    if hasattr(index, "d"):
        print("dim (index.d):", index.d)
        if args.dim:
            assert index.d == args.dim, f"Expected dim={args.dim}, got {index.d}"

    # HNSW-specific knobs
    is_hnsw = hasattr(index, "hnsw")
    print("is_hnsw:", is_hnsw)
    if is_hnsw:
        # set efSearch to ensure search actually explores enough
        index.hnsw.efSearch = args.ef_search
        print("hnsw.efSearch set to:", index.hnsw.efSearch)
        if hasattr(index.hnsw, "M"):
            print("hnsw.M:", index.hnsw.M)

    # Simple search test: query random vectors of same dim
    d = index.d if hasattr(index, "d") else None
    assert d is not None, "Could not infer index dimension (index.d missing)."

    rng = np.random.default_rng(args.seed)

    # For IP indexes, queries should be normalized if your vectors were normalized.
    # This script cannot know; do it anyway because most IP pipelines normalize.
    Xq = rng.normal(size=(args.n_queries, d)).astype("float32")
    faiss.normalize_L2(Xq)

    D, I = index.search(Xq, args.k)

    print("\n=== SEARCH OUTPUT SAMPLE ===")
    for qi in range(args.n_queries):
        print(f"q{qi}: ids={I[qi].tolist()} scores={D[qi].tolist()}")

    # Basic validity checks
    assert I.shape == (args.n_queries, args.k)
    assert np.all(I >= -1), "FAISS returned invalid ids"
    if index.ntotal > 0:
        # If ntotal is large enough, you expect mostly non -1 ids
        non_empty = (I != -1).sum()
        print("\nnon-empty results:", int(non_empty), "/", int(I.size))

    print("\nOK: index loads and returns results.")

if __name__ == "__main__":
    main()