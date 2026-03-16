import argparse, json, os
from typing import Dict, List, Set, Tuple
from rag.retrieval import Retriever,BM25Index, bm25_builder, bm25_index, ChunkStore, HybridRetriever


def load_queries(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--queries", default="eval/data/queries.jsonl")
    ap.add_argument("--out", default="eval/data/silver_pool.jsonl")
    ap.add_argument("--dense_k", type=int, default=100)
    ap.add_argument("--bm25_k", type=int, default=100)
    ap.add_argument("--hybrid_k", type=int, default=100)
    ap.add_argument("--max_pool", type=int, default=250)
    ap.add_argument("--use_hybrid", action="store_true")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    chunking_loc= "data//run_2//processed_chunks.parquet"
    embedding_loc= "data//run_2//embeddings"
    indexing_loc=  "data//run_2//index//flat_ip//b401572ca42d"
    bm25_path="artifacts//bm25"
    chunk_store_path="data//run_2//processed_chunks.parquet"

    dense = Retriever(chunks_path=chunking_loc,embeddings_dir=embedding_loc,index_dir=indexing_loc)
    bm25 = BM25Index.load(dir_path=bm25_path)
    store = ChunkStore(path=chunk_store_path)       # must have .get(chunk_id)->RetrievedChunk

    hybrid = HybridRetriever(
        dense=dense,
        bm25=bm25,
        chunk_store=store
    ) if args.use_hybrid else None

    queries = load_queries(args.queries)

    with open(args.out, "w", encoding="utf-8") as fout:
        for item in queries:
            qid = item["qid"]
            query = item["query"]

            # Dense candidates
            dense_hits = dense.retrieve(query=query, top_k=args.dense_k)
            dense_map = {c.chunk_id: float(c.score) for c in dense_hits}

            # BM25 candidates
            bm25_hits = bm25.search(query, top_k=args.bm25_k)
            bm25_map = {cid: float(sc) for cid, sc in bm25_hits}

            # Optional hybrid candidates (often redundant but can add a few)
            hybrid_map = {}
            if hybrid is not None:
                hy_hits = hybrid.retrieve(query=query, top_k=args.hybrid_k)
                hybrid_map = {c.chunk_id: float(c.score) for c in hy_hits}

            # Union pool ids
            pool_ids = list(set(dense_map) | set(bm25_map) | set(hybrid_map))

            # If pool too large, keep a cap by a simple heuristic:
            # rank by max(normalized-ish proxy): max of available scores (not true normalization)
            # This is just to keep review size bounded.
            def proxy(cid: str) -> float:
                return max(dense_map.get(cid, float("-inf")),
                           bm25_map.get(cid, float("-inf")),
                           hybrid_map.get(cid, float("-inf")))

            pool_ids.sort(key=proxy, reverse=True)
            pool_ids = pool_ids[: args.max_pool]

            # Hydrate pool with text so reranker & labeling can work
            pool = []
            for cid in pool_ids:
                ch = None
                # Prefer dense object if present to avoid store hit
                if cid in dense_map:
                    # find chunk in dense_hits (small list)
                    for d in dense_hits:
                        if d.chunk_id == cid:
                            ch = d
                            break
                if ch is None:
                    ch = store.get(cid)

                pool.append({
                    "chunk_id": cid,
                    "dense_score": dense_map.get(cid),
                    "bm25_score": bm25_map.get(cid),
                    "hybrid_score": hybrid_map.get(cid) if hybrid is not None else None,
                })

            fout.write(json.dumps({
                "qid": qid,
                "query": query,
                "pool": pool,
            }) + "\n")

if __name__ == "__main__":
    main()