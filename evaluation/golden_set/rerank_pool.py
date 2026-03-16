import argparse, json, os
from typing import Dict, List
import torch

from rag.rerank.cross_encoder_reranker_API import CrossEncoderReranker, CrossEncoderConfig
from rag.retrieval import ChunkStore
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--silver", default="eval/data/silver_pool.jsonl")
    ap.add_argument("--out", default="eval/data/reranked_pool.jsonl")
    ap.add_argument("--topn", type=int, default=15)
    ap.add_argument("--model", default="cross-encoder/ms-marco-MiniLM-L-6-v2")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--max_chars", type=int, default=2000)
    ap.add_argument("--device", default=None)  # "cuda" or "cpu"
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    reranker = CrossEncoderReranker(
        CrossEncoderConfig(
                model_name="cross-encoder/ms-marco-MiniLM-L-6-v2",
                model_type="api", # can be api, local
                batch_size=16,
                max_text_chars=2000,
                normalize_scores=False,
                device="cuda" if torch.cuda.is_available() else "cpu",
                url="https://2a53-34-50-185-45.ngrok-free.app"
            
        )
    )
    chunk_store_path="data//run_2//processed_chunks.parquet"
    store = ChunkStore(path=chunk_store_path)  
    with open(args.silver, "r", encoding="utf-8") as fin, open(args.out, "w", encoding="utf-8") as fout:
        for line in fin:
            item = json.loads(line)
            qid, query = item["qid"], item["query"]
            pool = item["pool"]

            inputs = [store.get(p["chunk_id"]) for p in pool]
            outs = reranker.rerank(query, inputs)
            score_by = {o.chunk_id: float(score) for (o,score) in outs}

            for p in pool:
                p["rerank_score"] = score_by.get(p["chunk_id"])

            pool.sort(key=lambda p: (p["rerank_score"] is not None, p["rerank_score"]), reverse=True)
            pool = pool[: args.topn]

            fout.write(json.dumps({
                "qid": qid,
                "query": query,
                "candidates": pool,
            }) + "\n")

if __name__ == "__main__":
    main()