from __future__ import annotations

import argparse

from evaluation.scripts.run_retrieval_eval import run
from rag.retrieval import Retriever,BM25Index,ChunkStore, HybridRetriever, HybridRerankRetriever
from rag.rerank.cross_encoder_reranker_API import CrossEncoderReranker, CrossEncoderConfig
import torch

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--golden",required=True, help="Path to eval/data/golden_set.json") #  evaluation/data/golden_set.json
    ap.add_argument("--dump_root", default="evaluation/retriever/runs", help="Root folder for run artifacts")
    args = ap.parse_args()
    

    chunking_loc= "data//run_2//processed_chunks.parquet"
    embedding_loc= "data//run_2//embeddings"
    indexing_loc=  "data//run_2//index//flat_ip//b401572ca42d"
    bm25_path="artifacts//bm25"
    chunk_store_path="data//run_2//processed_chunks.parquet"

    baseline = Retriever(chunks_path=chunking_loc,embeddings_dir=embedding_loc,index_dir=indexing_loc)
   
    bm25 = BM25Index.load(dir_path=bm25_path)
    store = ChunkStore(path=chunk_store_path)     
    hybrid = HybridRetriever(
        dense=baseline,
        bm25=bm25,
        chunk_store=store
    )

    reranker = CrossEncoderReranker(
        CrossEncoderConfig(
                model_name="cross-encoder/ms-marco-MiniLM-L-6-v2",
                model_type="api", # can be api, local
                batch_size=16,
                max_text_chars=2000,
                normalize_scores=False,
                device="cuda" if torch.cuda.is_available() else "cpu",
                url="https://b233-34-125-221-96.ngrok-free.app"
            
        )
    )
    hybrid_rerank = HybridRerankRetriever(retriever=hybrid,reranker=reranker)

    #baseline
    run(
        retriever=baseline,
        golden_path=args.golden,
        run_tag="baseline",
        notes="dense only",
    )

    # 2) +hybrid

    run(
        retriever=hybrid,
        golden_path=args.golden,
        run_tag="+hybrid",
        notes="dense + bm25 (hybrid)",
    )

    # 3) +hybrid+rerank

    run(
        retriever=hybrid_rerank,
        golden_path=args.golden,
        run_tag="+hybrid+rerank",
        notes="hybrid candidates -> cross-encoder rerank",
    )

    print("Step 6 eval complete. Check eval/results/ladder.csv")


if __name__ == "__main__":
    main()