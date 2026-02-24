# scripts/smoke_retrieval.py
from rag.retrieval.retrieve import Retriever

def main():
    r = Retriever(
        index_dir="data//index//113723aae9b1",
        embeddings_dir="data//embeddings//20260221_172629__2954117bb965",
        chunks_path="data/processed_chunks.parquet",
        device="cpu",
    )

    q = "What is diabetes and how is it diagnosed?"
    hits = r.retrieve(q, top_k=5)
    print(q,hits)
    for i, h in enumerate(hits, 1):
        print(f"\n#{i} score={h.score:.4f} chunk_id={h.chunk_id}")
        print(h.text[:300].replace("\n", " "))

if __name__ == "__main__":
    main()