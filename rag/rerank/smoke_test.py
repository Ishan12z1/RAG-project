# scripts/smoke_test_cross_encoder_reranker.py
from __future__ import annotations

from rag.rerank.cross_encoder_reranker_API import CrossEncoderReranker, CrossEncoderConfig
from rag.utils.contracts import RetrievedChunk
import torch 

def main():
    reranker = CrossEncoderReranker(
        CrossEncoderConfig(
                model_name="cross-encoder/ms-marco-MiniLM-L-6-v2",
                model_type="api", # can be api, local
                batch_size=16,
                max_text_chars=2000,
                normalize_scores=False,
                device="cuda" if torch.cuda.is_available() else "cpu",
                url="https://581b-34-50-185-45.ngrok-free.app"
            
        )
    )

    query = "What is diabetes, in simple terms?"
    candidates = [
        RetrievedChunk(chunk_id="c1", text="Diabetes is a condition where blood sugar stays too high.",score=0.66,citation="google.com",metadata=None),
        RetrievedChunk(chunk_id="c2", text="This section discusses hypertension and blood pressure.",score=0.55,citation="google.com",metadata=None),
        RetrievedChunk(chunk_id="c3", text="Diabetes mellitus: high blood glucose due to insulin issues.",score=0.6,citation="google.com",metadata=None),
    ]
    outs = reranker.rerank(query, candidates)
    score_by_id = {o.chunk_id: score for (o,score) in outs}

    print("Scores:", score_by_id)

    reranked = sorted(candidates, key=lambda c: score_by_id[c.chunk_id], reverse=True)
    print("\nReranked order:")
    for i, c in enumerate(reranked, start=1):
        print(f"{i}. {c.chunk_id} score={score_by_id[c.chunk_id]:.4f} text={c.text}")

    assert {c.chunk_id for c in candidates} == {o.chunk_id for (o,_) in outs}
    print("\nOK: cross-encoder reranker works.")


if __name__ == "__main__":
    main()