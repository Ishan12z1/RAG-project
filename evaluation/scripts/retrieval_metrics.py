import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import List
from rag.retrieval.retrieve import Retriever
from rag.utils.contracts import RetrievedChunk


def read_jsonl(path: Path):
    text = path.read_text(encoding="utf-8-sig").strip()
    if not text:
        return

    # JSON array (your current format)
    if text[0] == "[":
        data = json.loads(text)
        if not isinstance(data, list):
            raise ValueError("Expected a JSON array at top level.")
        for obj in data:
            yield obj
        return

    # JSONL (one JSON object per line)
    with path.open("r", encoding="utf-8-sig") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def pctl(values,p):
    if not values:
        return None
    xs=sorted(values)
    k=int(round((p/100.0)*(len(xs)-1)))
    return xs[max(0,min(k,len(xs)-1))]


def compute_metrics(per_query,ks=(1,5,10),mrr_k=10):
    eval_rows=[r for r in per_query if r.get("relevant_chunk_ids")]
    n=len(eval_rows)
    recall={k:0 for k in ks}
    mrr_sum=0.0

    for r in eval_rows:
        retrieved=r["retrieved_chunk_ids"]
        relevant = set(r["relevant_chunk_ids"]) 

        # Recall@K
        for k in ks:
            topk=retrieved[:k]
            if any(cid in relevant for cid in topk):
                recall[k]+=1
        
        # MRR
        rr=0.0
        for i,cid in enumerate(retrieved[:mrr_k],start=1):
            if cid in relevant:
                rr=1.0/i
                break
        mrr_sum+=rr


    recall = {f"recall@{k}": (recall[k] / n if n else 0.0) for k in ks}
    mrr = (mrr_sum / n if n else 0.0)
    return recall, mrr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--golden", type=str, default="eval/golden_set.jsonl")
    ap.add_argument("--out_dir", type=str, default="eval/results")
    ap.add_argument("--ks", type=str, default="1,5,10")
    ap.add_argument("--mrr_k", type=int, default=10)
    ap.add_argument("--max_k", type=int, default=10)
    ap.add_argument("--limit", type=int, default=0, help="0 = no limit (run all)")
    ap.add_argument("--nprobe", type=int, default=None)
    ap.add_argument("--ef_search", type=int, default=None)
    args = ap.parse_args()

    ks=tuple(int(x.strip()) for x in args.ks.split(",") if x.strip())
    max_k = max(args.max_k, max(ks), args.mrr_k)

    golden_path = Path(args.golden)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    per_query = []
    retrieve_ms = []
    retriever=Retriever(index_dir="artifacts/index",embeddings_dir="artifacts/embeddings")
    n_skipped_abstained=0

    for idx,row in enumerate(read_jsonl(golden_path),start=1):
        if args.limit and idx > args.limit:
            break

        qid = row["qid"]
        query = row["query"]
        bucket = row.get("bucket", "answerable")

        # Build relevant list from your golden format
        relevant_chunk_ids = [c["chunk_id"] for c in row.get("chunk_ids", [])]
        # If the row is abstained OR has no relevant chunks, skip it for recall/MRR
        if bucket != "answerable" or not relevant_chunk_ids:
            n_skipped_abstained += 1
            continue

        t0=time.perf_counter()
        results:List[RetrievedChunk]=retriever.retrieve(query,max_k)
        retrieved_chunk_ids= [r.chunk_id for r in results]
    
        dt_ms=(time.perf_counter()-t0)*1000.0
        retrieve_ms.append(dt_ms)

        per_query.append({
            "qid": qid,
            "query": query,
            "relevant_chunk_ids": relevant_chunk_ids,
            "retrieved_chunk_ids": retrieved_chunk_ids,
            "retrieve_ms": dt_ms,
        })
    
    
    recall, mrr = compute_metrics(per_query, ks=ks, mrr_k=args.mrr_k)

    report = {
        "n_queries_evaluated": len(per_query),
        "n_queries_skipped": n_skipped_abstained,
        **recall,
        f"mrr@{args.mrr_k}": mrr,
        "retrieve_p50_ms": median(retrieve_ms) if retrieve_ms else None,
        "retrieve_p95_ms": pctl(retrieve_ms, 95),
        "params": {
            "nprobe": args.nprobe,
            "ef_search": args.ef_search,
            "max_k": max_k,
            "ks": ks,
            "mrr_k": args.mrr_k,
        },
    }

    # JSON
    (out_dir / "retrieval_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    # CSV (single row)
    csv_cols = ["n_queries"] + [f"recall@{k}" for k in ks] + [f"mrr@{args.mrr_k}", "retrieve_p50_ms", "retrieve_p95_ms"]
    csv_line = ",".join(str(report.get(c, "")) for c in csv_cols)
    (out_dir / "retrieval_report.csv").write_text(",".join(csv_cols) + "\n" + csv_line + "\n", encoding="utf-8")

    # Optional per-query debug (keep; helps you fix misses quickly)
    with (out_dir / "retrieval_details.jsonl").open("w", encoding="utf-8") as f:
        for r in per_query:
            f.write(json.dumps(r) + "\n")

if __name__=="__main__":
    main()