from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple, Any
import time
import statistics
import json
import os

from rag.utils.contracts import RetrievalExample, RetrievalMetrics, PerQueryResult
from rag.retrieval.retrieve import Retriever

class RetrievalEvaluator:
    def __init__(self,
                 examples:Sequence[RetrievalExample],
                 ks:Sequence[int]=(1,3,5,10),
                 *,
                 name="retrieval_eval")->None:
        if not examples:
            raise ValueError("example is empty")
        if not ks:
            raise ValueError("ks is empty")
        
        self.examples=examples
        self.ks=sorted(set(ks))
        self.name=name
    
    def evaluate(self,
                 retriever:Retriever,   
                 *,
                 run_tag:str="baseline",     
                 notes: str = "",
                 max_k: Optional[int] = None,
                 dump_dir: Optional[str] = None,
                 strict_unique_ids: bool = True,
                 )->Tuple[RetrievalMetrics,List[PerQueryResult]]:
        
        max_k=max_k or max(self.ks)
        latencies_ms:List[float]=[]
        per_query:List[PerQueryResult]=[]

        # Counter for metrics at each k
        recall_hits={k:0 for k in self.ks} # query level recall: hit at least one relevant in top-k
        rr_sum={k:0 for k in self.ks} # reciprocal ranked sum across queries 
        
        for ex in self.examples:
            t0=time.perf_counter()
            retrieved=list(retriever.retrieve(ex.query,max_k))
            t1=time.perf_counter()
            latency_ms = (t1 - t0) * 1000.0

            retrieved = sorted(retrieved, key=lambda x: float(getattr(x, "score", 0.0)), reverse=True)[:max_k]
            retrieved_ids = [r.chunk_id for r in retrieved]

            if strict_unique_ids:
                if len(retrieved_ids) != len(set(retrieved_ids)):
                    raise ValueError(
                        f"Duplicate chunk_ids returned for qid={ex.qid}. "
                        f"Expected unique ids in ranked list."
                    )
            
            gold = ex.gold_chunk_ids
            first_hit_rank=None
            for idx, cid in enumerate(retrieved_ids[:max_k]):
                if cid in gold:
                    first_hit_rank= idx+1
                    break
                    
            
            for k in self.ks:
                topk=retrieved_ids[:k]
                hit=any(cid in gold for cid in topk)
                if hit:
                    recall_hits[k]+=1
                
                rr=0.0
                if first_hit_rank is not None and first_hit_rank<=k:
                    rr=1.0/ first_hit_rank

                rr_sum[k]+=rr
            
            latencies_ms.append(latency_ms)
            per_query.append(
                PerQueryResult(
                    qid=ex.qid,
                    query=ex.query,
                    gold_chunk_ids=sorted(list(ex.gold_chunk_ids)),
                    retrieved_chunk_ids=retrieved_ids[:max_k],
                    first_hit_rank=first_hit_rank,
                    latency_ms=latency_ms,
                )
            )
        n=len(self.examples)
        recall_at_k={k:recall_hits[k]/n for k in self.ks}
        mrr_at_k={k:rr_sum[k]/n for k in self.ks}
        hit_rate_at_k=dict(recall_at_k)

        p95=_percentile(latencies_ms,95)
        p50=_percentile(latencies_ms,50)
        mean_ms=statistics.mean(latencies_ms)
        
        metrics=RetrievalMetrics(
            recall_at_k=recall_at_k,
            mrr_at_k=mrr_at_k,
            hit_rate_at_k=hit_rate_at_k,
            p50_ms=p50,
            p95_ms=p95,
            mean_ms=mean_ms,
            n_queries=n
        )

        if dump_dir:
            os.makedirs(dump_dir, exist_ok=True)
            summary_path = os.path.join(dump_dir, "summary.json")
            perq_path = os.path.join(dump_dir, "per_query.jsonl")

            with open(summary_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "name": self.name,
                        "run_tag": run_tag,
                        "notes": notes,
                        "ks": self.ks,
                        "n_queries": n,
                        "metrics": {
                            "recall_at_k": metrics.recall_at_k,
                            "mrr_at_k": metrics.mrr_at_k,
                            "p50_ms": metrics.p50_ms,
                            "p95_ms": metrics.p95_ms,
                            "mean_ms": metrics.mean_ms,
                        },
                    },
                    f,
                    indent=2,
                )

            with open(perq_path, "w", encoding="utf-8") as f:
                for r in per_query:
                    f.write(
                        json.dumps(
                            {
                                "qid": r.qid,
                                "query": r.query,
                                "gold_chunk_ids": r.gold_chunk_ids,
                                "retrieved_chunk_ids": r.retrieved_chunk_ids,
                                "first_hit_rank": r.first_hit_rank,
                                "latency_ms": r.latency_ms,
                            }
                        )
                        + "\n"
                    )

        return metrics, per_query

def _percentile(xs:Sequence[float],p:float)->float:
    if not xs:
        return 0.0
    xs_sorted=sorted(xs)
    if len(xs_sorted)==1:
        return xs_sorted[0]
    
    # nearest rank percentile
    k=int(round((p/100)*(len(xs_sorted)-1)))
    k=max(0,min(k,len(xs_sorted)-1))
    return xs_sorted[k]

