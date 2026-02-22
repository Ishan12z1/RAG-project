from collections import Counter 
import random 
from typing import List, Dict
import pandas as pd 
from rag.chunking import ChunkPolicy
from pathlib import Path
import json 
import argparse
import sys
def _percentile(sorted_vals:List[int],p:float)-> int:
    if not sorted_vals:
        return 0 
    idx=int(round((p/100.0)*len(sorted_vals)-1))
    idx=max(0,min(len(sorted_vals)-1,idx))

    return sorted_vals[idx]

def compute_qa_report(df:pd.DataFrame,ploicy: ChunkPolicy,sample_limit_for_lines:int=5000)->Dict:
    report:Dict={}
    report["total_docs"]=int(df["doc_id"].nunique()) if len(df) else 0 
    report["total_chunks"]=int(len(df))
    
    toks=df["token_count"].astype(int).tolist() if len(df) else 0 
    toks_sorted=sorted(toks)
    report["token_min"]=int(min(toks)) if toks else 0 
    report["token_median"]=int(_percentile(toks_sorted,50))
    report["token_p95"]=int(_percentile(toks_sorted,95))
    report["token_max"]=int(max(toks)) if toks else 0 

    report["above_max_tokens"] = int((df["token_count"] < ploicy.max_tokens).sum()) if len(df) else 0 
    report["below_min_tokens"] = int((df["token_count"] < ploicy.min_tokens).sum()) if len(df) else 0 

    if len(df):
        per_doc = df.groupby("doc_id").size().astype(int).tolist()
        per_doc_sorted = sorted(per_doc)
        report["chunks_per_doc_min"] = int(min(per_doc))
        report["chunks_per_doc_median"] = int(_percentile(per_doc_sorted, 50))
        report["chunks_per_doc_p95"] = int(_percentile(per_doc_sorted, 95))
        report["chunks_per_doc_max"] = int(max(per_doc))
    else:
        report["chunks_per_doc_min"] = 0
        report["chunks_per_doc_median"] = 0
        report["chunks_per_doc_p95"] = 0
        report["chunks_per_doc_max"] = 0

    line_counts:Counter[str]=Counter()

    if len(df):
        # Sample chunks to keep runtime sane 
        n=min(len(df),sample_limit_for_lines)
        sampled_texts=df["chunk_text"].sample(n=n, random_state=42).tolist()

        for txt in sampled_texts:
            for line in txt.splitlines():
                s=line.strip()
                if not s :
                    continue 
                if len(s)>120:
                    continue

                letters= sum(ch.isalpha() for ch in s )
                if letters < 5:
                    continue 
                line_counts[s]+=1
    top_lines = []
    for line, cnt in line_counts.most_common(30):
        # only keep suspiciously frequent lines
        if cnt >= max(20, int(0.01 * max(1, report["total_chunks"]))):
            top_lines.append({"line": line, "count": int(cnt)})

    report["top_repeated_lines"] = top_lines
    return report


def write_chunk_samples(df: pd.DataFrame, out_path: Path, k: int = 12, seed: int = 42) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if len(df) == 0:
        out_path.write_text("No chunks found.\n", encoding="utf-8")
        return

    k = min(k, len(df))
    sample_df = df.sample(n=k, random_state=seed)

    lines: List[str] = []
    for _, r in sample_df.iterrows():
        lines.append("=" * 90)
        lines.append(f"chunk_id: {r['chunk_id']}  doc_id: {r['doc_id']}")
        lines.append(f"source: {r['source']}")
        lines.append(f"title: {r['title']}")
        lines.append(f"section_path: {r.get('section_path','')}")
        lines.append(f"token_count: {int(r['token_count'])}  offsets: [{int(r['start_offset'])}, {int(r['end_offset'])})")
        lines.append("-" * 90)
        txt = str(r["chunk_text"])
        # truncate for readability
        if len(txt) > 900:
            txt = txt[:900] + "\n...[truncated]..."
        lines.append(txt)
        lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")

def check_chunks(df):
    
    REQUIRED = ["chunk_id", "chunk_text"]

    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if df["chunk_id"].isna().any():
        raise ValueError("chunk_id contains nulls")

    if df["chunk_id"].duplicated().any():
        dupes = df.loc[df["chunk_id"].duplicated(), "chunk_id"].head(10).tolist()
        raise ValueError(f"chunk_id has duplicates (sample): {dupes}")

    if df["chunk_text"].isna().any():
        raise ValueError("text contains nulls")

    empty_text = (df["chunk_text"].astype(str).str.strip() == "").sum()
    if empty_text > 0:
        raise ValueError(f"Found {empty_text} empty text chunks")
    
    print("Chunking validation passed")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="QA checks for RAG chunk corpus")
    p.add_argument(
        "--parquet",
        default="data/processed_chunks.parquet",
        help="Path to processed chunks parquet",
    )
    p.add_argument(
        "--out_json",
        default="eval/results/chunk_stats.json",
        help="Where to write QA report JSON",
    )
    p.add_argument(
        "--samples_path",
        default="eval/results/chunk_samples.txt",
        help="Where to write human-readable chunk samples",
    )

    # QA sampling knobs
    p.add_argument("--samples_k", type=int, default=12, help="How many chunks to sample")
    p.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
    p.add_argument(
        "--sample_limit_for_lines",
        type=int,
        default=5000,
        help="Max chunks to scan when detecting repeated boilerplate lines",
    )

    # Policy knobs (used only for threshold checks + reporting)
    p.add_argument("--policy_version", default="v1")
    p.add_argument("--target_tokens", type=int, default=550)
    p.add_argument("--overlap_tokens", type=int, default=80)
    p.add_argument("--min_tokens", type=int, default=200)
    p.add_argument("--max_tokens", type=int, default=750)

    # Optional “fail the run” toggles (CI-friendly)
    p.add_argument(
        "--fail_on_above_max",
        action="store_true",
        help="Exit non-zero if any chunk exceeds max_tokens",
    )
    p.add_argument(
        "--fail_on_non_unique_ids",
        action="store_true",
        help="Exit non-zero if chunk_id is not unique",
    )
    p.add_argument(
        "--fail_on_duplicates",
        action="store_true",
        help="Exit non-zero if any exact duplicate chunks exist (by checksum)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    parquet_path = Path(args.parquet)
    if not parquet_path.exists():
        print(f"ERROR: parquet not found: {parquet_path}", file=sys.stderr)
        raise SystemExit(2)

    df = pd.read_parquet(parquet_path)
    # print(df)
    check_chunks(df)

    policy = ChunkPolicy(
        policy_version=args.policy_version,
        target_tokens=args.target_tokens,
        overlap_tokens=args.overlap_tokens,
        min_tokens=args.min_tokens,
        max_tokens=args.max_tokens,
    )

    report = compute_qa_report(df, policy, sample_limit_for_lines=args.sample_limit_for_lines)

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")

    write_chunk_samples(df, Path(args.samples_path), k=args.samples_k, seed=args.seed)

    # Optional threshold-based failures
    exit_code = 0
    if args.fail_on_above_max and report.get("above_max_tokens", 0) > 0:
        exit_code = 1
    if args.fail_on_non_unique_ids and not report.get("chunk_id_unique", True):
        exit_code = 1
    if args.fail_on_duplicates and report.get("duplicate_by_checksum", 0) > 0:
        exit_code = 1

    # Small console summary
    print(
        f"docs={report.get('total_docs')} chunks={report.get('total_chunks')} "
        f"median={report.get('token_median')} p95={report.get('token_p95')} "
        f"above_max={report.get('above_max_tokens')} dup_checksum={report.get('duplicate_by_checksum')} "
        f"ids_unique={report.get('chunk_id_unique')}"
    )

    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()

