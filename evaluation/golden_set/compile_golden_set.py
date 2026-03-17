import argparse, csv, json, os
from collections import defaultdict

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default="eval/data/labels.csv")
    ap.add_argument("--out", default="eval/data/golden_set.json")
    ap.add_argument("--bucket", default="answerable")
    args = ap.parse_args()

    by_qid = defaultdict(list)

    with open(args.labels, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            qid = row["qid"]
            by_qid[qid].append(row)

    golden = []
    for qid, rows in by_qid.items():
        # preserve CSV order
        rows.sort(key=lambda x: int(x["candidate_rank"]))

        query = rows[0]["query"]

        relevant = []
        best = None

        for row in rows:
            chunk_id = row["chunk_id"]
            rank0 = int(row["candidate_rank"]) - 1

            label = str(row.get("label", "")).strip()
            is_best = str(row.get("is_best", "")).strip()

            if label == "1":
                relevant.append({"row_index": rank0, "chunk_id": chunk_id})
            if is_best == "1":
                best = {"row_index": rank0, "chunk_id": chunk_id}

        # If no relevant labels, skip (or include with empty; skipping is typical for retrieval eval)
        if not relevant and best is None:
            continue

        # If best is not set but there are relevant chunks, set best = first relevant
        if best is None and relevant:
            best = relevant[0]

        # Ensure best is included in chunk_ids
        if best is not None:
            if all(x["chunk_id"] != best["chunk_id"] for x in relevant):
                relevant = [best] + relevant

        golden.append({
            "qid": qid,
            "bucket": args.bucket,
            "query": query,
            "best_chunk": best,
            "chunk_ids": relevant,
        })

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(golden, f, indent=2)

    print(f"Wrote {len(golden)} items to {args.out}")

if __name__ == "__main__":
    main()