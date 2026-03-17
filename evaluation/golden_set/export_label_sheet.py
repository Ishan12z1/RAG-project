import csv
import json

csv_file = "evaluation/data/labels.csv"
txt_file = "evaluation/data/best_c_ids.txt"
output_file = "evaluation/data/golden_set_2.json"


def parse_rank_file(path):
    """
    Parses lines like:
    a001 : 13,14,15,1,3
    a002: 5,2,3,4,11

    Returns:
        {
            "a001": [13, 14, 15, 1, 3],
            "a002": [5, 2, 3, 4, 11],
            ...
        }
    """
    rank_map = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or ":" not in line:
                continue

            qid, ranks_part = line.split(":", 1)
            qid = qid.strip()

            ranks = []
            for x in ranks_part.split(","):
                x = x.strip()
                if x.isdigit():
                    ranks.append(int(x))

            rank_map[qid] = ranks

    return rank_map


def load_csv_rows(path):
    """
    Build:
      data[qid]["query"] -> query text
      data[qid]["ranks"][candidate_rank] -> chunk_id
    """
    data = {}

    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            qid = row["qid"].strip()
            query = row["query"]
            candidate_rank = int(row["candidate_rank"])
            chunk_id = row["chunk_id"]

            if qid not in data:
                data[qid] = {
                    "query": query,
                    "ranks": {}
                }

            data[qid]["ranks"][candidate_rank] = chunk_id

    return data


def build_output(csv_data, rank_map):
    output = []

    for qid, rank_list in rank_map.items():
        if qid not in csv_data or not rank_list:
            continue

        query = csv_data[qid]["query"]
        rank_to_chunk = csv_data[qid]["ranks"]

        # Keep only ranks that actually exist in the CSV
        selected_chunk_ids = []
        for rank in rank_list:
            if rank in rank_to_chunk:
                selected_chunk_ids.append({
                    "chunk_id": rank_to_chunk[rank]
                })

        if not selected_chunk_ids:
            continue

        obj = {
            "qid": qid,
            "bucket": "answerable",
            "query": query,
            "best_chunk": {
                "chunk_id": selected_chunk_ids[0]["chunk_id"]
            },
            "chunk_ids": selected_chunk_ids
        }

        output.append(obj)

    return output


if __name__ == "__main__":
    rank_map = parse_rank_file(txt_file)
    csv_data = load_csv_rows(csv_file)
    output_data = build_output(csv_data, rank_map)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"Wrote {output_file}")