# eval/ladder.py
from __future__ import annotations

import csv
import os
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Dict, Optional


LADDER_PATH = os.path.join("eval", "results", "ladder.csv")

# Stable column order. Add columns here (only here).
LADDER_COLUMNS = [
    "timestamp_utc",
    "run_tag",
    "notes",
    "recall_at_5",
    "mrr",
    "unsupported_rate",
    "abstain_rate",
    "p50_ms",
    "p95_ms",
]

def _ensure_parent_dir(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)

def append_ladder_row(
    run_tag: str,
    metrics: Dict[str, float],
    notes: str = "",
    ladder_path: str = LADDER_PATH,
) -> None:
    """
    metrics keys should match LADDER_COLUMNS entries (excluding timestamp/run_tag/notes).
    Missing metrics are written as empty cells.
    Extra metrics are ignored (do not auto-add columns).
    """
    _ensure_parent_dir(ladder_path)

    row = {k: "" for k in LADDER_COLUMNS}
    row["timestamp_utc"] = datetime.now(timezone.utc).isoformat()
    row["run_tag"] = run_tag
    row["notes"] = notes

    for k, v in metrics.items():
        if k in row:
            row[k] = v

    file_exists = os.path.exists(ladder_path)
    write_header = (not file_exists) or (os.path.getsize(ladder_path) == 0)

    with open(ladder_path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=LADDER_COLUMNS)
        if write_header:
            w.writeheader()
        w.writerow(row)