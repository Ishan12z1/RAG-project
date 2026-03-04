# rebuild_parquet_from_manifest.py
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path, PureWindowsPath
from typing import Dict, List, Optional, Tuple

# ---- import your existing chunking pipeline ----
from rag.chunking.chunking import (
    TokenCounter,
    build_spans,
    compute_doc_id,
    compute_stats,
    expand_spans,
    extract_text_dispatch,
    guess_title,
    iter_raw_docs,
    normalize_text,
    pack_spans_into_chunks,
    process_one_doc,
    write_json,
    write_parquet,
)
from rag.utils.contracts import ChunkPolicy, ChunkRow, RawDoc


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def manifest_key_from_path(p: Path, corpus_root: Path) -> str:
    rel = p.relative_to(corpus_root.parent)  # strips "data/"
    return normalize_key(str(rel))

def normalize_key(s: str) -> str:
    s = s.strip().replace("/", "\\")
    return str(PureWindowsPath(s))

def read_manifest_jsonl(path: Path) -> Dict[str, dict]:
    """
    Map normalized saved_path -> manifest record.
    Tolerant: skips bad lines.
    """
    out: Dict[str, dict] = {}
    if not path.exists():
        return out

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            sp = rec.get("saved_path")
            if sp:
                out[str(Path(sp))] = rec  # normalize slashes
    return out


def discover_existing_docs(
    corpus_root: Path,
    include_exts: Tuple[str, ...] = (".pdf", ".html", ".htm", ".txt", ".md", ".docx"),
) -> List[RawDoc]:
    """
    Disk is source of truth. This means:
    - deleted files in manifest won't break anything
    - new files/folders automatically included
    """
    docs: List[RawDoc] = []
    for p in sorted(corpus_root.rglob("*")):
        if not p.is_file():
            continue
        if p.suffix.lower() not in include_exts:
            continue
        rel = str(p.relative_to(corpus_root))
        docs.append(RawDoc(path=p, source=rel, doc_id=compute_doc_id(p)))
    return docs


def build_chunks_from_disk(
    corpus_root: Path,
    out_parquet: Path,
    stats_json: Path,
    policy: ChunkPolicy,
    corpus_mode: str = "generic",
    manifest_path: Optional[Path] = None,
) -> None:
    """
    Produces the SAME parquet schema as your ChunkRow dataclass:
      chunk_id, doc_id, source, title, url, section_path, chunk_index,
      start_offset, end_offset, token_count, checksum, chunk_text
    """

    # Optional: read manifest just to report what's missing / extra (no schema changes)
    manifest_raw = read_manifest_jsonl(manifest_path) if manifest_path else {}
    manifest = {normalize_key(k): v for k, v in manifest_raw.items()}
    tc = TokenCounter()
    docs = discover_existing_docs(corpus_root)

    # Optional reporting (doesn't affect output)
    if manifest:
        manifest_paths = set(manifest.keys())
        disk_paths = {str(Path(corpus_root / d.source)) for d in docs}
        missing_on_disk = sorted(manifest_paths - disk_paths)
        if missing_on_disk:
            print(f"[info] manifest entries missing on disk: {len(missing_on_disk)}")
        extra_on_disk = sorted(disk_paths - manifest_paths)
        if extra_on_disk:
            print(f"[info] files on disk not in manifest: {len(extra_on_disk)}")

    all_rows: List[ChunkRow] = []
    for d in docs:
        try:
            key = manifest_key_from_path(d.path,corpus_root)
            rec = manifest.get(key, {})  # now it should hit
            metadata = {
                "url": rec.get("source_url"),
                "title": rec.get("title") or d.path.stem,
            }
            rows = process_one_doc(d, policy, tc, corpus_mode=corpus_mode,metadata=metadata)

            all_rows.extend(rows)
        except Exception as e:
            # Do not crash because one file is bad; skip it and continue.
            # If you prefer "fail fast", replace this with: raise
            print(f"[warn] failed processing {d.source}: {type(e).__name__}: {e}")

    write_parquet(all_rows, out_parquet)
    stats = compute_stats(all_rows)
    # Add a couple of run-level fields without affecting chunk parquet schema
    stats["_built_at"] = _utcnow_iso()
    stats["_corpus_root"] = str(corpus_root)
    write_json(stats, stats_json)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--corpus_root", default="data/diabetes_canada_corpus")
    p.add_argument("--manifest", default="data/diabetes_canada_corpus/manifest.jsonl")
    p.add_argument("--out_parquet", default="data//run_2//processed_chunks.parquet")
    p.add_argument("--stats_json", default="data//run_2//chunk_stats.json")
    p.add_argument("--corpus_mode", choices=["generic", "kilt"], default="generic")

    # same policy args as your chunking.py
    p.add_argument("--policy_version", default="v1")
    p.add_argument("--target_tokens", type=int, default=550)
    p.add_argument("--overlap_tokens", type=int, default=80)
    p.add_argument("--min_tokens", type=int, default=200)
    p.add_argument("--max_tokens", type=int, default=750)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Match your KILT defaults exactly
    if args.corpus_mode == "kilt":
        if args.target_tokens == 550:
            args.target_tokens = 280
        if args.overlap_tokens == 80:
            args.overlap_tokens = 60
        if args.min_tokens == 200:
            args.min_tokens = 100
        if args.max_tokens == 750:
            args.max_tokens = 420

    policy = ChunkPolicy(
        policy_version=args.policy_version,
        target_tokens=args.target_tokens,
        overlap_tokens=args.overlap_tokens,
        min_tokens=args.min_tokens,
        max_tokens=args.max_tokens,
    )

    build_chunks_from_disk(
        corpus_root=Path(args.corpus_root),
        out_parquet=Path(args.out_parquet),
        stats_json=Path(args.stats_json),
        policy=policy,
        corpus_mode=args.corpus_mode,
        manifest_path=Path(args.manifest) if args.manifest else None,
    )


if __name__ == "__main__":
    main()