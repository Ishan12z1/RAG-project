# rag/chunking.py
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd


@dataclass(frozen=True)
class ChunkPolicy:
    policy_version: str = "v1"
    target_tokens: int = 550
    overlap_tokens: int = 80
    min_tokens: int = 200
    max_tokens: int = 750

@dataclass(frozen=True)
class RawDoc:
    path: Path
    source: str          # relative path string
    doc_id: str          # stable hash of file content


@dataclass(frozen=True)
class ExtractedDoc:
    raw: RawDoc
    title: str
    normalized_text: str
    page_texts: Optional[List[str]] = None  # only set for PDFs

@dataclass(frozen=True)
class Span:
    start: int
    end: int
    section_path: str
    kind: str  # "heading", "para", "bullets", "table", "other"

@dataclass(frozen=True)
class ChunkRow:
    chunk_id: str
    doc_id: str
    source: str
    title: str
    section_path: str
    chunk_index: int
    start_offset: int
    end_offset: int
    token_count: int
    checksum: str
    chunk_text: str
    page_start: Optional[int] = None
    page_end: Optional[int] = None


def sha1_bytes(b: bytes) -> str:
    return hashlib.sha1(b).hexdigest()


def compute_doc_id(path: Path) -> str:
    data = path.read_bytes()
    return sha1_bytes(data)


def iter_raw_docs(input_dir: Path) -> List[RawDoc]:
    files: List[Path] = []
    for p in input_dir.rglob("*"):
        if p.is_file():
            files.append(p)

    files.sort(key=lambda x: str(x.relative_to(input_dir)).lower())

    out: List[RawDoc] = []
    for p in files:
        rel = str(p.relative_to(input_dir))
        out.append(RawDoc(path=p, source=rel, doc_id=compute_doc_id(p)))
    return out

class TokenCounter:
    def __init__(self):
        self._enc=None
        try: 
            import tiktoken
            self._enc=tiktoken.get_encoding("cl100k_base")
        except Exception:
            self._enc=None
    
    def count(self,text:str) -> int:
        if self._enc:
            return len(self._enc.encode(text))
        # Approx fallback: words plus punctuation as a proxy
        words = re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE)
        return int(len(words) * 1.0)

    def count_many(self, texts: Sequence[str]) -> int:
        return sum(self.count(t) for t in texts)


def extract_text_dispatch(doc: RawDoc) -> Tuple[str, Optional[List[str]]]:
    ext = doc.path.suffix.lower()

    if ext in {".txt", ".md"}:
        return extract_text_plain(doc.path), None
    if ext == ".docx":
        return extract_text_docx(doc.path), None
    if ext in {".html", ".htm"}:
        return extract_text_html(doc.path), None
    if ext == ".pdf":
        text, pages = extract_text_pdf(doc.path)
        return text, pages

    raise ValueError(f"Unsupported file type: {ext} for {doc.source}")

def extract_text_plain(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")

def extract_text_docx(path: Path) -> str:
    from docx import Document  # python-docx is installed

    d = Document(str(path))
    parts: List[str] = []
    for para in d.paragraphs:
        t = para.text.strip()
        if t:
            parts.append(t)
    return "\n\n".join(parts)

def extract_text_html(path: Path) -> str:
    raw = path.read_text(encoding="utf-8", errors="ignore")
    try:
        from bs4 import BeautifulSoup  # type: ignore
        soup = BeautifulSoup(raw, "html.parser")
        text = soup.get_text("\n")
        return text
    except Exception:
        # Fallback: strip tags roughly
        text = re.sub(r"<[^>]+>", " ", raw)
        text = re.sub(r"\s+", " ", text)
        return text

def extract_text_pdf(path: Path) -> Tuple[str, List[str]]:
    pages: List[str] = []

    # Try pypdf first, then PyPDF2
    reader = None
    try:
        from pypdf import PdfReader  # type: ignore
        reader = PdfReader(str(path))
    except Exception:
        try:
            from PyPDF2 import PdfReader  # type: ignore
            reader = PdfReader(str(path))
        except Exception as e:
            raise RuntimeError(
                "PDF extraction requires pypdf or PyPDF2. Install one of them."
            ) from e

    for page in reader.pages:
        t = page.extract_text() or ""
        pages.append(t)

    full = "\n\n".join(pages)
    return full, pages

_BULLET_RE = re.compile(r"^\s*([-*•]|(\d+[\.\)]))\s+")
_TABLE_LIKE_RE = re.compile(r".*\|.*\|.*")  # simple markdown table heuristic


def normalize_text(text: str) -> str:
    t = text.replace("\r\n", "\n").replace("\r", "\n")

    # Trim trailing spaces on each line, preserve line breaks
    lines = [ln.rstrip() for ln in t.split("\n")]

    # Collapse excessive blank lines (keep at most 2)
    out_lines: List[str] = []
    blank_run = 0
    for ln in lines:
        if ln.strip() == "":
            blank_run += 1
            if blank_run <= 2:
                out_lines.append("")
        else:
            blank_run = 0
            out_lines.append(ln)

    t2 = "\n".join(out_lines).strip()
    return t2 + "\n"  # ensure trailing newline for offset math stability


_MD_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$")


def guess_title(source: str, normalized_text: str) -> str:
    for ln in normalized_text.splitlines():
        s = ln.strip()
        if not s:
            continue
        m = _MD_HEADING_RE.match(s)
        if m:
            return m.group(2).strip()
        if len(s) <= 120:
            return s
        break
    return Path(source).stem


def is_heading_line(line: str) -> Optional[str]:
    s = line.strip()
    if not s:
        return None

    m = _MD_HEADING_RE.match(s)
    if m:
        return m.group(2).strip()

    # Simple heuristic: short-ish, no ending punctuation, mostly title case or caps
    if len(s) <= 80 and not s.endswith((".", "!", "?", ";")):
        letters = sum(ch.isalpha() for ch in s)
        if letters >= 8:
            caps = sum(ch.isupper() for ch in s if ch.isalpha())
            if caps / max(1, letters) > 0.6:
                return s
    return None

# 
def build_spans(normalized_text: str) -> List[Tuple[int, int, str]]:
    lines = normalized_text.splitlines(keepends=True)

    # For each line, track its start offset in the full string
    starts: List[int] = []
    pos = 0
    for ln in lines:
        starts.append(pos)
        pos += len(ln)

    section_stack: List[str] = []
    spans: List[Span] = []

    i = 0
    while i < len(lines):
        ln_raw = lines[i]
        ln = ln_raw.rstrip("\n")
        ln_stripped = ln.strip()

        # Blank line
        if ln_stripped == "":
            i += 1
            continue

        # Heading line
        heading = is_heading_line(ln)
        if heading is not None:
            section_stack = section_stack[:3]  # keep path short
            section_stack.append(heading)
            sec = " / ".join(section_stack)
            s = starts[i]
            e = s + len(ln_raw)
            spans.append(Span(s, e, sec, "heading"))
            i += 1
            continue

        sec = " / ".join(section_stack) if section_stack else ""

        # Bullet block
        if _BULLET_RE.match(ln):
            s = starts[i]
            j = i
            while j < len(lines):
                ln_j = lines[j].rstrip("\n")
                if ln_j.strip() == "":
                    break
                if _BULLET_RE.match(ln_j) or ln_j.startswith(" " * 2):
                    j += 1
                else:
                    break
            e = starts[j - 1] + len(lines[j - 1])
            spans.append(Span(s, e, sec, "bullets"))
            i = j
            continue

        # Table-like block (markdown pipes)
        if _TABLE_LIKE_RE.match(ln):
            s = starts[i]
            j = i
            while j < len(lines):
                ln_j = lines[j].rstrip("\n")
                if ln_j.strip() == "":
                    break
                if _TABLE_LIKE_RE.match(ln_j):
                    j += 1
                else:
                    break
            e = starts[j - 1] + len(lines[j - 1])
            spans.append(Span(s, e, sec, "table"))
            i = j
            continue

        # Paragraph block until blank line
        s = starts[i]
        j = i
        while j < len(lines) and lines[j].strip() != "":
            j += 1
        e = starts[j - 1] + len(lines[j - 1])
        spans.append(Span(s, e, sec, "para"))
        i = j

    return spans


_SENT_BOUNDARY_RE = re.compile(r"(?<=[\.\!\?])\s+")


def split_span_if_needed(
    text: str,
    span: Span,
    policy: ChunkPolicy,
    tc: TokenCounter
) -> List[Span]:
    chunk_text = text[span.start:span.end]
    if tc.count(chunk_text) <= policy.max_tokens:
        return [span]

    # Try sentence split for paragraphs
    if span.kind in {"para", "other"}:
        return split_span_by_sentences(text, span, policy, tc)

    # Bullets and tables: hard split (preserve lines as much as possible)
    return hard_split_span(text, span, policy, tc)


def split_span_by_sentences(
    text: str,
    span: Span,
    policy: ChunkPolicy,
    tc: TokenCounter
) -> List[Span]:
    s = text[span.start:span.end]
    parts = _SENT_BOUNDARY_RE.split(s)

    # If splitting fails (one massive sentence), fallback
    if len(parts) <= 1:
        return hard_split_span(text, span, policy, tc)

    spans: List[Span] = []
    local_pos = 0
    cur_start = span.start
    cur_end = span.start
    cur_buf: List[str] = []

    def flush(buf: List[str], start_off: int, end_off: int) -> None:
        if start_off < end_off:
            spans.append(Span(start_off, end_off, span.section_path, span.kind))

    for p in parts:
        if not p:
            continue
        # Find this part in the remaining substring
        idx = s.find(p, local_pos)
        if idx < 0:
            continue
        part_start = span.start + idx
        part_end = part_start + len(p)

        candidate_buf = cur_buf + [p]
        candidate_text = " ".join(candidate_buf).strip()
        if tc.count(candidate_text) > policy.max_tokens and cur_buf:
            flush(cur_buf, cur_start, cur_end)
            cur_buf = [p]
            cur_start = part_start
            cur_end = part_end
        else:
            cur_buf = candidate_buf
            cur_end = part_end

        local_pos = idx + len(p)

    flush(cur_buf, cur_start, cur_end)

    # If still too big, hard split those spans
    final: List[Span] = []
    for sp in spans:
        if tc.count(text[sp.start:sp.end]) <= policy.max_tokens:
            final.append(sp)
        else:
            final.extend(hard_split_span(text, sp, policy, tc))
    return final


def hard_split_span(
    text: str,
    span: Span,
    policy: ChunkPolicy,
    tc: TokenCounter
) -> List[Span]:
    s = text[span.start:span.end]
    tok = tc.count(s)
    if tok <= policy.max_tokens:
        return [span]

    # Estimate chars per token, split by char window
    chars_per_token = max(1.0, len(s) / max(1, tok))
    max_chars = int(chars_per_token * policy.max_tokens)

    out: List[Span] = []
    local = 0
    while local < len(s):
        cut = min(len(s), local + max_chars)

        # Prefer cutting at whitespace
        if cut < len(s):
            ws = s.rfind(" ", local, cut)
            if ws > local + int(max_chars * 0.6):
                cut = ws

        start_off = span.start + local
        end_off = span.start + cut
        out.append(Span(start_off, end_off, span.section_path, span.kind))
        local = cut

    return out

def expand_spans(
    normalized_text: str,
    spans: List[Span],
    policy: ChunkPolicy,
    tc: TokenCounter
) -> List[Span]:
    out: List[Span] = []
    for sp in spans:
        out.extend(split_span_if_needed(normalized_text, sp, policy, tc))
    return out


def pack_spans_into_chunks(
    normalized_text: str,
    spans: List[Span],
    policy: ChunkPolicy,
    tc: TokenCounter
) -> List[Tuple[int, int, str, int]]:
    """
    Returns list of tuples:
    (start_offset, end_offset, section_path, token_count)
    """
    chunks: List[Tuple[int, int, str, int]] = []
    n = len(spans)
    i = 0

    while i < n:
        start_i = i
        start_off = spans[i].start
        section_path = spans[i].section_path

        cur_end = spans[i].end
        cur_tokens = 0

        j = i
        while j < n:
            cand_start = spans[start_i].start
            cand_end = spans[j].end
            cand_text = normalized_text[cand_start:cand_end]
            cand_tokens = tc.count(cand_text)

            if cand_tokens <= policy.max_tokens and cand_tokens <= policy.target_tokens:
                cur_end = cand_end
                cur_tokens = cand_tokens
                j += 1
                continue

            # Allow one overshoot up to max_tokens if still reasonable
            if cand_tokens <= policy.max_tokens and cur_tokens == 0:
                cur_end = cand_end
                cur_tokens = cand_tokens
                j += 1
            break

        if cur_tokens == 0:
            # Fallback: at least emit one span
            cur_end = spans[i].end
            cur_tokens = tc.count(normalized_text[start_off:cur_end])
            j = i + 1

        chunks.append((start_off, cur_end, section_path, cur_tokens))

        # Compute overlap: move i forward, but keep last spans covering overlap_tokens
        if j >= n:
            break

        # Find earliest index k within [start_i, j) such that tokens(spans[k:j]) >= overlap_tokens
        k = j - 1
        overlap_tokens = 0
        while k > start_i:
            seg_text = normalized_text[spans[k].start:spans[j - 1].end]
            overlap_tokens = tc.count(seg_text)
            if overlap_tokens >= policy.overlap_tokens:
                break
            k -= 1

        i = k

        # Safety to avoid infinite loops when overlap equals the whole chunk
        if i <= start_i:
            i = start_i + 1

    return chunks

def compute_chunk_id(
    policy_version: str,
    doc_id: str,
    section_path: str,
    start_offset: int,
    end_offset: int,
    chunk_text: str
) -> Tuple[str, str]:
    checksum = sha1_bytes(chunk_text.encode("utf-8", errors="ignore"))
    base = f"{policy_version}|{doc_id}|{section_path}|{start_offset}|{end_offset}|{checksum}"
    cid = sha1_bytes(base.encode("utf-8"))[:16]
    return cid, checksum
def write_parquet(rows: List[ChunkRow], out_path: Path) -> None:
    df = pd.DataFrame([r.__dict__ for r in rows])
    out_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        df.to_parquet(out_path, index=False)
    except Exception as e:
        raise RuntimeError(
            "Parquet write failed. Install pyarrow or fastparquet."
        ) from e


def compute_stats(rows: List[ChunkRow]) -> Dict:
    token_counts = [r.token_count for r in rows]
    token_counts_sorted = sorted(token_counts)

    def pct(p: float) -> int:
        if not token_counts_sorted:
            return 0
        idx = int(round((p / 100.0) * (len(token_counts_sorted) - 1)))
        return token_counts_sorted[max(0, min(len(token_counts_sorted) - 1, idx))]

    checksums = [r.checksum for r in rows]
    dup = len(checksums) - len(set(checksums))

    docs = len(set(r.doc_id for r in rows))

    return {
        "total_docs": docs,
        "total_chunks": len(rows),
        "token_count_min": min(token_counts) if token_counts else 0,
        "token_count_median": pct(50),
        "token_count_p95": pct(95),
        "token_count_max": max(token_counts) if token_counts else 0,
        "duplicate_chunks_exact": dup,
    }


def write_json(obj: Dict, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def process_one_doc(
    doc: RawDoc,
    policy: ChunkPolicy,
    tc: TokenCounter
) -> List[ChunkRow]:
    raw_text, page_texts = extract_text_dispatch(doc)
    normalized = normalize_text(raw_text)
    title = guess_title(doc.source, normalized)

    spans = build_spans(normalized)
    spans = expand_spans(normalized, spans, policy, tc)

    packed = pack_spans_into_chunks(normalized, spans, policy, tc)

    rows: List[ChunkRow] = []
    for idx, (s, e, section_path, tok) in enumerate(packed):
        chunk_text = normalized[s:e].strip()
        if not chunk_text:
            continue

        # Optional: suppress tiny chunks unless it is a heading-only chunk
        if tok < policy.min_tokens and len(chunk_text.splitlines()) > 1:
            continue

        chunk_id, checksum = compute_chunk_id(
            policy.policy_version, doc.doc_id, section_path, s, e, chunk_text
        )

        rows.append(
            ChunkRow(
                chunk_id=chunk_id,
                doc_id=doc.doc_id,
                source=doc.source,
                title=title,
                section_path=section_path,
                chunk_index=idx,
                start_offset=s,
                end_offset=e,
                token_count=tok,
                checksum=checksum,
                chunk_text=chunk_text,
            )
        )

    return rows


def build_chunks_corpus(
    input_dir: Path,
    out_parquet: Path,
    stats_json: Path,
    policy: ChunkPolicy
) -> None:
    tc = TokenCounter()
    docs = iter_raw_docs(input_dir)

    all_rows: List[ChunkRow] = []
    for d in docs:
        try:
            all_rows.extend(process_one_doc(d, policy, tc))
        except Exception as e:
            raise RuntimeError(f"Failed processing {d.source}: {e}") from e

    write_parquet(all_rows, out_parquet)
    stats = compute_stats(all_rows)
    write_json(stats, stats_json)



def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input_dir", default="data/raw_docs")
    p.add_argument("--out_parquet", default="data/processed_chunks.parquet")
    p.add_argument("--stats_json", default="eval/results/chunk_stats.json")

    p.add_argument("--policy_version", default="v1")
    p.add_argument("--target_tokens", type=int, default=550)
    p.add_argument("--overlap_tokens", type=int, default=80)
    p.add_argument("--min_tokens", type=int, default=200)
    p.add_argument("--max_tokens", type=int, default=750)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    policy = ChunkPolicy(
        policy_version=args.policy_version,
        target_tokens=args.target_tokens,
        overlap_tokens=args.overlap_tokens,
        min_tokens=args.min_tokens,
        max_tokens=args.max_tokens,
    )

    build_chunks_corpus(
        input_dir=Path(args.input_dir),
        out_parquet=Path(args.out_parquet),
        stats_json=Path(args.stats_json),
        policy=policy,
    )


if __name__ == "__main__":
    main()

