### Goals 
- Produce retrival friendly chunks 
- Preserve the definatinos, lists and section context. 
- Deterministic output same inputs + same policy -> same outputs 


### chunking type and parametes : 
- token based splitting 
- target_tokem= 550
- max_tokens = 800
- min_tokens= 200
- overlap = 80 
    we are using the above because a chunking of 500-600 tokens is usually good , works well with top k retrival, if k==5 then around 2500-3000 tokens. 

### Split Strategy : 
1. Respect document strucutre : 
    - headings -> keep heading with its content 
    - bullets -> try to keep bullets together
    - paragraphs -> split at paragraph boundries 
2. If split is too large : 
    - split on sentance boundaries
3. Last resort : 
    - split by tokens (hard split )

### Section-aware behavior
- Maintain a `section_path` like: "2 Methods / 2.3 Evaluation" when possible.
- Each chunk includes:
  - section heading (if available)
  - up to N leading lines of the section as context (optional, keep small)

### Tables / code blocks
- Tables:
  - preserve row/column structure as text (do not smash into one line)
  - if table is huge, split by rows with overlap of 1–2 rows
- Code blocks:
  - keep intact when possible
  - split only if exceeding max_tokens (split by logical blocks)

### Normalization rules
- Preserve newlines for bullets and tables.
- Collapse excessive whitespace, but do not remove meaningful line breaks.
- Remove repeated headers/footers if detected (common in PDFs).


### Metadata schema (per chunk)
Required fields:
- doc_id
- source (filename/url)
- title 
- section_path
- chunk_id
- start_offset, end_offset
- token_count
- chunk_index 
Optional : 
- page_start, page_end
- created_at
- checksum



### chunk_id definition (deterministic)
- policy_version: "v1"
- chunk_id = sha1(policy_version + "|" + doc_id + "|" + section_path + "|" +
                  start_offset + "|" + end_offset + "|" + checksum)[:16]

### Output artifact
- data/processed_chunks.parquet columns:
  - chunk_id, doc_id, source, title, section_path,
    start_offset, end_offset, token_count, chunk_text, (optional pdf page fields)

### QA checks (written to eval/results/chunk_stats.json)
- total_docs, total_chunks
- token_count distribution (min/median/p95/max)
- % chunks over max_tokens (should be ~0)
- duplicate rate (exact checksum duplicates)
- top repeated boilerplate lines (to detect headers/footers)
- sample 10 chunks printed for manual inspection

### Known failure modes + mitigations
- Topic drift: reduce target_tokens or enforce section boundary splits.
- Broken tables: switch to row-based table splitting.
- Missing definitions: increase overlap or ensure heading stays with body.