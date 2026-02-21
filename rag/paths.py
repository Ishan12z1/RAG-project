from __future__ import annotations
from pathlib import Path
from datetime import datetime, timezone
from rag.utils.config_utils import ensure_dir

def embeddings_dir(root:str, emb_hash:str)->Path:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return ensure_dir(Path(root) / f"{ts}__{emb_hash}")

def index_dir(root:str,index_hash:str)->Path:
    return ensure_dir(Path(root)/index_hash)