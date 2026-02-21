from __future__ import annotations
import hashlib
import json
from pathlib import Path
from typing import Any,Dict
import yaml 

def load_yaml(path:str|Path)->Dict[str,Any]:
    with open(path,'r',encoding='utf-8') as f:
        return yaml.safe_load(f)

def stable_hash_dict(d:Dict[str,Any])->str:
    payload= json.dumps(d,sort_keys=True,separators=(",",":"),ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]

def ensure_dir(p:str|Path)-> Path:
    p=Path(p)
    p.mkdir(parents=True,exist_ok=True)
    return p