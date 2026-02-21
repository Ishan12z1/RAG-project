import numpy as np 
from pathlib import Path
import json
from typing import Dict,Any

def _l2_normalize(mat:np.ndarray)->np.ndarray:
    norms=np.linalg.norm(mat,axis=1,keepdims=True)
    norms=np.clip(norms,1e-12,None) # we are clipping to prevent divide by zero error 
    return mat/norms

def read_meta(path:str|Path)->Dict[str,Any]:
    return json.loads((Path(path) / "meta.json").read_text(encoding="utf-8"))