from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any,Dict,List

import numpy as np
import pandas as pd 
from tqdm import tqdm

from rag.utils.config_utils import load_yaml,stable_hash_dict
from rag.utils.paths import embeddings_dir
from rag.utils.helper_functions import _l2_normalize
from rag.embedding.embedding_provider import ProviderSpec
from rag.embedding.hf_embeddings import SentTransEmb
import argparse

# we are insuring a clean string with length less than max_chars (important for memeory and speed)
def _clean_text(s:str,max_chars:int)->str:
    s= "" if s is None else str(s)
    s=s.strip()
    if len(s) > max_chars:
        s=s[:max_chars]
    return s

# normalizing helper function for normalizing vectors and the query so that cosine similarity becomes inner product



#Sorting by chunk_id ensures stable alignment: row 0 in vectors always corresponds to the same chunk_id.
def _load_chunks(parquet_path:str) -> pd.DataFrame:
    df=pd.read_parquet(parquet_path,columns=["chunk_id","chunk_text"])
    df=df.rename(columns={
        "chunk_text":"text"
    })
    df=df.sort_values("chunk_id").reset_index(drop=True) 
    return df 

def _write_ids_jsonl(path:Path,chunk_ids:List[str])-> None:
    with open (path,'w',encoding="utf-8") as f:
        for cid in chunk_ids: 
            f.write(cid+"\n")

def main(chunks_path:str,cfg_path:str,out_dir:str)->None:
    
    cfg:Dict[str,Any]=load_yaml(cfg_path)
    emb_hash=stable_hash_dict(cfg)

    
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    vectors_path = out_dir / "vectors.npy"
    ids_path = out_dir / "chunk_ids.jsonl"
    meta_path = out_dir / "meta.json"

    # load the chunks 
    df=_load_chunks(chunks_path)
    chunk_ids=df["chunk_id"].astype(str).tolist()

    # get the texts from the chunks 
    max_chars=int(cfg.get("max_text_chars",8000))
    texts=[_clean_text(t,max_chars) for t in df['text'].tolist()]

    if str(cfg.get("model_name", "")).startswith("BAAI/bge"):
        prefix = "passage: "
        texts = [prefix + t[: max_chars - len(prefix)] for t in texts]

    # load the configs 
    batch_size=int(cfg.get("batch_size",64))
    normalize=bool(cfg.get("normalize",True))
    model_name=str(cfg.get("model_name"))
    if model_name is None or str(model_name).strip() == "" or str(model_name).lower() == "none":
        raise ValueError(f"Invalid model_name in {cfg_path}: {model_name}")
    
    # load the embedding model 
    provider=SentTransEmb(ProviderSpec(model_name))

    # here we are infering the dimension of the outputs by running a embeding, saving it into a temp file
    # this is being done because different models can have different embedding output dimensions 
    # Also to not blow the memory up we are using memmap to use only memory beings used to load the specific chunk of the file
    # we are readinf or writing to.
    t0=time.time()
    first_batch = texts[: min(batch_size, len(texts))]

    first_vecs=provider.embed_texts(first_batch)
    first_arr=np.array(first_vecs,dtype=np.float32)
    if first_arr.ndim !=2:
        raise ValueError(f"Unexpected embedding shape: {first_arr.shape}")
    # infering the dimension d 
    d=int(first_arr.shape[1])

    N=len(texts)
    tmp_path=out_dir/ "vectors.tmp.memmap"
    mm=np.memmap(tmp_path,dtype=np.float32,mode="w+",shape=(N,d))

    if normalize:
        first_arr=_l2_normalize(first_arr)
    mm[:first_arr.shape[0],:]=first_arr
    
    start=first_arr.shape[0]
    for i in tqdm(range(start,N,batch_size),desc="Embedding chunks"):
        batch = texts[i:i+batch_size]
        vecs=provider.embed_texts(batch)
        arr=np.array(vecs,dtype=np.float32)

        if arr.ndim!=2 or arr.shape[1]!=d:
            raise ValueError(f"Dim mismatch: expected (*, {d}), got {arr.shape}")

        if normalize:
            arr=_l2_normalize(arr)
        mm[i:i+arr.shape[0],:]=arr

    mm.flush()

    _write_ids_jsonl(ids_path, chunk_ids)

    final = np.asarray(mm)
    np.save(vectors_path, final)

    try:
        tmp_path.unlink(missing_ok=True)
    except Exception:
        pass

    meta = {
        "emb_hash": emb_hash,
        "model_name": model_name,
        "dim": d,
        "normalized": normalize,
        "num_chunks": N,
        "max_text_chars": max_chars,
        "batch_size": batch_size,
        "created_at_unix": int(time.time()),
        "source_chunks_path": chunks_path,
        "config": cfg,
        "embed_seconds": round(time.time() - t0, 3),
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"OK: wrote {N} embeddings of dim {d} to {out_dir}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--chunks_path",default="data//processed_chunks.parquet")
    p.add_argument("--configure_path",default="configs//embeddings.yaml")
    p.add_argument("--output_dir",default="data//embeddings")
    
    args=p.parse_args()

    main(args.chunks_path,args.configure_path,args.output_dir)
