from __future__ import annotations
import argparse,json,time,hashlib
from pathlib import Path
from typing import Any,Dict, Tuple
import numpy as np
import yaml 
import faiss
from rag.utils.config_utils import stable_hash_dict, load_yaml


# return the latest sub directory , sorting by folder names
def latest_subdir(root:Path)->Path:
    subs=[p for p in root.iterdir() if p.is_dir()]
    if not subs:
        raise FileNotFoundError(f"No subdirs under {root}")
    return sorted(subs,key=lambda p:p.name)[-1]

# loading the previously saved embeddings
def load_embeddings(emb_dir:Path) -> Tuple[np.ndarray,list[str],Dict[str,Any]]:
    meta=json.loads((emb_dir/"meta.json").read_text(encoding="utf-8"))
    vecs:np.ndarray=np.load(emb_dir/"vectors.npy")
    ids=(emb_dir/"chunk_ids.jsonl").read_text(encoding="utf-8").splitlines()

    # FAISS works best with float32 
    if vecs.dtype != np.float32:
        vecs=vecs.astype(np.float32)
    
    if vecs.shape[0] !=len(ids):
        raise ValueError(f"dimension mismatch between no. of chunks in  vector shape :{vecs.shape[0]} and in ids present : {len(ids)}")

    if len(ids) != len(set(ids)):
        raise ValueError("chunk_ids are not unique")

    if int(meta['dim']) != int(vecs.shape[1]):
        raise ValueError(f"dimension mismatch between embedding dimension between  vector shape :{vecs.shape[1]} and meta dimensions : {int(meta["dim"]) }")

    return vecs,ids,meta

def build_flat_ip_index(vecs:np.ndarray)->faiss.Index:
    d=vecs.shape[1]
    index=faiss.IndexFlatIP(d) # this mean flat indexing and search by inner product
    index.add(vecs)
    return index

def build_flat_l2_index(vecs:np.ndarray)->faiss.Index:
    d=vecs.shape[1]
    index=faiss.IndexFlatL2(d) # this means faiss flat indexing and L2 distance 
    index.add(vecs)
    return index

def main()->None:
    ap=argparse.ArgumentParser()
    ap.add_argument("--index_cfg",default="configs/index.yaml")
    ap.add_argument("--embeddings_root",default="data/embeddings")
    ap.add_argument("--emb_hash",default=None,help="If omitted, uses latest embeddings subdir.")
    args=ap.parse_args()

    index_cfg=load_yaml(args.index_cfg)

    emb_root=Path(args.embeddings_root)
    emb_dir=emb_root / args.emb_hash if args.emb_hash else latest_subdir(emb_root)

    vecs,chunk_ids,emb_meta=load_embeddings(emb_dir)

    # Bind index to the embeddings it was built from (prevents mixing runs)
    index_cfg_bound=dict(index_cfg)
    index_cfg_bound["emb_hash"]=emb_meta["emb_hash"]
    index_hash=stable_hash_dict(index_cfg_bound)

    out_dir=Path(index_cfg["storage"]["root_dir"])
    out_dir=out_dir/ index_hash
    out_dir.mkdir(parents=True,exist_ok=True)

    index_type=str(index_cfg.get("index_type","flat_ip")).lower()

    t0=time.time()
    if index_type=="flat_ip":
        index=build_flat_ip_index(vecs)
        metric="ip"
    elif index_type=="flat_l2":
        index=build_flat_l2_index(vecs)
        metric="l2"
    else:
        raise ValueError(f"Unsupported index_type : {index_type}")

    build_seconds=time.time()-t0

    index_path=out_dir/"faiss.index"
    faiss.write_index(index,str(index_path))

    # meta.json for index
    index_meta = {
        "index_hash": index_hash,
        "index_type": index_type,
        "metric": metric,
        "params": {},  # flat has no tunables
        "emb_hash": emb_meta["emb_hash"],
        "embedding_model": emb_meta["model_name"],
        "dim": int(vecs.shape[1]),
        "num_vectors": int(vecs.shape[0]),
        "created_at_unix": int(time.time()),
    }
    (out_dir / "meta.json").write_text(json.dumps(index_meta, indent=2), encoding="utf-8")

        # build report
    # For IndexFlat, memory is roughly N * d * 4 bytes for float32 storage
    approx_index_bytes = int(vecs.shape[0] * vecs.shape[1] * 4)

    report = {
        "index_hash": index_hash,
        "emb_hash": emb_meta["emb_hash"],
        "index_type": index_type,
        "num_vectors": int(vecs.shape[0]),
        "dim": int(vecs.shape[1]),
        "build_seconds": round(build_seconds, 4),
        "approx_index_bytes": approx_index_bytes,
        "index_path": str(index_path).replace("\\", "/"),
        "emb_dir": str(emb_dir).replace("\\", "/"),
    }

    results_dir=Path("eval/results")
    results_dir.mkdir(parents=True,exist_ok=True)
    (results_dir/f"index_build_{index_hash}.json").write_text(
        json.dumps(report,indent=2),encoding="utf-8"
    )

    print(f"OK : built {index_type} index at {index_path}")
    print(json.dumps(report,indent=2))

if __name__=="__main__":
    main()