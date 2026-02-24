import json, numpy as np

meta = json.load(open("data/embeddings/2954117bb965/meta.json", "r", encoding="utf-8"))
vecs = np.load("data/embeddings/2954117bb965/vectors.npy")
ids  = open("data/embeddings/2954117bb965/chunk_ids.jsonl", "r", encoding="utf-8").read().splitlines()

print("vecs shape:", vecs.shape, "dtype:", vecs.dtype)
print("ids:", len(ids), "unique:", len(set(ids)))
print("meta dim:", meta["dim"], "meta num_chunks:", meta["num_chunks"])

assert vecs.dtype == np.float32
assert vecs.shape[0] == len(ids) == meta["num_chunks"]
assert vecs.shape[1] == meta["dim"]
assert len(ids) == len(set(ids))   # chunk_id must be unique

if meta["normalized"]:
    norms = np.linalg.norm(vecs, axis=1)
    print("norm min/max:", norms.min(), norms.max())