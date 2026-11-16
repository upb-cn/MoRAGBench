"""Tie-aware recall for ANN results (numpy only).

Plain recall compares item IDs. Datasets with duplicate or equally distant vectors
(e.g. ny_times) then count a correct answer as a miss whenever a different item with
the *same* distance is returned. ann-benchmarks avoids this by comparing distances:
a returned item counts if its true distance to the query is <= the distance of the
k-th true neighbour + epsilon (epsilon = 1e-3, as in ann-benchmarks).

The true distance of every returned item is recomputed here from the original vectors,
so this works for every index type, including binary ones that report Hamming distances.
"""
import numpy as np

EPSILON = 1e-3


def _load(dataset_dir, meta, name):
    m = meta[name]
    return np.fromfile(f"{dataset_dir}/{m['path']}", dtype=m["dtype"]).reshape(m["shape"])


def tie_aware_recall(neighbors, dataset_dir, meta, angular, ks=(1, 10)):
    """neighbors: (n_queries, k) array of returned ids (-1 = no result).
    angular: True for normalised/angular datasets (distance = 1 - dot), False for euclidean."""
    train = np.memmap(f"{dataset_dir}/{meta['train']['path']}", dtype=meta["train"]["dtype"],
                      mode="r", shape=tuple(meta["train"]["shape"]))
    test = _load(dataset_dir, meta, "test")
    gt_dist = _load(dataset_dir, meta, "distances")
    neighbors = np.asarray(neighbors)
    nq, k_ret = neighbors.shape
    out = {}
    for k in ks:
        if k > k_ret:
            continue
        ids = neighbors[:, :k]
        valid = ids >= 0
        vecs = train[np.where(valid, ids, 0)]                      # (nq, k, dim)
        q = test[:nq, None, :]
        if angular:
            d = 1.0 - np.einsum("qkd,qkd->qk", vecs, np.broadcast_to(q, vecs.shape))
        else:
            d = np.linalg.norm(vecs - q, axis=2)
        threshold = gt_dist[:nq, k - 1][:, None] + EPSILON
        hits = (d <= threshold) & valid & np.isfinite(d)
        # queries whose ground truth itself is undefined (e.g. all-zero vectors) are skipped
        ok = np.isfinite(gt_dist[:nq, k - 1])
        out[str(k)] = float(hits[ok].sum(axis=1).mean() / k)
        out[f"skipped_queries@{k}"] = int((~ok).sum())
    return out
