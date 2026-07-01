"""Save out-of-fold prediction pools (kNN + linear probe) per (model, cohort)
so that bootstrap CIs and PAIRED model/readout comparisons are fast and
reproducible (shared resample indices). Mirrors the fold logic of
paper_run_probe.py exactly.

Out: results/paper/pools/<model>_<suffix>.npz with
  labels [N,C] int8, knn [N,C] f32, probe [N,C] f32, label_columns, ids

Env: CTFM_MAXCELLS (chunk count; background jobs get reaped on this box).
Run inline chunks: CTFM_MAXCELLS=1 .venv/bin/python scripts/paper_save_pools.py
"""
from __future__ import annotations
import os, sys, time, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paper_patient_groups import ctrate_dedup_keep, group_array

R = Path(os.environ.get("CTFM_RESULTS", "results"))
EMB = R / "embeddings"
OUT = R / "paper" / "pools"
OUT.mkdir(parents=True, exist_ok=True)

MODELS = ["colipri-crm", "ctclip", "ctssg", "spectre-large", "flexict",
          "ctfm", "merlin", "pillar0", "curia2", "flexict3d", "voxelfm"]
COHORTS = [("radchest", "kfold"), ("ctrate", "group_kfold")]
K, SEED, NF = 5, 42, 5

# CT-CLIP's default (project=False) cache is the mean-pool of CTViT encoded
# tokens, which is pathologically anisotropic (effective rank ~4/512, off-diag
# cosine ~0.87) because the token grid shares a dominant common component. That
# cripples cosine-kNN and the L2-regularized probe. CT-CLIP's CANONICAL image
# embedding is the `to_visual_latent` projection (ct_clip.py:715-767) -- the same
# representation used for zero-shot -- cached as ctclip-zs (identical ids/labels/
# row order). Read kNN/probe from it; the pool is still saved under "ctclip".
# FlexiCT/SPECTRE were checked and need no swap (projected kNN delta <0.004).
CACHE_OVERRIDE = {"ctclip": "ctclip-zs"}


def kfold(n, k, seed):
    rng = np.random.default_rng(seed); perm = rng.permutation(n)
    folds = np.array_split(perm, k)
    for i in range(k):
        yield np.concatenate([folds[j] for j in range(k) if j != i]), folds[i]


def gkfold(groups, k, seed):
    groups = np.asarray(groups); n = len(groups)
    uniq = np.array(sorted(set(groups.tolist())))
    rng = np.random.default_rng(seed); perm = rng.permutation(uniq)
    buckets = [[] for _ in range(k)]
    for i, g in enumerate(perm):
        buckets[i % k].append(g)
    idx = np.arange(n)
    for i in range(k):
        tg = set(buckets[i]); mask = np.fromiter((g in tg for g in groups), bool, n)
        yield idx[~mask], idx[mask]


def knn_col(Xtr_n, ytr, Xte_n, k):
    sim = Xte_n @ Xtr_n.T
    ke = min(k, Xtr_n.shape[0])
    ti = np.argpartition(-sim, ke - 1, axis=1)[:, :ke]
    ts = np.take_along_axis(sim, ti, axis=1)
    w = np.clip(ts + 1.0, 1e-6, None); w /= w.sum(1, keepdims=True)
    return (ytr[ti] * w).sum(1)


def one(model, suffix, strategy):
    path = EMB / f"{CACHE_OVERRIDE.get(model, model)}_{suffix}.pt"
    if not path.exists():
        print(f"  MISSING {path.name}", flush=True); return False
    d = torch.load(path, map_location="cpu", weights_only=False)
    X = d["embeddings"].numpy().astype(np.float32)
    y = d["labels"].numpy().astype(np.float32)
    labs = list(d["label_columns"]); groups = d.get("groups")
    ids = d.get("ids"); n, C = X.shape[0], y.shape[1]
    Xn = X / np.linalg.norm(X, axis=1, keepdims=True).clip(min=1e-8)
    if suffix == "ctrate":
        # Keep one reconstruction per (patient, study), then patient-grouped folds
        # so a patient never spans train/test -- removes near-duplicate-recon leakage.
        ids_l = [str(x) for x in (ids if ids is not None else [])]
        keep = ctrate_dedup_keep(ids_l)
        X, y, Xn = X[keep], y[keep], Xn[keep]
        ids = [ids_l[i] for i in keep]
        groups = group_array("ctrate", ids)
        n, C = X.shape[0], y.shape[1]
        strategy = "group_kfold"
        print(f"  ctrate dedup: {len(ids_l)} -> {n} scans, {int(groups.max())+1} patients", flush=True)
    splits = list(gkfold(groups, NF, SEED) if strategy == "group_kfold" else kfold(n, NF, SEED))
    knn = np.zeros((n, C), np.float32); probe = np.zeros((n, C), np.float32)
    t0 = time.time()
    for tr, te in splits:
        Xtr_n, Xte_n, Xtr, Xte = Xn[tr], Xn[te], X[tr], X[te]
        for c in range(C):
            ytr = y[tr, c]
            knn[te, c] = knn_col(Xtr_n, ytr, Xte_n, K)
            npos = ytr.sum()
            if npos == 0 or npos == len(ytr):
                probe[te, c] = float(npos / max(len(ytr), 1)); continue
            clf = LogisticRegression(C=1.0, class_weight="balanced", solver="lbfgs", max_iter=2000)
            clf.fit(Xtr, ytr); probe[te, c] = clf.predict_proba(Xte)[:, 1]
    np.savez_compressed(OUT / f"{model}_{suffix}.npz",
                        labels=y.astype(np.int8), knn=knn, probe=probe,
                        label_columns=np.array(labs, dtype=object),
                        ids=np.array(ids if ids is not None else [], dtype=object))
    print(f"  saved {model}/{suffix} ({time.time()-t0:.0f}s)", flush=True)
    return True


def main():
    maxc = int(os.environ.get("CTFM_MAXCELLS", "0")); n = 0
    for suffix, strat in COHORTS:
        for m in MODELS:
            if (OUT / f"{m}_{suffix}.npz").exists():
                continue
            if maxc and n >= maxc:
                print("hit MAXCELLS", flush=True); return 0
            if one(m, suffix, strat):
                n += 1
    print("pools done", flush=True); return 0


if __name__ == "__main__":
    raise SystemExit(main())
