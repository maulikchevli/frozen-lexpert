"""Full linear-probe vs kNN matrix for the transfer-science paper.

For each (model, cohort, label) runs kNN (k=5 cosine distance-wt)
and linear probe (LogReg L2 C=1 balanced) under IDENTICAL folds (5-fold;
group_kfold for grouped cohorts via `groups`), pools out-of-fold predictions,
computes per-label PR-AUC + prevalence, and (optionally) a paired bootstrap CI.

Env:
  CTFM_NBOOT   number of bootstrap resamples (default 0 = point only, fast)

Output (resumable, appended per (model,cohort)):
  results/paper/probe_vs_knn_full.csv

Run:
  .venv/bin/python scripts/paper_run_probe.py            # point estimates
  CTFM_NBOOT=1000 .venv/bin/python scripts/paper_run_probe.py  # + CIs
"""
from __future__ import annotations

import csv
import os
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paper_patient_groups import ctrate_dedup_keep, group_array

HERE = Path(os.environ.get("CTFM_RESULTS", "results"))
EMB_DIR = HERE / "embeddings"
OUT_CSV = HERE / "paper" / "probe_vs_knn_full.csv"

MODELS = ["colipri-crm", "ctclip", "ctssg", "spectre-large", "flexict",
          "ctfm", "merlin", "pillar0", "curia2", "flexict3d", "voxelfm"]

# See paper_save_pools.py: CT-CLIP's avg-pooled token cache is degenerate; read
# kNN/probe from its canonical projection (ctclip-zs), saved under "ctclip".
CACHE_OVERRIDE = {"ctclip": "ctclip-zs"}

# CT-RATE uses patient-grouped folds (with one-reconstruction-per-study dedup),
# matching paper_save_pools.py and the paper's stated 5-fold patient-grouped CV.
COHORTS = [
    ("RadChestCT",   "radchest", "kfold"),
    ("CT-RATE",      "ctrate",   "group_kfold"),
]

K, WEIGHT, SEED, N_FOLDS = 5, "distance", 42, 5
PROBE_C, PROBE_MAX_ITER = 1.0, 2000
N_BOOT = int(os.environ.get("CTFM_NBOOT", "0"))


def kfold_splits(n: int, k: int, seed: int):
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    folds = np.array_split(perm, k)
    for i in range(k):
        test = folds[i]
        train = np.concatenate([folds[j] for j in range(k) if j != i])
        yield train, test


def group_kfold_splits(groups, k: int, seed: int):
    groups = np.asarray(groups)
    n = len(groups)
    if n == 0:
        return
    uniq = np.array(sorted(set(groups.tolist())))
    rng = np.random.default_rng(seed)
    perm = rng.permutation(uniq)
    buckets = [[] for _ in range(k)]
    for i, g in enumerate(perm):
        buckets[i % k].append(g)
    all_idx = np.arange(n)
    for i in range(k):
        test_groups = set(buckets[i])
        test_mask = np.fromiter((g in test_groups for g in groups),
                                dtype=bool, count=n)
        yield all_idx[~test_mask], all_idx[test_mask]


def knn_probs_one_class(Xtr_n, ytr, Xte_n, k, weighting):
    sim = Xte_n @ Xtr_n.T
    k_eff = min(k, Xtr_n.shape[0])
    top_idx = np.argpartition(-sim, k_eff - 1, axis=1)[:, :k_eff]
    top_sim = np.take_along_axis(sim, top_idx, axis=1)
    if weighting == "distance":
        w = np.clip(top_sim + 1.0, 1e-6, None)
        w = w / w.sum(axis=1, keepdims=True)
    else:
        w = np.full_like(top_sim, 1.0 / k_eff)
    return (ytr[top_idx] * w).sum(axis=1)


def run_one_cohort(model, cohort, suffix, strategy):
    cache_path = EMB_DIR / f"{CACHE_OVERRIDE.get(model, model)}_{suffix}.pt"
    if not cache_path.exists():
        print(f"  MISSING {cache_path.name}", file=sys.stderr, flush=True)
        return []
    d = torch.load(cache_path, map_location="cpu", weights_only=False)
    X = d["embeddings"].numpy().astype(np.float32)
    y = d["labels"].numpy().astype(np.float32)
    label_columns = list(d["label_columns"])
    groups = d.get("groups")
    if suffix == "ctrate":
        # Keep one reconstruction per (patient, study), then patient-grouped
        # folds so a patient never spans train/test (mirrors paper_save_pools.py).
        ids_l = [str(x) for x in (d.get("ids") if d.get("ids") is not None else [])]
        keep = ctrate_dedup_keep(ids_l)
        X, y = X[keep], y[keep]
        groups = group_array("ctrate", [ids_l[i] for i in keep])
        strategy = "group_kfold"
        print(f"  ctrate dedup: {len(ids_l)} -> {X.shape[0]} scans, "
              f"{int(groups.max()) + 1} patients", flush=True)
    n = X.shape[0]
    norms = np.linalg.norm(X, axis=1, keepdims=True).clip(min=1e-8)
    X_norm = X / norms

    splits = (list(group_kfold_splits(groups, N_FOLDS, SEED))
              if strategy == "group_kfold"
              else list(kfold_splits(n, N_FOLDS, SEED)))

    cols = {lab: i for i, lab in enumerate(label_columns)}
    knn_pool = {c: np.zeros(n) for c in cols}
    probe_pool = {c: np.zeros(n) for c in cols}

    t0 = time.time()
    for tr_idx, te_idx in splits:
        Xtr_n, Xte_n = X_norm[tr_idx], X_norm[te_idx]
        Xtr_r, Xte_r = X[tr_idx], X[te_idx]
        for lab, col in cols.items():
            ytr = y[tr_idx, col]
            knn_pool[lab][te_idx] = knn_probs_one_class(Xtr_n, ytr, Xte_n, K, WEIGHT)
            npos = ytr.sum()
            if npos == 0 or npos == len(ytr):
                probe_pool[lab][te_idx] = float(npos / max(len(ytr), 1))
                continue
            clf = LogisticRegression(C=PROBE_C, class_weight="balanced",
                                     solver="lbfgs", max_iter=PROBE_MAX_ITER)
            clf.fit(Xtr_r, ytr)
            probe_pool[lab][te_idx] = clf.predict_proba(Xte_r)[:, 1]
    print(f"  {model:14s} {cohort:14s} folds {time.time()-t0:5.1f}s", flush=True)

    rng = np.random.default_rng(SEED)
    rows = []
    for lab, col in cols.items():
        yc = y[:, col]
        npos = int(yc.sum())
        if npos == 0:
            continue
        knn_ap = float(average_precision_score(yc, knn_pool[lab]))
        probe_ap = float(average_precision_score(yc, probe_pool[lab]))
        prev = npos / n
        rec = {"concept": lab, "model": model, "dataset": cohort,
               "n_samples": n, "n_positives": npos, "prevalence": prev,
               "knn_pr_auc": knn_ap, "probe_pr_auc": probe_ap,
               "diff_probe_minus_knn": probe_ap - knn_ap}
        if N_BOOT > 0:
            kp, pp = knn_pool[lab], probe_pool[lab]
            ka = np.full(N_BOOT, np.nan); pa = np.full(N_BOOT, np.nan)
            df = np.full(N_BOOT, np.nan)
            for b in range(N_BOOT):
                bi = rng.integers(0, n, size=n)
                yi = yc[bi]
                if yi.sum() == 0:
                    continue
                k_ = average_precision_score(yi, kp[bi])
                p_ = average_precision_score(yi, pp[bi])
                ka[b], pa[b], df[b] = k_, p_, p_ - k_
            v = ~np.isnan(df)
            rec["knn_ci_low"], rec["knn_ci_high"] = np.nanpercentile(ka[v], [2.5, 97.5])
            rec["probe_ci_low"], rec["probe_ci_high"] = np.nanpercentile(pa[v], [2.5, 97.5])
            rec["diff_ci_low"], rec["diff_ci_high"] = np.nanpercentile(df[v], [2.5, 97.5])
            lo, hi = rec["diff_ci_low"], rec["diff_ci_high"]
            rec["diff_sign"] = ("probe>kNN" if lo > 0 else "kNN>probe" if hi < 0 else "tied")
        rows.append(rec)
    return rows


FIELDS = ["concept", "model", "dataset", "n_samples", "n_positives", "prevalence",
          "knn_pr_auc", "knn_ci_low", "knn_ci_high",
          "probe_pr_auc", "probe_ci_low", "probe_ci_high",
          "diff_probe_minus_knn", "diff_ci_low", "diff_ci_high", "diff_sign"]


def _fmt(r):
    out = {k: r.get(k, "") for k in FIELDS}
    for k in ("prevalence", "knn_pr_auc", "knn_ci_low", "knn_ci_high",
              "probe_pr_auc", "probe_ci_low", "probe_ci_high",
              "diff_probe_minus_knn", "diff_ci_low", "diff_ci_high"):
        if isinstance(out[k], (int, float)) and out[k] == out[k]:
            out[k] = f"{out[k]:.4f}"
    return out


def main():
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if OUT_CSV.exists():
        with OUT_CSV.open() as f:
            for r in csv.DictReader(f):
                done.add((r["model"], r["dataset"]))
        print(f"resume: {len(done)} done cells", flush=True)
    first = not OUT_CSV.exists() or OUT_CSV.stat().st_size == 0
    max_cells = int(os.environ.get("CTFM_MAXCELLS", "0"))  # 0 = no limit
    n_done_now = 0
    t = time.time()
    for cohort, suffix, strategy in COHORTS:
        for model in MODELS:
            if (model, cohort) in done:
                continue
            if max_cells and n_done_now >= max_cells:
                print(f"hit MAXCELLS={max_cells}, exiting cleanly", flush=True)
                return 0
            rows = run_one_cohort(model, cohort, suffix, strategy)
            n_done_now += 1
            mode = "w" if first else "a"
            with OUT_CSV.open(mode, newline="") as f:
                w = csv.DictWriter(f, fieldnames=FIELDS)
                if first:
                    w.writeheader()
                for r in rows:
                    w.writerow(_fmt(r))
            first = False
            print(f"  appended {len(rows)} rows {model}/{cohort}", flush=True)
    print(f"total {(time.time()-t)/60:.1f} min  N_BOOT={N_BOOT}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
