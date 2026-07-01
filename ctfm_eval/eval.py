from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np
import torch

from .embeddings import EmbeddingBatch
from .knn import knn_multilabel_probs, multilabel_metrics


# ---------- Splitters ----------

def kfold_splits(n: int, k: int, seed: int) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    if k < 2 or k > n:
        raise ValueError(f"k must be in [2, n]; got k={k}, n={n}")
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    folds = np.array_split(perm, k)
    for i in range(k):
        test = folds[i]
        train = np.concatenate([folds[j] for j in range(k) if j != i])
        yield train, test


def loo_splits(n: int) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    all_idx = np.arange(n)
    for i in range(n):
        yield np.delete(all_idx, i), np.array([i])


def holdout_splits(
    n: int, test_frac: float, seed: int, n_repeats: int = 1,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    if not 0.0 < test_frac < 1.0:
        raise ValueError("test_frac must be in (0,1)")
    n_test = max(1, int(round(n * test_frac)))
    for r in range(n_repeats):
        rng = np.random.default_rng(seed + r)
        perm = rng.permutation(n)
        yield perm[n_test:], perm[:n_test]


def group_kfold_splits(
    groups: list[str] | np.ndarray, k: int, seed: int,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Patient-grouped k-fold: all samples from a single group land in the
    same fold (train or test). Required whenever a cohort holds multiple scans
    per patient (e.g. CT-RATE's up to 16 reconstructions per study) — scan-level
    k-fold would leak correlated scans of the same patient across train/test.

    Implementation: shuffle the unique groups, round-robin them into `k`
    buckets (which keeps the fold sample-sizes within ~1 group of each
    other), then yield (train_idx, test_idx) per bucket.
    """
    groups = np.asarray(groups)
    n = len(groups)
    if n == 0:
        return
    uniq = np.array(sorted(set(groups.tolist())))
    if k < 2 or k > len(uniq):
        raise ValueError(
            f"k must be in [2, n_groups]; got k={k}, n_groups={len(uniq)}"
        )
    rng = np.random.default_rng(seed)
    perm = rng.permutation(uniq)
    # Round-robin assignment for balanced bucket sizes.
    buckets: list[list[str]] = [[] for _ in range(k)]
    for i, g in enumerate(perm):
        buckets[i % k].append(g)

    all_idx = np.arange(n)
    for i in range(k):
        test_groups = set(buckets[i])
        test_mask = np.fromiter((g in test_groups for g in groups), dtype=bool, count=n)
        yield all_idx[~test_mask], all_idx[test_mask]


# ---------- Eval config ----------

@dataclass(slots=True)
class KnnEvalConfig:
    k: int = 5
    weighting: str = "uniform"
    threshold: float = 0.5
    strategy: str = "kfold"              # "kfold" | "loo" | "holdout" | "group_kfold"
    folds: int = 5                       # for kfold / group_kfold
    test_frac: float = 0.2               # for holdout
    n_repeats: int = 1                   # for holdout
    seed: int = 42
    n_boot: int = 1000                   # bootstrap resamples for 95% CIs


def _splits_for(cfg: KnnEvalConfig, n: int, groups: list[str] | None = None):
    if cfg.strategy == "kfold":
        return list(kfold_splits(n, cfg.folds, cfg.seed))
    if cfg.strategy == "loo":
        return list(loo_splits(n))
    if cfg.strategy == "holdout":
        return list(holdout_splits(n, cfg.test_frac, cfg.seed, cfg.n_repeats))
    if cfg.strategy == "group_kfold":
        if groups is None:
            raise ValueError(
                "strategy='group_kfold' requires per-sample groups (e.g. "
                "patient_id). Cached EmbeddingBatch.groups must be populated."
            )
        if len(groups) != n:
            raise ValueError(f"groups length {len(groups)} != n {n}")
        return list(group_kfold_splits(groups, cfg.folds, cfg.seed))
    raise ValueError(f"unknown strategy {cfg.strategy!r}")


# ---------- Bootstrap ----------

def _bootstrap_metrics(
    probs: torch.Tensor,
    targets: torch.Tensor,
    threshold: float,
    n_boot: int,
    seed: int,
) -> tuple[dict[str, tuple[float, float]], dict[str, tuple[list[float], list[float]]]]:
    """Resample pooled predictions with replacement; recompute metrics each draw.

    Returns two dicts of 95% CI bounds:
      scalar_ci:    {metric_name: (lo, hi)}
      per_class_ci: {metric_name: ([lo per class], [hi per class])}
    """
    n = probs.shape[0]
    rng = np.random.default_rng(seed)

    scalar_runs: dict[str, list[float]] = {}
    per_class_runs: dict[str, list[list[float]]] = {}
    for _ in range(n_boot):
        idx = torch.from_numpy(rng.integers(0, n, size=n))
        m = multilabel_metrics(probs[idx], targets[idx], threshold=threshold)
        for k, v in m.items():
            if isinstance(v, (int, float)):
                scalar_runs.setdefault(k, []).append(float(v))
            elif isinstance(v, list):
                per_class_runs.setdefault(k, []).append(v)

    scalar_ci: dict[str, tuple[float, float]] = {}
    for k, vals in scalar_runs.items():
        arr = np.asarray(vals, dtype=np.float64)
        scalar_ci[k] = (float(np.nanpercentile(arr, 2.5)),
                        float(np.nanpercentile(arr, 97.5)))

    per_class_ci: dict[str, tuple[list[float], list[float]]] = {}
    for k, vals in per_class_runs.items():
        arr = np.asarray(vals, dtype=np.float64)           # [n_boot, n_classes]
        lows = np.nanpercentile(arr, 2.5, axis=0)
        highs = np.nanpercentile(arr, 97.5, axis=0)
        per_class_ci[k] = (lows.tolist(), highs.tolist())
    return scalar_ci, per_class_ci


def _merge_scalar(point: dict, ci: dict[str, tuple[float, float]]) -> dict:
    out: dict = {}
    for k, v in point.items():
        if not isinstance(v, (int, float)):
            continue
        lo, hi = ci.get(k, (float("nan"), float("nan")))
        out[k] = {"point": float(v), "ci95_low": lo, "ci95_high": hi}
    return out


def _merge_per_class(
    point: dict,
    ci: dict[str, tuple[list[float], list[float]]],
    labels: list[str],
) -> dict:
    out: dict = {}
    for k, v in point.items():
        if not isinstance(v, list):
            continue
        lows, highs = ci.get(k, ([float("nan")] * len(v), [float("nan")] * len(v)))
        out[k] = {
            lbl: {"point": float(v[i]), "ci95_low": float(lows[i]), "ci95_high": float(highs[i])}
            for i, lbl in enumerate(labels)
        }
    return out


# ---------- Eval driver ----------

def evaluate(batch: EmbeddingBatch, cfg: KnnEvalConfig) -> dict:
    """Produce one prediction per sample using the chosen splitter, pool them,
    compute each metric once, and bootstrap over samples for 95% CIs.

    This is the same pipeline regardless of strategy:
      - kfold / loo / group_kfold: every sample appears once in the pool.
      - holdout with n_repeats>1: samples may repeat (that's fine for bootstrap).
    """
    probs_pool, targets_pool = pool_predictions(batch, cfg)

    point = multilabel_metrics(probs_pool, targets_pool, threshold=cfg.threshold)
    scalar_ci, per_class_ci = _bootstrap_metrics(
        probs_pool, targets_pool, threshold=cfg.threshold,
        n_boot=cfg.n_boot, seed=cfg.seed,
    )

    return {
        "config": _cfg_to_dict(cfg),
        "n_samples": int(batch.embeddings.shape[0]),
        "n_splits": len(_splits_for(cfg, batch.embeddings.shape[0], batch.groups)),
        "n_pooled_preds": int(probs_pool.shape[0]),
        "aggregate": _merge_scalar(point, scalar_ci),
        "per_class": _merge_per_class(point, per_class_ci, batch.label_columns),
        "label_columns": batch.label_columns,
    }


def _cfg_to_dict(cfg: KnnEvalConfig) -> dict:
    return {
        "k": cfg.k, "weighting": cfg.weighting, "threshold": cfg.threshold,
        "strategy": cfg.strategy, "folds": cfg.folds, "test_frac": cfg.test_frac,
        "n_repeats": cfg.n_repeats, "seed": cfg.seed, "n_boot": cfg.n_boot,
    }


# ---------- Pooled prediction cache ----------

def pool_predictions(batch: EmbeddingBatch, cfg: KnnEvalConfig) -> tuple[torch.Tensor, torch.Tensor]:
    """Run kNN under the chosen CV strategy and return the pooled prediction
    matrix (probs, targets), both [N, C]. Every sample appears exactly once
    in kfold/LOO/group_kfold, possibly multiple times in holdout with
    n_repeats>1."""
    X, Y = batch.embeddings, batch.labels
    n = X.shape[0]
    splits = _splits_for(cfg, n, batch.groups)
    test_idx_chunks: list[np.ndarray] = []
    probs_chunks: list[torch.Tensor] = []
    for train_idx, test_idx in splits:
        p = knn_multilabel_probs(
            X[train_idx], Y[train_idx], X[test_idx],
            k=cfg.k, weighting=cfg.weighting,
        )
        test_idx_chunks.append(np.asarray(test_idx))
        probs_chunks.append(p)
    test_idx_pool = np.concatenate(test_idx_chunks)
    probs_pool = torch.cat(probs_chunks, dim=0)
    targets_pool = Y[torch.from_numpy(test_idx_pool)]
    return probs_pool, targets_pool


# ---------- Paired bootstrap comparison ----------

def compare_paired(
    probs_a: torch.Tensor,
    probs_b: torch.Tensor,
    targets: torch.Tensor,
    label_columns: list[str],
    threshold: float = 0.5,
    n_boot: int = 1000,
    seed: int = 42,
) -> dict:
    """Paired bootstrap test: for each resample, compute metric_A - metric_B
    on the *same* sample indices. Returns 95% CI of the difference for each
    scalar metric and each per-class metric.

    A positive diff means model A > model B. If the CI excludes 0, the
    difference is significant at p < 0.05.
    """
    assert probs_a.shape == probs_b.shape == targets.shape
    n = probs_a.shape[0]
    rng = np.random.default_rng(seed)

    scalar_diffs: dict[str, list[float]] = {}
    per_class_diffs: dict[str, list[list[float]]] = {}

    for _ in range(n_boot):
        idx = torch.from_numpy(rng.integers(0, n, size=n))
        t = targets[idx]
        ma = multilabel_metrics(probs_a[idx], t, threshold=threshold)
        mb = multilabel_metrics(probs_b[idx], t, threshold=threshold)
        for k in ma:
            va, vb = ma[k], mb[k]
            if isinstance(va, (int, float)):
                scalar_diffs.setdefault(k, []).append(float(va) - float(vb))
            elif isinstance(va, list):
                per_class_diffs.setdefault(k, []).append(
                    [a - b for a, b in zip(va, vb)]
                )

    out: dict = {"scalar": {}, "per_class": {}}

    for k, diffs in scalar_diffs.items():
        arr = np.asarray(diffs, dtype=np.float64)
        point = float(np.nanmean(arr))
        lo = float(np.nanpercentile(arr, 2.5))
        hi = float(np.nanpercentile(arr, 97.5))
        sig = "A" if lo > 0 else ("B" if hi < 0 else "ns")
        out["scalar"][k] = {"diff": point, "ci95_low": lo, "ci95_high": hi, "sig": sig}

    for k, diffs in per_class_diffs.items():
        arr = np.asarray(diffs, dtype=np.float64)  # [n_boot, n_classes]
        points = np.nanmean(arr, axis=0)
        lows = np.nanpercentile(arr, 2.5, axis=0)
        highs = np.nanpercentile(arr, 97.5, axis=0)
        out["per_class"][k] = {}
        for i, lbl in enumerate(label_columns):
            sig = "A" if lows[i] > 0 else ("B" if highs[i] < 0 else "ns")
            out["per_class"][k][lbl] = {
                "diff": float(points[i]),
                "ci95_low": float(lows[i]),
                "ci95_high": float(highs[i]),
                "sig": sig,
            }

    return out
