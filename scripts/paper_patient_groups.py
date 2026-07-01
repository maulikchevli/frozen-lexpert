"""Patient-grouped, sample-paired bootstrap resampling for the paper CIs.

Several cohorts hold multiple scans per patient -- CT-RATE has 3{,}039 scans from
1{,}304 patients (up to 16 reconstructions each) -- so a scan-level bootstrap
treats correlated scans as independent and yields anticonservative intervals. We
resample PATIENTS with replacement and keep all of a patient's scans together.

We also reindex every pool to a canonical id order. Per-cohort resamples are
shared across models so paired diffs are valid, but CT-RATE pools are stored in
different row orders across models; sorting by id makes the shared resample paired
by SAMPLE, not by row index.

Public API:
  SEED                              per-cohort bootstrap seed
  group_array(suf, ids)            -> int[n] patient index per scan
  cohort_counts(suf, B)            -> int[B,n] patient-grouped multiplicities (cached, canonical order)
  load_sorted(model, suf, readout) -> (Y int8[n,C], S float32[n,C], labs list) in canonical id order
"""
from __future__ import annotations
from pathlib import Path
import os, re, glob
import numpy as np

R = Path(os.environ.get("CTFM_RESULTS", "results"))
POOLS = R / "paper" / "pools"
EMB = R / "embeddings"
SEED = {"radchest": 11, "ctrate": 22}


def group_array(suf, ids):
    """Map each scan id to a 0..G-1 patient index (order = first appearance)."""
    ids = [str(x) for x in ids]
    if suf == "radchest":
        key = ids                                   # one scan per patient
    elif suf == "ctrate":
        key = [(re.match(r"(valid_\d+)", s) or re.match(r"(.*)", s)).group(1) for s in ids]
    else:
        raise ValueError(suf)
    uniq = {}
    for k in key:
        uniq.setdefault(k, len(uniq))
    return np.array([uniq[k] for k in key], dtype=np.int64)


def ctrate_dedup_keep(ids):
    """Row indices that keep ONE reconstruction per (patient, study) for CT-RATE.

    CT-RATE val ships up to 16 reconstructions of the same scan
    (valid_<patient>_<study>_<recon>.nii.gz). Kept together in scan-level k-fold
    they leak near-duplicates across train/test. We keep the lowest-recon volume
    per (patient, study) so the eval unit is one scan per study; a patient may
    still recur across studies, but group_array() then confines that patient to a
    single CV fold. Deterministic and id-only, so the kept set is identical across
    all models (paired bootstrap stays valid). Unparseable ids are each kept once.
    """
    ids = [str(x) for x in ids]
    best = {}                                  # key -> (recon, row_idx)
    for i, s in enumerate(ids):
        stem = s[:-7] if s.endswith(".nii.gz") else s
        m = re.match(r"(valid_\d+)_([A-Za-z0-9]+)_(\d+)$", stem)
        if m:
            key, recon = (m.group(1), m.group(2)), int(m.group(3))
        else:
            key, recon = (s,), 0
        if key not in best or recon < best[key][0]:
            best[key] = (recon, i)
    return np.array(sorted(v[1] for v in best.values()), dtype=np.int64)


def _canonical(suf):
    """Canonical (id-sorted) order shared by all models in the cohort.
    Returns (sorted_ids list, perm-from-reference, group_array on sorted ids)."""
    ref = next(p for p in sorted(glob.glob(str(POOLS / f"*_{suf}.npz"))) if "_zs" not in p)
    ids = np.array([str(x) for x in np.load(ref, allow_pickle=True)["ids"]])
    order = np.argsort(ids, kind="mergesort")
    ids_sorted = ids[order]
    g = group_array(suf, ids_sorted)
    return ids_sorted, g


_COUNTS = {}


def cohort_counts(suf, B=1000):
    key = (suf, B)
    if key not in _COUNTS:
        _ids, g = _canonical(suf)
        G = int(g.max()) + 1
        rng = np.random.default_rng(SEED[suf])
        draws = rng.integers(0, G, size=(B, G))      # draw G patients w/ replacement
        pmult = np.zeros((B, G), dtype=np.int32)
        for b in range(B):
            pmult[b] = np.bincount(draws[b], minlength=G)
        _COUNTS[key] = pmult[:, g]                    # (B,n) scan multiplicities, canonical order
    return _COUNTS[key]


def scan_counts(suf, n, B=1000):
    """Scan-level (ungrouped) resample, for the zero-shot pools that ship no ids
    and so cannot be patient-grouped. zs feeds only the readout-decomposition,
    which has its own cell-level bootstrap, not the patient-sensitive transfer
    tables, so a scan-level resample here does not affect a grouped claim."""
    rng = np.random.default_rng(SEED[suf] + 1000)
    draws = rng.integers(0, n, size=(B, n))
    out = np.zeros((B, n), dtype=np.int32)
    for b in range(B):
        out[b] = np.bincount(draws[b], minlength=n)
    return out


def load_sorted(model, suf, readout):
    """Load a pool's labels + one readout's scores (knn|probe), reindexed to the
    canonical id order so it aligns with cohort_counts. zs is handled separately
    (scan_counts) because its pool carries no ids."""
    base = np.load(POOLS / f"{model}_{suf}.npz", allow_pickle=True)
    ids = np.array([str(x) for x in base["ids"]])
    order = np.argsort(ids, kind="mergesort")
    Y = base["labels"].astype(np.int8)[order]
    labs = list(base["label_columns"])
    S = base[readout].astype(np.float32)[order]
    return Y, S, labs


def load_zs(model, suf):
    """Load a zs pool in its own row order (no ids available)."""
    z = np.load(POOLS / f"{model}_{suf}_zs.npz", allow_pickle=True)
    return z["labels"].astype(np.int8), z["zs"].astype(np.float32), list(z["label_columns"])
