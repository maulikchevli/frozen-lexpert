"""Shared-resample bootstrap from saved pools -> per (cohort,model,readout)
bootstrap skill arrays. Resamples are seeded per cohort, so they are IDENTICAL
across models/readouts -> paired model and readout comparisons (per the
project's paired-bootstrap policy).

For each (cohort, model, readout) saves results/paper/boot/<cohort>__<model>__<readout>.npz:
  macro[B]            bootstrap macro skill (mean over labels)
  ftypes (dict-like)  per finding_type bootstrap skill [B]
  point_macro, point_per_label, labels(list), prevalence(list)

Readouts: knn, probe (from <model>_<suffix>.npz), zs (from <model>_<suffix>_zs.npz).
Env: CTFM_B (default 1000), CTFM_COHORT (radchest|ctrate|all),
     CTFM_MAXCELLS (chunk; reap-safe).
"""
from __future__ import annotations
import os, sys, time, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.metrics import average_precision_score
sys.path.insert(0, str(Path(__file__).resolve().parent))
import paper_patient_groups as PG

R = Path(os.environ.get("CTFM_RESULTS", "results"))
POOLS = R / "paper" / "pools"
BOOT = R / "paper" / "boot"; BOOT.mkdir(parents=True, exist_ok=True)
B = int(os.environ.get("CTFM_B", "1000"))
SUFFIXES = {"radchest": "RadChestCT", "ctrate": "CT-RATE"}

# Optional per-finding "finding_type" table used only for the finding-type
# decomposition (ft__* outputs). Absent it, every label falls into "other" and
# the core macro-skill bootstrap is unaffected.
_CF_PATH = R / "paper" / "concept_features.csv"
CF = pd.read_csv(_CF_PATH) if _CF_PATH.exists() else None


def ftype_of(cohort, label):
    if CF is None:
        return "other"
    m = CF[(CF.cohort == cohort) & (CF.label == label)]
    if len(m):
        ft = m.finding_type.iloc[0]
        return ft if isinstance(ft, str) else "other"
    return "other"


def skill(ap, prev):
    return (ap - prev) / (1.0 - prev) if prev < 0.999 else 0.0


def _ap_counts(tp, pp, starts):
    """Count-weighted average precision (tie-correct), DESC-score order. tp =
    counts*y (true-positive weight), pp = counts (predicted-positive weight),
    both (B,n) along descending score; starts = equal-score group boundaries.
    AP = sum_g (dR_g) * precision_at_group_end_g; nan where no positives.
    Matches sklearn average_precision_score (validated per call)."""
    tp_g = np.add.reduceat(tp, starts, axis=1)         # (B,G) per equal-score group
    pp_g = np.add.reduceat(pp, starts, axis=1)
    tp_cum = np.cumsum(tp_g, axis=1)
    pp_cum = np.cumsum(pp_g, axis=1)
    P = tp_cum[:, -1]
    with np.errstate(invalid="ignore", divide="ignore"):
        prec = tp_cum / np.maximum(pp_cum, 1e-12)      # precision at group end
        ap = (tp_g * prec).sum(1) / np.maximum(P, 1e-12)
    return np.where(P > 0, ap, np.nan)


def boot_unit(cohort_suffix, cohort_name, model, readout):
    """Vectorized, patient-grouped macro-skill bootstrap. Resamples patients
    (not scans) via PG.cohort_counts so multi-scan patients do not inflate the
    effective sample size; pools are reindexed to a shared canonical id order so
    the per-cohort resample is paired by sample. Output format is unchanged."""
    try:
        if readout == "zs":                             # zs pools carry no ids -> scan-level
            Y, S, labs = PG.load_zs(model, cohort_suffix)
            counts = PG.scan_counts(cohort_suffix, Y.shape[0], B)
        else:                                           # knn/probe -> patient-grouped, canonical
            Y, S, labs = PG.load_sorted(model, cohort_suffix, readout)
            counts = PG.cohort_counts(cohort_suffix, B)
    except FileNotFoundError:
        return None
    n, C = Y.shape
    fts = [ftype_of(cohort_name, l) for l in labs]
    keep = [c for c in range(C) if int(Y[:, c].sum()) >= 20]
    if not keep:
        return None
    totw = counts.sum(1).astype(np.float64)             # (B,) resampled weight, shared across labels
    sk = np.full((len(keep), B), np.nan)                # skill per kept label per resample
    point_ap, prev = {}, {}
    ones = np.ones((1, n))
    for j, c in enumerate(keep):
        s = S[:, c].astype(np.float64); y = Y[:, c].astype(np.float64)
        order = np.argsort(-s, kind="mergesort")        # descending score
        ys = y[order]; ss = s[order]
        starts = np.concatenate([[0], np.where(np.diff(ss) != 0)[0] + 1])
        ap_pt = float(_ap_counts(ones * ys, ones, starts)[0])
        ap_sk = float(average_precision_score(y.astype(int), s))
        assert abs(ap_pt - ap_sk) < 1e-9, (cohort_suffix, model, readout, labs[c], ap_pt, ap_sk)
        point_ap[labs[c]] = ap_sk; prev[labs[c]] = float(y.mean())
        cs = counts[:, order]
        ap_b = _ap_counts(cs * ys, cs, starts)          # (B,) AP per resample
        P_b = (cs * ys).sum(1)
        with np.errstate(invalid="ignore"):
            prev_b = np.where(totw > 0, P_b / totw, np.nan)
            sk[j] = np.where((P_b > 0) & (prev_b < 0.999),
                             (ap_b - prev_b) / (1.0 - prev_b), np.nan)
    macro = np.nanmean(sk, axis=0)
    out = {"macro": macro,
           "point_macro": float(np.mean([skill(point_ap[l], prev[l]) for l in point_ap])),
           "labels": np.array(list(point_ap.keys()), dtype=object),
           "point_ap": np.array([point_ap[l] for l in point_ap]),
           "point_per_label": np.array([skill(point_ap[l], prev[l]) for l in point_ap]),
           "prevalence": np.array([prev[l] for l in point_ap])}
    fts_keep = [fts[c] for c in keep]
    for ft in sorted(set(fts_keep)):
        rows = [j for j, f in enumerate(fts_keep) if f == ft]
        out[f"ft__{ft}"] = np.nanmean(sk[rows], axis=0)
    np.savez_compressed(BOOT / f"{cohort_suffix}__{model}__{readout}.npz", **out)
    return float(np.nanmean(macro))


KNN_PROBE_MODELS = ["colipri-crm", "ctclip", "ctssg", "spectre-large", "flexict",
                    "ctfm", "merlin", "pillar0", "curia2", "flexict3d", "voxelfm"]
ZS_MODELS = ["colipri-crm", "ctclip", "flexict", "merlin", "pillar0", "spectre-large"]


def main():
    only = os.environ.get("CTFM_COHORT", "all")
    maxc = int(os.environ.get("CTFM_MAXCELLS", "0")); n = 0
    units = []
    for suf, name in SUFFIXES.items():
        if only != "all" and only != suf:
            continue
        for m in KNN_PROBE_MODELS:
            units += [(suf, name, m, "knn"), (suf, name, m, "probe")]
        for m in ZS_MODELS:
            units.append((suf, name, m, "zs"))
    for suf, name, m, ro in units:
        if (BOOT / f"{suf}__{m}__{ro}.npz").exists():
            continue
        if maxc and n >= maxc:
            print("hit MAXCELLS", flush=True); return 0
        t = time.time(); r = boot_unit(suf, name, m, ro)
        if r is not None:
            n += 1
            print(f"  {suf:9s} {m:14s} {ro:5s} macro={r:.3f} ({time.time()-t:.0f}s)", flush=True)
    print("bootstrap units done", flush=True); return 0


if __name__ == "__main__":
    raise SystemExit(main())
