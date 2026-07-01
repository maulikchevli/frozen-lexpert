"""Per-finding ROC-AUC + PR-AUC for every (dataset, model, readout, label).

Reads the saved out-of-fold prediction pools (no GPU, no re-extraction):
  ${CTFM_RESULTS}/paper/pools/<model>_<suffix>.npz    -> readouts knn, probe [N,C]
  ${CTFM_RESULTS}/paper/pools/<model>_<suffix>_zs.npz  -> readout  zs        [N,Cz]

For each label with >=1 positive and >=1 negative, computes the point-estimate
PR-AUC (average precision) and ROC-AUC from the pooled predictions, alongside
class prevalence and positive count. ROC-AUC is requested here as an explicit,
prevalence-invariant complement to PR-AUC; per project policy prevalence is
always carried next to PR-AUC.

Output:
  ${CTFM_RESULTS}/paper/perclass_roc_pr.csv
    dataset,suffix,model,readout,label,n_samples,n_pos,prevalence,pr_auc,roc_auc

This is the shared input for the per-class grid (Fig. 5), the per-organ figure,
and the Kendall-W finding-type concordance. Public cohorts only (RadChestCT +
CT-RATE).
"""
from __future__ import annotations

import csv
import os
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

R = Path(os.environ.get("CTFM_RESULTS", "results")) / "paper"
POOLS = R / "pools"
OUT_CSV = R / "perclass_roc_pr.csv"

# Same model order / suffixes as the rest of the paper pipeline.
MODELS = ["colipri-crm", "ctclip", "ctssg", "spectre-large", "flexict",
          "ctfm", "merlin", "pillar0", "curia2", "flexict3d", "voxelfm"]
COHORTS = [("RadChestCT", "radchest"),
           ("CT-RATE", "ctrate")]

FIELDS = ["dataset", "suffix", "model", "readout", "label",
          "n_samples", "n_pos", "prevalence", "pr_auc", "roc_auc"]


def perclass(Y, S, labels):
    """Yield (label, n_pos, prevalence, pr_auc, roc_auc) for usable columns."""
    n = Y.shape[0]
    for c, lab in enumerate(labels):
        yc = Y[:, c].astype(int)
        npos = int(yc.sum())
        if npos == 0 or npos == n:          # ROC/PR undefined with one class
            continue
        s = S[:, c].astype(np.float64)
        yield (lab, npos, npos / n,
               float(average_precision_score(yc, s)),
               float(roc_auc_score(yc, s)))


def main():
    rows = []
    for dataset, suf in COHORTS:
        for model in MODELS:
            # kNN + linear probe live in the main pool.
            main_pool = POOLS / f"{model}_{suf}.npz"
            if main_pool.exists():
                d = np.load(main_pool, allow_pickle=True)
                Y = d["labels"]
                labels = [str(x) for x in d["label_columns"]]
                for ro in ("knn", "probe"):
                    if ro not in d.files:
                        continue
                    for lab, npos, prev, ap, roc in perclass(Y, d[ro], labels):
                        rows.append(dict(dataset=dataset, suffix=suf, model=model,
                                         readout=ro, label=lab,
                                         n_samples=int(Y.shape[0]), n_pos=npos,
                                         prevalence=round(prev, 4),
                                         pr_auc=round(ap, 4), roc_auc=round(roc, 4)))
            # Zero-shot lives in a separate pool (VLMs only).
            zs_pool = POOLS / f"{model}_{suf}_zs.npz"
            if zs_pool.exists():
                d = np.load(zs_pool, allow_pickle=True)
                Y = d["labels"]
                labels = [str(x) for x in d["label_columns"]]
                for lab, npos, prev, ap, roc in perclass(Y, d["zs"], labels):
                    rows.append(dict(dataset=dataset, suffix=suf, model=model,
                                     readout="zs", label=lab,
                                     n_samples=int(Y.shape[0]), n_pos=npos,
                                     prevalence=round(prev, 4),
                                     pr_auc=round(ap, 4), roc_auc=round(roc, 4)))
        print(f"  {dataset:14s} done", flush=True)

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {OUT_CSV}  ({len(rows)} rows)")


if __name__ == "__main__":
    main()
