"""Macro AUROC point + 95% bootstrap CI per (cohort, model, readout), covering
BOTH cosine kNN and zero-shot (ZS). Feeds the macro-AUROC forest of the overview
figure (Fig. 2a).

Method matches paper_auroc_by_cohort.py exactly (same macro_auroc estimator,
MIN_POS=20, B=1000). kNN uses the patient-grouped shared resample
(PG.cohort_counts); ZS pools carry no ids, so they use the scan-level resample
(PG.scan_counts) -- the same convention the paper uses for zero-shot.

Writes ${CTFM_RESULTS}/paper/overview_macro_auroc_ci.csv
  cohort,model,readout,point,ci_low,ci_high,n_labels
Public cohorts only (RadChestCT + CT-RATE).
"""
from __future__ import annotations
import os
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # for scripts/ imports
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import paper_patient_groups as PG  # noqa: E402
from paper_auroc_by_cohort import macro_auroc  # noqa: E402

R = Path(os.environ.get("CTFM_RESULTS", "results")) / "paper"
POOLS = R / "pools"
SUF = ["ctrate", "radchest"]
MODELS = ["pillar0", "colipri-crm", "flexict", "ctssg", "curia2", "spectre-large",
          "merlin", "flexict3d", "voxelfm", "ctfm", "ctclip"]
B = 1000


def main():
    rows = []
    for suf in SUF:
        grouped = PG.cohort_counts(suf, B)            # patient-grouped (kNN)
        for m in MODELS:
            # kNN
            if (POOLS / f"{m}_{suf}.npz").exists():
                Y, S, _ = PG.load_sorted(m, suf, "knn")
                res = macro_auroc(Y, S, grouped)
                if res:
                    pt, lo, hi, nl = res
                    rows.append(dict(cohort=suf, model=m, readout="knn",
                                     point=round(pt, 4), ci_low=round(lo, 4),
                                     ci_high=round(hi, 4), n_labels=nl))
            # zero-shot (own row order, scan-level resample)
            if (POOLS / f"{m}_{suf}_zs.npz").exists():
                Yz, Sz, _ = PG.load_zs(m, suf)
                counts_z = PG.scan_counts(suf, Yz.shape[0], B)
                res = macro_auroc(Yz, Sz, counts_z)
                if res:
                    pt, lo, hi, nl = res
                    rows.append(dict(cohort=suf, model=m, readout="zs",
                                     point=round(pt, 4), ci_low=round(lo, 4),
                                     ci_high=round(hi, 4), n_labels=nl))
            done = [r for r in rows if r["model"] == m and r["cohort"] == suf]
            print(f"  {suf:9s} {m:14s} " + "  ".join(
                f"{r['readout']}={r['point']:.3f}[{r['ci_low']:.3f},{r['ci_high']:.3f}](n{r['n_labels']})"
                for r in done), flush=True)
    out = R / "overview_macro_auroc_ci.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print("wrote", out)


if __name__ == "__main__":
    main()
