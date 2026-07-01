"""Macro AUROC per (model, cohort) with shared-resample bootstrap CIs.

Parallels the macro-skill table but in AUROC, the prevalence-invariant metric
used for the cross-dataset transfer claim. Same construction as
paper_bootstrap.py: read the saved per-sample prediction pools, keep labels with
>=20 positives, macro = mean over kept labels, CI from a per-cohort shared
resample (1000) so the ranking comparison is paired.

Bootstrap AUROC is the count-weighted P(score_pos > score_neg) (continuous-score,
ties get half credit); the POINT estimate uses sklearn roc_auc_score (exact).
Outputs:
  ${CTFM_RESULTS}/paper/macro_auroc_ci.csv     cohort,model,readout,point,ci_low,ci_high,n_labels
  ${CTFM_RESULTS}/paper/table_cohort_auroc.tex  kNN AUROC-by-cohort table
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
from sklearn.metrics import roc_auc_score  # noqa: E402
import paper_patient_groups as PG  # noqa: E402

R = Path(os.environ.get("CTFM_RESULTS", "results")) / "paper"
POOLS = R / "pools"
OUT_TEX = R / "table_cohort_auroc.tex"
B = 1000
MIN_POS = 20
SUF = {"radchest": "RadChestCT", "ctrate": "CT-RATE"}
MODELS = ["pillar0", "colipri-crm", "flexict", "ctssg", "curia2", "spectre-large",
          "merlin", "flexict3d", "voxelfm", "ctfm", "ctclip"]
PRETTY = {"pillar0": "Pillar-0", "colipri-crm": "COLIPRI", "flexict": "FlexiCT",
          "ctssg": "CT-SSG", "curia2": "Curia-2", "spectre-large": "SPECTRE",
          "merlin": "Merlin", "flexict3d": "FlexiCT-SSL", "voxelfm": "VoxelFM",
          "ctfm": "CT-FM", "ctclip": "CT-CLIP"}
READOUTS = ["knn", "probe"]


def _spearman(a, b):
    """Spearman rank correlation without scipy (rank -> Pearson)."""
    a = np.asarray(a, float); b = np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 2:
        return float("nan")
    ra = np.argsort(np.argsort(a[ok])); rb = np.argsort(np.argsort(b[ok]))
    return float(np.corrcoef(ra, rb)[0, 1])


def _auroc_counts(cpos, cneg, starts):
    """Tie-correct AUROC = P(pos>neg)+0.5 P(pos==neg) for count-weighted samples.
    cpos,cneg are (B,n) along ascending-score order; starts indexes equal-score
    group boundaries. Returns (B,) with nan where a class is empty."""
    cpos_g = np.add.reduceat(cpos, starts, axis=1)        # (B,G) per equal-score group
    cneg_g = np.add.reduceat(cneg, starts, axis=1)
    cneg_below = np.cumsum(cneg_g, axis=1) - cneg_g       # negs strictly lower than the group
    conc = (cpos_g * (cneg_below + 0.5 * cneg_g)).sum(1)  # +half credit for ties
    P = cpos.sum(1); N = cneg.sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where((P > 0) & (N > 0), conc / (P * N), np.nan)


def macro_auroc(Y, S, counts):
    n, C = Y.shape
    keep = [c for c in range(C) if int(Y[:, c].sum()) >= MIN_POS]
    if not keep:
        return None
    point, boot = [], []
    ones = np.ones((1, n))
    for c in keep:
        s = S[:, c].astype(np.float64)
        y = Y[:, c].astype(np.float64)
        order = np.argsort(s, kind="mergesort")           # ascending
        ys = y[order]
        ss = s[order]
        starts = np.concatenate([[0], np.where(np.diff(ss) != 0)[0] + 1])
        # exact point from the same formula (counts=1); validate vs sklearn
        p_formula = float(_auroc_counts(ones * ys, ones * (1 - ys), starts)[0])
        p_sklearn = float(roc_auc_score(y.astype(int), s))
        assert abs(p_formula - p_sklearn) < 1e-9, (p_formula, p_sklearn)
        point.append(p_sklearn)
        cs = counts[:, order]                             # (B,n) resample multiplicities
        boot.append(_auroc_counts(cs * ys, cs * (1 - ys), starts))
    macro_b = np.nanmean(np.vstack(boot), axis=0)
    return (float(np.mean(point)),
            float(np.nanpercentile(macro_b, 2.5)),
            float(np.nanpercentile(macro_b, 97.5)),
            len(keep))


def main():
    rows = []
    for suf in SUF:
        # one patient-grouped, canonical-order resample per cohort -> paired across models
        counts = PG.cohort_counts(suf, B)
        for m in MODELS:
            if not (POOLS / f"{m}_{suf}.npz").exists():
                continue
            for ro in READOUTS:
                Y, S, _ = PG.load_sorted(m, suf, ro)    # canonical id order, aligns with counts
                res = macro_auroc(Y, S, counts)
                if res is None:
                    continue
                pt, lo, hi, nl = res
                rows.append(dict(cohort=suf, model=m, readout=ro,
                                 point=round(pt, 4), ci_low=round(lo, 4),
                                 ci_high=round(hi, 4), n_labels=nl))
                print(f"  {suf:9s} {m:14s} {ro:5s} AUROC={pt:.3f} [{lo:.3f},{hi:.3f}] ({nl} labels)", flush=True)
    df = pd.DataFrame(rows)
    R.mkdir(parents=True, exist_ok=True)
    df.to_csv(R / "macro_auroc_ci.csv", index=False)
    print("wrote", R / "macro_auroc_ci.csv")

    # cross-cohort Spearman of the kNN AUROC ranking (two public cohorts -> one pair)
    kn = df[df.readout == "knn"].pivot(index="model", columns="cohort", values="point")
    cohs = [c for c in ["ctrate", "radchest"] if c in kn.columns]
    if len(cohs) == 2:
        rho = _spearman(kn[cohs[0]].values, kn[cohs[1]].values)
        print(f"\nkNN cross-cohort Spearman (AUROC), {cohs[0]} vs {cohs[1]}: {rho:.2f}")

    # LaTeX table (kNN), model order = skill table
    def cell(m, suf):
        r = df[(df.model == m) & (df.cohort == suf) & (df.readout == "knn")]
        if not len(r):
            return "--"
        r = r.iloc[0]
        return f"{r.point:.3f}\\,[{r.ci_low:.2f},{r.ci_high:.2f}]"
    lines = [r"\begin{tabular}{lcc}", r"\toprule",
             r"model & CT-RATE & RadChest\\", r"\midrule"]
    for m in MODELS:
        lines.append(f"{PRETTY[m]} & {cell(m,'ctrate')} & {cell(m,'radchest')}\\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    OUT_TEX.write_text("\n".join(lines) + "\n")
    print("wrote", OUT_TEX)


if __name__ == "__main__":
    main()
