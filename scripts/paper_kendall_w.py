"""Kendall's coefficient of concordance (W) for the finding-type difficulty
ordering across encoders (the consistency stat behind "difficulty is set by the
finding, not the model").

Builds the encoder x finding-type matrix of mean linear-probe AUROC (pooled over
the public cohorts, labels with >=20 positives), ranks the finding types within
each encoder, and reduces the rank matrix to Kendall's W:
    W = 12 * sum_j (R_j - Rbar)^2 / (m^2 * (n^3 - n))
with m = #encoders (raters), n = #finding types. Also reports the mean pairwise
Spearman of the per-encoder difficulty vectors. numpy only (no scipy).

Reads ${CTFM_RESULTS}/paper/{perclass_roc_pr,concept_features}.csv;
writes ${CTFM_RESULTS}/paper/kendall_w.txt.

NOTE: the paper computes W over three cohorts (incl. a private clinical cohort);
this public version recomputes the identical statistic over RadChestCT + CT-RATE
only, so the numeric W is a faithful reproduction of the methodology, not the
paper's exact printed value.
"""
from __future__ import annotations
import os
from pathlib import Path
import numpy as np
import pandas as pd

R = Path(os.environ.get("CTFM_RESULTS", "results")) / "paper"
MIN_POS = 20
# ten published encoders (FlexiCT-SSL / flexict3d excluded, matching the paper)
MODELS = ["colipri-crm", "ctclip", "ctssg", "spectre-large", "flexict",
          "ctfm", "merlin", "pillar0", "curia2", "voxelfm"]


def _spearman(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    ra = np.argsort(np.argsort(a)); rb = np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def main():
    pc = pd.read_csv(R / "perclass_roc_pr.csv")
    cf = pd.read_csv(R / "concept_features.csv")
    # merge finding_type onto each per-label probe cell (match on cohort + label)
    cfm = cf[["cohort", "label", "finding_type"]].rename(columns={"cohort": "dataset"})
    m = pc[(pc.readout == "probe") & (pc.n_pos >= MIN_POS) & (pc.model.isin(MODELS))].copy()
    m = m.merge(cfm, on=["dataset", "label"], how="left")
    m = m.dropna(subset=["finding_type"])
    m = m[m.finding_type.astype(str).str.len() > 0]

    # encoder x finding-type mean AUROC (pooled across public cohorts' labels)
    vecs = (m.groupby(["model", "finding_type"]).roc_auc.mean()
            .unstack().reindex(MODELS))
    # keep only finding types present for every encoder (complete rank matrix)
    complete = [ft for ft in vecs.columns if vecs[ft].notna().all()]
    dropped = [ft for ft in vecs.columns if ft not in complete]
    V = vecs[complete].to_numpy()                 # (m_encoders, n_ftypes)
    mm, n = V.shape

    ranks = np.argsort(np.argsort(V, axis=1), axis=1) + 1   # rank finding types within each encoder
    Rj = ranks.sum(0)
    W = float(12 * ((Rj - Rj.mean()) ** 2).sum() / (mm ** 2 * (n ** 3 - n)))
    pw = [_spearman(V[i], V[j]) for i in range(mm) for j in range(i + 1, mm)]
    mean_rho = float(np.nanmean(pw))

    ft_mean = {ft: float(np.nanmean(vecs[ft])) for ft in complete}
    easy_hard = sorted(ft_mean, key=ft_mean.get, reverse=True)

    lines = [
        f"Kendall W (finding-type difficulty concordance across {mm} encoders) = {W:.2f}",
        f"mean pairwise Spearman of per-encoder difficulty vectors = {mean_rho:.2f}",
        f"finding types (n={n}): {complete}",
        f"dropped (incomplete across encoders): {dropped}",
        f"difficulty band (mean AUROC): {min(ft_mean.values()):.2f}-{max(ft_mean.values()):.2f}",
        "easy -> hard: " + " > ".join(f"{ft}({ft_mean[ft]:.2f})" for ft in easy_hard),
        "(public 2-cohort recompute; see script header)",
    ]
    out = R / "kendall_w.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print("wrote", out)


if __name__ == "__main__":
    main()
