"""Per-organ linear-probe capability (Fig. 2b), RadChest.

For each organ system, the range of probe AUROC across the ten published encoders
(grey bar = min..max, tick = median), with three focus encoders overlaid. Reads
${CTFM_RESULTS}/paper/{perclass_roc_pr,concept_features}.csv; writes
${CTFM_RESULTS}/paper/figs/per_organ_capability.pdf.

Standalone public version (RadChest). The paper's Fig. 2 is a composite that also
reports a third, private cohort excluded from this public release.
"""
from __future__ import annotations
import os
os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.environ.get("TMPDIR", "/tmp"), "mpl"))
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

R = Path(os.environ.get("CTFM_RESULTS", "results")) / "paper"
FIG = R / "figs"
MIN_POS = 20
SUFFIX, COHORT = "radchest", "RadChestCT"

# ten published encoders (FlexiCT-SSL / flexict3d excluded, matching the paper)
PUBLISHED = ["colipri-crm", "ctclip", "ctssg", "spectre-large", "flexict",
             "ctfm", "merlin", "pillar0", "curia2", "voxelfm"]
PRETTY = {"colipri-crm": "COLIPRI", "ctclip": "CT-CLIP", "ctfm": "CT-FM", "ctssg": "CT-SSG",
          "curia2": "Curia-2", "flexict": "FlexiCT", "merlin": "Merlin", "pillar0": "Pillar-0",
          "spectre-large": "SPECTRE", "voxelfm": "VoxelFM"}
FOCUS = ["flexict", "colipri-crm", "pillar0"]
FCOL = {"flexict": "#4C72B0", "colipri-crm": "#DD8452", "pillar0": "#55A868"}
FMARK = {"flexict": "o", "colipri-crm": "D", "pillar0": "^"}
GREY_CTX, GREY_MED = "#c8c8c8", "#8a8a8a"

ORG_TAX = ["lung", "airways", "pleura", "mediastinum_vascular", "cardiac",
           "devices_surgical", "bones_chestwall", "chestwall_extrathoracic"]
ORG_PRETTY = {"lung": "Lung", "airways": "Airways", "pleura": "Pleura",
              "mediastinum_vascular": "Mediastinum", "cardiac": "Cardiac",
              "devices_surgical": "Devices", "bones_chestwall": "Bones",
              "chestwall_extrathoracic": "Extrathoracic"}


def organ_means():
    pc = pd.read_csv(R / "perclass_roc_pr.csv")
    cf = pd.read_csv(R / "concept_features.csv")
    m = pc[(pc.suffix == SUFFIX) & (pc.readout == "probe") & (pc.n_pos >= MIN_POS)].copy()
    cfm = cf[cf.cohort == COHORT][["label", "organ_system"]]
    j = m.merge(cfm, on="label", how="left")
    g = j.groupby(["model", "organ_system"]).roc_auc.mean().unstack().reindex(PUBLISHED)
    return g[[o for o in ORG_TAX if o in g.columns]]


def main():
    mo = organ_means()
    present = [o for o in ORG_TAX if o in mo.columns]
    order = sorted(present, key=lambda o: float(mo[o].median()))   # easy -> hard
    yi = {o: k for k, o in enumerate(order)}

    fig, ax = plt.subplots(figsize=(4.2, 3.4))
    for o in order:
        col = mo[o].values
        if np.all(np.isnan(col)):
            continue
        y = yi[o]
        ax.hlines(y, np.nanmin(col), np.nanmax(col), color=GREY_CTX, lw=3.0, zorder=1)
        ax.plot([np.nanmedian(col)] * 2, [y - 0.2, y + 0.2], color=GREY_MED, lw=0.9, zorder=2)
    for m in FOCUS:
        xs = [mo.loc[m, o] if o in mo.columns else np.nan for o in order]
        ax.scatter(xs, [yi[o] for o in order], marker=FMARK[m], s=22, color=FCOL[m],
                   zorder=4, edgecolors="white", linewidths=0.5)
    ax.set_ylim(-0.6, len(order) - 0.4)
    ax.set_xlim(0.5, 0.97)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([ORG_PRETTY[o] for o in order])
    for x in [0.6, 0.7, 0.8, 0.9]:
        ax.axvline(x, color="#efefef", lw=0.6, zorder=0)
    ax.set_xlabel("linear-probe AUROC (RadChest)")
    ax.set_title("Per-organ capability", fontsize=9)
    handles = [Line2D([0], [0], marker=FMARK[m], color="none", markerfacecolor=FCOL[m],
                      markeredgecolor="white", markersize=6, label=PRETTY[m]) for m in FOCUS]
    handles += [Line2D([0], [0], color=GREY_CTX, lw=3, label="all-10 range"),
                Line2D([0], [0], color=GREY_MED, lw=0.9, label="median")]
    ax.legend(handles=handles, fontsize=6, ncol=2, loc="lower right", frameon=False)
    FIG.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(); fig.savefig(FIG / "per_organ_capability.pdf"); plt.close(fig)
    print("wrote", FIG / "per_organ_capability.pdf")


if __name__ == "__main__":
    main()
