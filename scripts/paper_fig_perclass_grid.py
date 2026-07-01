"""Single-page per-finding AUROC grid (Fig. 5).

Layout: 4 row-bands = {RadChest, CT-RATE} x {high-prev, low-prev} (~10 findings
each, top-/bottom-k by prevalence within each cohort's >=20-positive set); 3
column-blocks = readouts {kNN, probe, zero-shot}, 10 encoder columns each.
Cell colour = AUROC (viridis). Zero-shot is defined only for the 6 report-aligned
encoders, so the 4 SSL/Sup columns of the zero-shot block are blank (light grey).
Finding names are coloured by organ system. Reads
${CTFM_RESULTS}/paper/{perclass_roc_pr,concept_features}.csv; writes
${CTFM_RESULTS}/paper/figs/perclass_grid.pdf.

Public cohorts only (RadChestCT + CT-RATE). The paper additionally reports a
third, private cohort that is excluded from this public release.
"""
import os
import tempfile
os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "mpl"))
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import matplotlib.patches as mp

R = Path(os.environ.get("CTFM_RESULTS", "results")) / "paper"
FIG = R / "figs"

COHORTS = [("radchest", "RadChestCT", "RadChest"),
           ("ctrate", "CT-RATE", "CT-RATE")]
READOUTS = [("knn", "kNN"), ("probe", "probe"), ("zs", "zero-shot")]
# VL (report-aligned, have zero-shot) first, then SSL/Sup; best->worst within group.
MODELS = ["flexict", "colipri-crm", "spectre-large", "pillar0", "merlin", "ctclip",
          "curia2", "ctssg", "voxelfm", "ctfm"]
MPRETTY = {"flexict": "FlexiCT", "colipri-crm": "COLIPRI", "spectre-large": "SPECTRE",
           "pillar0": "Pillar-0", "merlin": "Merlin", "ctclip": "CT-CLIP", "curia2": "Curia-2",
           "ctssg": "CT-SSG", "voxelfm": "VoxelFM", "ctfm": "CT-FM"}
OCOL = {"lung": "#56B4E9", "airways": "#0072B2", "pleura": "#009E73",
        "mediastinum_vascular": "#E69F00", "cardiac": "#D55E00", "devices_surgical": "#CC79A7",
        "bones_chestwall": "#999999", "chestwall_extrathoracic": "#AA4499",
        "breast": "#F0E442", "abdomen": "#117733", "neck": "#882255"}
OPRETTY = {"lung": "Lung", "airways": "Airways", "pleura": "Pleura",
           "mediastinum_vascular": "Mediastinum", "cardiac": "Cardiac", "devices_surgical": "Devices",
           "bones_chestwall": "Bones", "chestwall_extrathoracic": "Extrathoracic",
           "breast": "Breast", "abdomen": "Abdomen", "neck": "Neck"}
ORG = list(OCOL)
VMIN, VMAX = 0.5, 0.9
KPER = 10  # findings per band; CT-RATE clipped to len//2 so high/low stay disjoint

pc = pd.read_csv(R / "perclass_roc_pr.csv")
cf = pd.read_csv(R / "concept_features.csv")

# ---- ordered finding rows + band spans ----
rows, bands = [], []  # rows: dict(suf,label,name,organ); bands: (start,end,text)
for suf, cfname, pretty in COHORTS:
    base = (pc[(pc.suffix == suf) & (pc.readout == "probe") & (pc.n_pos >= 20)]
            .drop_duplicates("label")[["label", "prevalence", "n_pos"]])
    zsf = set(pc[(pc.suffix == suf) & (pc.readout == "zs")].label)
    base = base[base.label.isin(zsf)]  # only findings with a zero-shot score -> no blank ZS rows
    meta = cf[cf.cohort == cfname][["label", "finding_clean", "organ_system"]]
    base = base.merge(meta, on="label", how="left").sort_values("prevalence", ascending=False)
    k = min(KPER, len(base) // 2)
    for bandname, bdf in [("high", base.head(k)), ("low", base.tail(k))]:
        start = len(rows)
        for _, r in bdf.iterrows():
            rows.append(dict(suf=suf, label=r.label,
                             name=str(r.finding_clean).replace("_", " "), organ=r.organ_system,
                             prev=float(r.prevalence)))
        bands.append((start, len(rows), f"{pretty}\n{bandname}-prev"))
NRO = len(rows)

# ---- value matrix per readout block, concatenated ----
def auc(suf, ro, label, mod):
    v = pc[(pc.suffix == suf) & (pc.readout == ro) & (pc.label == label) & (pc.model == mod)].roc_auc
    return float(v.iloc[0]) if len(v) else np.nan

VL = MODELS[:6]  # the six report-aligned encoders; only these have zero-shot
BLOCKMODS = {"knn": MODELS, "probe": MODELS, "zs": VL}
blocks, colmods, blockstart = [], [], []
cur = 0
for ro, _ in READOUTS:
    mods = BLOCKMODS[ro]
    M = np.full((NRO, len(mods)), np.nan)
    for i, row in enumerate(rows):
        for j, mod in enumerate(mods):
            M[i, j] = auc(row["suf"], ro, row["label"], mod)
    blocks.append(M)
    blockstart.append(cur); colmods.append(mods); cur += len(mods)
full = np.hstack(blocks)
NCOL = full.shape[1]

# ---- draw ----
fig = plt.figure(figsize=(7.2, 8.5))
ax = fig.add_axes([0.24, 0.10, 0.68, 0.80])
cmap = plt.get_cmap("viridis").copy(); cmap.set_bad("#e9e9e9")
ax.set_facecolor("#e9e9e9")
xedges = np.arange(NCOL + 1) - 0.5
yedges = np.arange(NRO + 1) - 0.5
norm = Normalize(VMIN, VMAX)
im = ax.pcolormesh(xedges, yedges, np.ma.masked_invalid(full), cmap=cmap,
                   norm=norm, shading="flat", antialiased=False, linewidth=0)
ax.set_xlim(-0.5, NCOL - 0.5); ax.set_ylim(NRO - 0.5, -0.5)

# ---- per-cell AUROC numbers (x100); best encoder per finding within each readout
# block is bold AND outlined in red. ----
def _cell_color(v):
    r, g, b, _ = cmap(norm(v))
    return "black" if (0.2126 * r + 0.7152 * g + 0.0722 * b) > 0.55 else "white"
FS_CELL = 4.0
WIN_EDGE = "#e8000b"
for M, bs in zip(blocks, blockstart):
    for i in range(NRO):
        rowvals = M[i, :]
        if not np.any(np.isfinite(rowvals)):
            continue
        jmax = int(np.nanargmax(rowvals))
        ax.add_patch(mp.Rectangle((bs + jmax - 0.5, i - 0.5), 1, 1, fill=False,
                                  edgecolor=WIN_EDGE, lw=0.55, zorder=5))
        for j in range(M.shape[1]):
            v = M[i, j]
            if not np.isfinite(v):
                continue
            ax.text(bs + j, i, f"{v * 100:.0f}", ha="center", va="center", zorder=6,
                    fontsize=FS_CELL, color=_cell_color(v),
                    fontweight="bold" if j == jmax else "normal")

# block separators + readout titles
for b in blockstart[1:]:
    ax.axvline(b - 0.5, color="white", lw=2.4)
for b, mods, (ro, rolab) in zip(blockstart, colmods, READOUTS):
    ax.text(b + (len(mods) - 1) / 2, -3.9, rolab, ha="center", va="bottom", fontsize=10, fontweight="bold")

# band separators (white gutter between bands)
for start, end, _ in bands[:-1]:
    ax.axhline(end - 0.5, color="white", lw=2.0)

# encoder x labels (repeated per block)
xt, xl = [], []
for b, mods in zip(blockstart, colmods):
    for j, mod in enumerate(mods):
        xt.append(b + j); xl.append(MPRETTY[mod])
ax.set_xticks(xt); ax.set_xticklabels(xl, rotation=90, fontsize=5.0)
ax.xaxis.tick_top(); ax.tick_params(axis="x", length=0, pad=1)
for s in ax.spines.values():
    s.set_visible(False)

# finding y labels (name + its prevalence in that cohort), coloured by organ
def _pct(p):
    return f"{p*100:.0f}%" if p * 100 >= 1 else f"{p*100:.1f}%"
ax.set_yticks(range(NRO))
ax.set_yticklabels([f"{r['name']}  {_pct(r['prev'])}" for r in rows], fontsize=5.4)
ax.tick_params(axis="y", length=0, pad=1)
for tick, r in zip(ax.get_yticklabels(), rows):
    tick.set_color(OCOL.get(r["organ"], "#333333"))

# band labels on the right via a twin y-axis
axR = ax.twinx(); axR.set_ylim(NRO - 0.5, -0.5)
axR.set_yticks([(s + e - 1) / 2 for s, e, _ in bands])
axR.set_yticklabels([t for _, _, t in bands], fontsize=6.4, fontweight="bold")
axR.tick_params(length=0, pad=3)
for s in axR.spines.values():
    s.set_visible(False)

# ---- bottom legend row: AUROC colourbar (left) + organ key (right) ----
cax = fig.add_axes([0.30, 0.072, 0.20, 0.011])
cb = fig.colorbar(im, cax=cax, orientation="horizontal")
cb.set_label("AUROC", fontsize=7.5); cb.ax.tick_params(labelsize=6.5)

oh = [mp.Patch(color=OCOL[o], label=OPRETTY[o]) for o in ORG]
fig.legend(handles=oh, loc="center", bbox_to_anchor=(0.70, 0.060),
           ncol=4, fontsize=5.6, frameon=False, handlelength=1.0,
           columnspacing=1.1, labelspacing=0.3, handletextpad=0.4,
           title="organ", title_fontsize=6.4)

FIG.mkdir(parents=True, exist_ok=True)
out = FIG / "perclass_grid.pdf"
fig.savefig(out, bbox_inches="tight", pad_inches=0)
fig.savefig(os.path.join(tempfile.gettempdir(), "perclass_grid.png"), dpi=200,
            bbox_inches="tight", pad_inches=0)
plt.close(fig)
print("wrote", out, "| rows", NRO, "| cols", NCOL, "| bands", [(t.replace(chr(10), '/'), e - s) for s, e, t in bands])
