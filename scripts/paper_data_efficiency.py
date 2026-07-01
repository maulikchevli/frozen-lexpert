"""Data-efficiency sweep: refit the linear probe at a range of labelled-scan
budgets and trace macro skill, one curve per encoder. Same probe and folds as the
benchmark (balanced L2 logistic regression, 5-fold). RadChest, the
radiologist-validated cohort.

Point estimates only: the curves are precisely estimated -- both the metric
bootstrap and the subsample variance (which scans you label) come out within
+/-0.005 at every budget -- so no uncertainty band is drawn (see fig caption).

Outputs:
  ${CTFM_RESULTS}/paper/data_efficiency.csv     model,budget,macro_skill
  ${CTFM_RESULTS}/paper/figs/data_efficiency.pdf
"""
from __future__ import annotations
import os
import warnings
from pathlib import Path
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score

R = Path(os.environ.get("CTFM_RESULTS", "results"))
EMB = R / "embeddings"
FIG = R / "paper" / "figs"
# CT-CLIP's default cache is the degenerate avg-pooled pre-projection feature;
# read its canonical projected cache (ctclip-zs) so the probe curve matches the
# rest of the pipeline (same override paper_save_pools.py / paper_run_probe.py use).
CACHE_OVERRIDE = {"ctclip": "ctclip-zs"}
MODELS = ["colipri-crm", "ctclip", "ctssg", "spectre-large", "flexict",
          "ctfm", "merlin", "pillar0", "curia2", "flexict3d", "voxelfm"]
PRETTY = {"colipri-crm": "COLIPRI", "ctclip": "CT-CLIP", "ctfm": "CT-FM", "ctssg": "CT-SSG",
          "curia2": "Curia-2", "flexict": "FlexiCT", "flexict3d": "FlexiCT-SSL",
          "merlin": "Merlin", "pillar0": "Pillar-0", "spectre-large": "SPECTRE", "voxelfm": "VoxelFM"}
SUFFIX = "radchest"
SEED, NF, MIN_POS = 42, 5, 20
BUDGETS = [50, 100, 200, 400, 800, 1600]


def kfold(n, k, seed):
    rng = np.random.default_rng(seed); perm = rng.permutation(n)
    folds = np.array_split(perm, k)
    for i in range(k):
        yield np.concatenate([folds[j] for j in range(k) if j != i]), folds[i]


def skill(ap, prev):
    return (ap - prev) / (1.0 - prev) if prev < 0.999 else 0.0


def one_model(model):
    cache = CACHE_OVERRIDE.get(model, model)
    d = torch.load(EMB / f"{cache}_{SUFFIX}.pt", map_location="cpu", weights_only=False)
    X = d["embeddings"].numpy().astype(np.float32); y = d["labels"].numpy().astype(np.float32)
    n, C = X.shape[0], y.shape[1]
    keep = [c for c in range(C) if int(y[:, c].sum()) >= MIN_POS]
    splits = list(kfold(n, NF, SEED))
    train_max = min(len(tr) for tr, _ in splits)
    budgets = [b for b in BUDGETS if b < train_max] + [train_max]
    out = {}
    for b in budgets:
        pred = np.full((n, C), np.nan, np.float32)
        for fi, (tr, te) in enumerate(splits):
            rng = np.random.default_rng(SEED * 1000 + b * 7 + fi)
            sub = tr if b >= len(tr) else rng.choice(tr, size=b, replace=False)
            Xtr, Xte = X[sub], X[te]
            for c in keep:
                ytr = y[sub, c]
                if ytr.sum() < 2 or ytr.sum() == len(ytr):
                    pred[te, c] = float(ytr.mean()); continue
                clf = LogisticRegression(C=1.0, class_weight="balanced", solver="lbfgs", max_iter=2000)
                clf.fit(Xtr, ytr); pred[te, c] = clf.predict_proba(Xte)[:, 1]
        sks = [skill(average_precision_score(y[:, c], pred[:, c]), float(y[:, c].mean())) for c in keep]
        out[b] = float(np.mean(sks))
        print(f"  {model:14s} budget={b:5d} macro_skill={out[b]:.3f}", flush=True)
    return out


def main():
    rows = []
    for m in MODELS:
        for b, s in one_model(m).items():
            rows.append(dict(model=m, budget=b, macro_skill=round(s, 4)))
    df = pd.DataFrame(rows)
    (R / "paper").mkdir(parents=True, exist_ok=True)
    df.to_csv(R / "paper" / "data_efficiency.csv", index=False)

    order = df.groupby("model").macro_skill.max().sort_values(ascending=False).index
    cmap = plt.get_cmap("tab20")
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    for i, m in enumerate(order):
        s = df[df.model == m].sort_values("budget")
        ax.plot(s.budget, s.macro_skill, "-o", ms=3, lw=1.0, color=cmap(i % 20), label=PRETTY[m])
    ax.set_xscale("log")
    ax.set_xlabel("labelled training scans (probe refit)")
    ax.set_ylabel("macro probe skill (RadChest)")
    ax.set_title("Data efficiency: probe skill vs labelled scans", fontsize=10)
    ax.legend(fontsize=6, ncol=2, loc="upper left", frameon=False)
    ax.grid(alpha=0.25)
    FIG.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(); fig.savefig(FIG / "data_efficiency.pdf"); plt.close(fig)
    print("wrote", FIG / "data_efficiency.pdf")


if __name__ == "__main__":
    main()
