# Evaluating the Raw Representational Power of 3D CT Foundation Models

Companion code for the paper (anonymous submission). It reproduces the
**frozen-encoder readouts** on the two **public** datasets used in the paper:

* **RadChestCT** (2,284 validation scans)
* **CT-RATE** (validation split)

For each of ten pretrained 3D CT foundation models we extract image-level
embeddings using **each model's own author-prescribed preprocessing**, cache
them once, and then evaluate three readouts on the frozen features:

* **k-NN** (k = 5, cosine-weighted)
* **Zero-shot** classification (vision–language models only, fixed prompt pair)
* **Linear probe** (one-vs-rest ℓ2 logistic regression, `C = 1`, balanced)

All readouts use **5-fold patient-grouped cross-validation** and **1,000
paired, patient-level bootstrap resamples** for 95% confidence intervals and
paired model/readout comparisons.

> **Anonymity.** This repository is fully anonymized for double-blind review:
> no author names, institutions, absolute paths, or private data are included.
> The private clinical validation cohort reported in the paper is a restricted
> dataset and is **not** part of this release — all of its loading/evaluation
> code has been removed. Everything here runs on the two public datasets above.

---

## Models

| Model | Family | k-NN | Linear probe | Zero-shot |
|---|---|:--:|:--:|:--:|
| SPECTRE   | VL (SigLIP text head)        | ✓ | ✓ | ✓ |
| COLIPRI   | VL (CLIP-style)              | ✓ | ✓ | ✓ |
| CT-CLIP   | VL (CLIP-style)              | ✓ | ✓ | ✓ |
| Merlin    | VL                           | ✓ | ✓ | ✓ |
| Pillar-0  | VL (multi-windowing)         | ✓ | ✓ | ✓ |
| FlexiCT   | VL / agglomerative DINO·iBOT | ✓ | ✓ | ✓ |
| CT-SSG    | Supervised 3D classifier     | ✓ | ✓ |   |
| CT-FM     | SSL                          | ✓ | ✓ |   |
| Curia-2   | Slice-based DINOv2 SSL       | ✓ | ✓ |   |
| VoxelFM   | 3D DINOv2 SSL                | ✓ | ✓ |   |

`flexict3d` (a FlexiCT SSL-teacher variant) is included as an optional ablation
in the readout scripts.

Model weights are **not** shipped. Each extractor loads weights from the
upstream author's public release (Hugging Face Hub or a cloned repo). See
[`docs/model_preprocessing.md`](docs/model_preprocessing.md) for the exact
per-model install, weights source, preprocessing contract, and embedding call.

---

## Install

Python ≥ 3.11.

```bash
pip install -r requirements.txt
```

`requirements.txt` covers the core stack (torch, MONAI, scikit-learn, Hydra,
transformers, open-clip, …). A few models ship their own packages / repos that
you install only if you want to run that model — the exact commands are in
[`docs/model_preprocessing.md`](docs/model_preprocessing.md) (e.g. `spectre-fm`,
`colipri`, `merlin-vlm`, `rad-vision-engine` for Pillar-0, and cloned repos for
CT-SSG / CT-CLIP / CT-FM / VoxelFM / FlexiCT).

All scripts are run **from the repository root** (`ctfm_eval/` and `scripts/`
are importable from there — no editable install required).

## Data & environment variables

Download the two public datasets yourself and point the code at them via
environment variables (copy `.env.example` and fill it in):

| Variable | Meaning |
|---|---|
| `RADCHEST_ROOT` | RadChestCT root (contains `json/`, the metadata CSV, `label_categories_refined.json`) |
| `CTRATE_ROOT`   | CT-RATE root (contains `validation/` and `reports/`) |
| `CTRATE_META`   | Path to the CT-RATE metadata CSV that carries the DICOM rescale slope/intercept (the CT-CHAT `validation_metadata.csv`) |
| `CTFM_RESULTS`  | Output root for caches / pools / bootstraps (default: `./results`) |

The Hydra extract configs reference these as `${oc.env:RADCHEST_ROOT}` etc., so
they resolve at run time from your shell environment.

---

## Pipeline

### 1. Extract embeddings (GPU; one run per model × dataset)

```bash
python -m scripts.extract --config-name colipri_radchest
python -m scripts.extract --config-name colipri_ctrate
# ... one config per (model, dataset) in configs/extract/
```

Each run writes one cache to `${CTFM_RESULTS}/embeddings/<model>_<dataset>.pt`.
Smoke-test any config with `dataset.max_samples=2`.

### 2a. k-NN + linear probe (the paper's main readouts)

```bash
# per-label kNN + linear-probe PR-AUC matrix (point estimates)
python scripts/paper_run_probe.py

# out-of-fold prediction pools (patient-grouped; shared resample indices)
python scripts/paper_save_pools.py

# 1,000-resample patient-grouped paired bootstrap -> CIs & paired diffs
CTFM_B=1000 python scripts/paper_bootstrap.py
```

### 2b. Simple multi-model k-NN comparison (CPU)

```bash
python -m scripts.evaluate --config-name compare_models          # RadChestCT
python -m scripts.evaluate --config-name compare_models_ctrate   # CT-RATE
```

Reads the caches, runs cosine k-NN + bootstrap CIs + pairwise paired bootstrap,
writes JSON/CSV to `${CTFM_RESULTS}/knn_eval/`.

### 2c. Zero-shot (vision–language models only)

```bash
python -m scripts.zero_shot_score --config-name colipri     # one VL model
python -m scripts.zero_shot_compare --config-name compare   # aggregate + paired
```

Prompt pair: `"A chest CT scan showing {finding}."` /
`"A chest CT scan showing no {finding}."`, scored as
`σ(cos(z, t⁺) − cos(z, t⁻))` on ℓ2-normalized image/text embeddings. The scan's
own report is never used.

### 2d. AUROC, finding-type analysis, and paper figures

These read the pools / CSVs produced above — no GPU:

```bash
# per-(model, cohort, readout, label) ROC-AUC + PR-AUC — the shared input below
python scripts/paper_perclass_roc_pr.py

# macro AUROC per (model, cohort) with paired-bootstrap CIs (+ LaTeX table)
python scripts/paper_auroc_by_cohort.py
python scripts/paper_overview_macro_ci.py    # macro AUROC incl. zero-shot (Fig. 2a forest)

# finding-type taxonomy (label -> organ_system + phenotype); a prebuilt
# results/paper/concept_features.csv is shipped — this regenerates it
python scripts/paper_build_concept_features.py

# Kendall's W: finding-type difficulty concordance across encoders
python scripts/paper_kendall_w.py

# figures
python scripts/paper_fig_perclass_grid.py    # per-finding AUROC grid (Fig. 5)
python scripts/paper_fig_per_organ.py        # per-organ linear-probe AUROC (Fig. 2b)
python scripts/paper_data_efficiency.py      # data-efficiency curves (Fig. 2c)
```

Outputs land in `${CTFM_RESULTS}/paper/` (CSVs, `.tex` tables, and `figs/*.pdf`).

---

## Layout

```
ctfm_eval/                 library
  embeddings.py            per-model extractors + shared preprocessing helpers + EmbeddingBatch
  datasets.py              RadChestCT + CT-RATE loaders (author-prescribed transforms)
  knn.py                   cosine-kNN + multilabel metrics (PR-AUC / F1)
  eval.py                  CV splitters, out-of-fold pooling, bootstrap CIs, paired bootstrap
  zero_shot.py             prompt builders + per-model VL scorers
spectre_wrapper.py         thin wrapper over the upstream SPECTRE inference library
scripts/
  extract.py               Hydra entry point: (model, dataset) -> embedding cache
  evaluate.py              multi-model kNN comparison + pairwise bootstrap
  paper_run_probe.py       per-label kNN + linear-probe PR-AUC matrix
  paper_save_pools.py      out-of-fold pools for the paired bootstrap
  paper_patient_groups.py  patient-grouping + CT-RATE dedup utilities
  paper_bootstrap.py       patient-grouped shared-resample bootstrap CIs (skill + finding-type)
  paper_perclass_roc_pr.py per-(model,cohort,readout,label) ROC-AUC + PR-AUC
  paper_auroc_by_cohort.py macro AUROC per cohort + bootstrap CIs (+ LaTeX table)
  paper_overview_macro_ci.py macro AUROC (kNN + zero-shot) for the overview forest
  paper_build_concept_features.py label -> organ_system + finding-type taxonomy
  paper_kendall_w.py       finding-type difficulty concordance (Kendall W)
  paper_fig_perclass_grid.py per-finding AUROC grid (Fig. 5)
  paper_fig_per_organ.py   per-organ linear-probe AUROC (Fig. 2b)
  paper_data_efficiency.py data-efficiency curves (Fig. 2c)
  zero_shot_score.py       per-model zero-shot scoring
  zero_shot_compare.py     zero-shot aggregation + paired comparison
configs/{extract,evaluate,zero_shot}/   Hydra configs (paths via env vars)
docs/model_preprocessing.md             per-model preprocessing contracts
docs/finding_types.md                   finding -> phenotype mapping (human-readable, all 110 labels)
results/paper/concept_features.csv      shipped label -> organ/finding-type table (public cohorts)
```

## Metric policy

Matching the paper, we report two **paired** metrics — neither is read alone:

- **AUROC** — prevalence-independent, so encoders can be ranked comparably across
  cohorts with very different label frequencies. Because AUROC tolerates false
  positives on rare findings (it can sit well above chance even when a model is of
  little practical use), it is always paired with skill
  (`scripts/paper_auroc_by_cohort.py`, `scripts/paper_overview_macro_ci.py`,
  `scripts/paper_perclass_roc_pr.py`).
- **Skill** = prevalence-normalized PR-AUC = (AP − π)/(1 − π), where AP is the
  average precision and π the finding's prevalence, so a model scores 0 at the
  base rate and 1 for a perfect ranking. Skill restores the prevalence
  sensitivity that AUROC discards (`scripts/paper_bootstrap.py`).

Point estimates are computed once on the pooled out-of-fold predictions; 95% CIs
come from 1,000 paired patient-level bootstrap resamples (identical resample
indices across models and readouts), so the CI of the *difference* is the basis
for any "A beats B" claim. Macro averages are taken over findings with at least
20 positives.

## Reproducibility scope

This release runs on the two public cohorts only. Consequences for matching the
paper's exact printed numbers:

- The private clinical cohort is excluded (see **Anonymity** above), so paper
  numbers that are specific to it — e.g. its rows/columns in the multi-cohort
  figures — are not reproducible here.
- Statistics the paper computes over all three cohorts (Kendall's W, the
  per-organ and finding-type difficulty ordering) are recomputed here over
  RadChestCT + CT-RATE with **identical methodology**, so the exact values differ
  from the three-cohort figures in the paper.
- The contrast × extent analysis (paper Fig. 4) relies primarily on the private
  cohort and is **not** part of this release.

Everything else — the AUROC / skill readouts, the per-class grid, and the
per-organ and data-efficiency figures — is fully reproducible on the public
cohorts.

## License

No license file is included in this anonymous submission; add one before any
public release.
