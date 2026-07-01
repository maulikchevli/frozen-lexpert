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
  paper_bootstrap.py       patient-grouped shared-resample bootstrap CIs
  zero_shot_score.py       per-model zero-shot scoring
  zero_shot_compare.py     zero-shot aggregation + paired comparison
configs/{extract,evaluate,zero_shot}/   Hydra configs (paths via env vars)
docs/model_preprocessing.md             per-model preprocessing contracts
```

## Metric policy

PR-AUC (average precision), **not** ROC-AUC, on these imbalanced multi-label
tasks, always reported next to class prevalence (PR-AUC's random baseline =
prevalence). Point estimates are computed once on the pooled out-of-fold
predictions; uncertainty comes from the patient-level bootstrap. Model
comparisons use the **paired** bootstrap (same resample indices for both
models) so the CI of the *difference* is the basis for any "A beats B" claim.

## License

No license file is included in this anonymous submission; add one before any
public release.
