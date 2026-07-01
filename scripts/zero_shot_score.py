"""Zero-shot classification for one VL model on cached image embeddings.

Usage:
    uv run python -m scripts.zero_shot_score --config-name colipri

Outputs:
    <output_dir>/<name>_predictions.pt       {probs, targets, label_columns}
    <output_dir>/<name>.json                 {aggregate + per_class metrics}
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

from ctfm_eval.embeddings import EmbeddingBatch
from ctfm_eval.eval import _bootstrap_metrics, _merge_per_class, _merge_scalar
from ctfm_eval.knn import multilabel_metrics
from ctfm_eval.zero_shot import (
    build_prompts,
    build_prompts_ensemble,
    build_scorer,
    score_against_prompts,
    score_against_prompts_ensemble,
)


def _device(spec: str | None) -> torch.device:
    if spec:
        return torch.device(spec)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@hydra.main(version_base=None, config_path="../configs/zero_shot", config_name=None)
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    batch = EmbeddingBatch.load(cfg.embedding_path)
    if cfg.get("label_category"):
        import json as _json
        with open(cfg.get("label_categories",
                          os.path.join(os.environ.get("RADCHEST_ROOT", "."),
                                       "label_categories_refined.json"))) as f:
            cats = _json.load(f)
        batch = batch.subset_labels(list(cats[cfg.label_category]))
        print(f"[label_category={cfg.label_category}] using {len(batch.label_columns)} labels")
    elif cfg.get("label_subset"):
        batch = batch.subset_labels(list(cfg.label_subset))
        print(f"[label_subset] using {len(batch.label_columns)} labels")
    print(f"loaded {batch.embeddings.shape}  labels={len(batch.label_columns)}  "
          f"dataset={batch.dataset_name}")

    prompt_style = str(cfg.get("prompt_style", "generic"))
    ensemble = bool(cfg.get("prompt_ensemble", False))
    if ensemble:
        pos_sets, neg_sets = build_prompts_ensemble(batch.label_columns)
        # Flatten: one prompt per (label, template) pair for encode_text.
        print(f"[prompt ensemble] {len(pos_sets[0])} templates × {len(batch.label_columns)} labels")
        for lbl, p_list in list(zip(batch.label_columns, pos_sets))[:3]:
            print(f"  {lbl}: {p_list[0]!r} ... ({len(p_list)} paraphrases)")
        pos, neg = pos_sets, neg_sets
    else:
        pos, neg = build_prompts(batch.label_columns, style=prompt_style)
        for lbl, p, n in zip(batch.label_columns, pos, neg):
            print(f"  {lbl:32s}  pos={p!r}  neg={n!r}")

    device = _device(cfg.get("device"))
    scorer = build_scorer(cfg.model.name)
    if ensemble:
        probs = score_against_prompts_ensemble(batch.embeddings, pos, neg, scorer, device)
    else:
        probs = score_against_prompts(batch.embeddings, pos, neg, scorer, device)  # [N, C]
    targets = batch.labels

    threshold = float(cfg.get("threshold", 0.5))
    n_boot = int(cfg.get("n_boot", 1000))
    seed = int(cfg.get("seed", 42))

    point = multilabel_metrics(probs, targets, threshold=threshold)
    scalar_ci, per_class_ci = _bootstrap_metrics(
        probs, targets, threshold=threshold, n_boot=n_boot, seed=seed,
    )

    result = {
        "config": {
            "model": cfg.model.name,
            "threshold": threshold, "n_boot": n_boot, "seed": seed,
            "prompt_style": prompt_style,
            "prompt_ensemble": ensemble,
        },
        "n_samples": int(probs.shape[0]),
        "aggregate": _merge_scalar(point, scalar_ci),
        "per_class": _merge_per_class(point, per_class_ci, batch.label_columns),
        "label_columns": batch.label_columns,
    }

    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"probs": probs, "targets": targets, "label_columns": batch.label_columns,
         "model_name": cfg.model.name},
        out_dir / f"{cfg.model.name}_predictions.pt",
    )
    with (out_dir / f"{cfg.model.name}.json").open("w") as f:
        json.dump(result, f, indent=2)
    print(f"\nmacro PR-AUC = {point['macro_pr_auc']:.3f}   "
          f"macro F1 = {point['macro_f1']:.3f}   "
          f"-> {out_dir}")


if __name__ == "__main__":
    main()
