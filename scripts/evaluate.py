"""Run kNN evaluation on cached embeddings; compare multiple models side-by-side.

Usage:
    uv run python -m scripts.evaluate --config-name compare_spectre
"""
from __future__ import annotations

import csv
import json
import os
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from itertools import combinations

from ctfm_eval.embeddings import EmbeddingBatch
from ctfm_eval.eval import KnnEvalConfig, compare_paired, evaluate, pool_predictions


def _eval_cfg(cfg: DictConfig) -> KnnEvalConfig:
    e = cfg.eval
    return KnnEvalConfig(
        k=int(e.k),
        weighting=str(e.weighting),
        threshold=float(e.threshold),
        strategy=str(e.strategy),
        folds=int(e.get("folds", 5)),
        test_frac=float(e.get("test_frac", 0.2)),
        n_repeats=int(e.get("n_repeats", 1)),
        seed=int(e.get("seed", 42)),
        n_boot=int(e.get("n_boot", 1000)),
    )


def _fmt(entry: dict) -> str:
    return f"{entry['point']:.3f} [{entry['ci95_low']:.3f},{entry['ci95_high']:.3f}]"


def _print_comparison(results: dict[str, dict]) -> None:
    keys = ["macro_f1", "micro_f1", "macro_pr_auc", "subset_accuracy", "hamming_accuracy"]
    name_w = max(len(n) for n in results) + 2
    col_w = 26
    header = f"{'model':<{name_w}}" + "".join(f"{k:>{col_w}}" for k in keys)
    print("\n" + header)
    print("-" * len(header))
    for name, res in results.items():
        cells = [_fmt(res["aggregate"][k]).rjust(col_w) for k in keys]
        print(f"{name:<{name_w}}" + "".join(cells))


def _print_prevalence(results: dict[str, dict]) -> None:
    first = next(iter(results.values()))
    if "prevalence_per_class" not in first.get("per_class", {}):
        return
    label_columns = first["label_columns"]
    label_w = max(len(c) for c in label_columns) + 2
    print("\nClass prevalence (positive fraction)")
    print(f"{'label':<{label_w}}{'prevalence':>14}")
    print("-" * (label_w + 14))
    for c in label_columns:
        p = first["per_class"]["prevalence_per_class"].get(c, {"point": float("nan")})
        print(f"{c:<{label_w}}{p['point']:>14.3f}")


def _print_per_class(results: dict[str, dict], metric_key: str, title: str) -> None:
    """Per-class table: rows = labels, columns = models. Each cell = point [lo, hi]."""
    label_columns = next(iter(results.values()))["label_columns"]
    model_names = list(results.keys())

    label_w = max(len(c) for c in label_columns) + 2
    col_w = max(24, max(len(n) for n in model_names) + 2)

    print(f"\n{title}")
    header = f"{'label':<{label_w}}" + "".join(f"{n:>{col_w}}" for n in model_names)
    print(header)
    print("-" * len(header))
    for c in label_columns:
        cells = []
        for name in model_names:
            d = results[name]["per_class"][metric_key].get(
                c, {"point": float("nan"), "ci95_low": float("nan"), "ci95_high": float("nan")},
            )
            cells.append(_fmt(d).rjust(col_w))
        print(f"{c:<{label_w}}" + "".join(cells))


@hydra.main(version_base=None, config_path="../configs/evaluate", config_name=None)
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    eval_cfg = _eval_cfg(cfg)
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Optional: subset labels by category (or explicit list). Uses
    # `$RADCHEST_ROOT/label_categories_refined.json` (override with the
    # `label_categories` config key) when `label_category` is set; otherwise
    # falls back to `label_subset` (explicit list) or the cache's native labels.
    label_subset: list[str] | None = None
    if cfg.get("label_category"):
        import json as _json
        with open(cfg.get("label_categories",
                          os.path.join(os.environ.get("RADCHEST_ROOT", "."),
                                       "label_categories_refined.json"))) as f:
            cats = _json.load(f)
        label_subset = list(cats[cfg.label_category])
        print(f"\n[label_category={cfg.label_category}] using {len(label_subset)} labels")
    elif cfg.get("label_subset"):
        label_subset = list(cfg.label_subset)
        print(f"\n[label_subset] using {len(label_subset)} labels")

    # --- Per-model evaluation ---
    results: dict[str, dict] = {}
    pooled: dict[str, tuple] = {}             # name -> (probs, targets)
    batches: dict[str, EmbeddingBatch] = {}
    cache_paths: dict[str, str] = {}          # name -> embedding cache path
    for entry in cfg.embeddings:
        name = entry.name
        path = entry.path
        print(f"\n=== {name}  ({path}) ===")
        if not Path(path).exists():
            # Skip (don't crash) when a cache isn't on disk yet — lets the same
            # compare config be run while extraction is still in flight (e.g.
            # models extracting on other GPUs) and re-run idempotently as
            # caches appear. Loud, never silent: the skipped model is simply
            # absent from the comparison.
            print(f"[skip] missing embedding cache for {name}: {path}")
            continue
        batch = EmbeddingBatch.load(path)
        if label_subset is not None:
            batch = batch.subset_labels(label_subset)
        print(f"loaded {batch.embeddings.shape}  labels={len(batch.label_columns)}  "
              f"model={batch.model_name}  dataset={batch.dataset_name}")
        probs, targets = pool_predictions(batch, eval_cfg)
        pooled[name] = (probs, targets)
        batches[name] = batch
        cache_paths[name] = path
        res = evaluate(batch, eval_cfg)
        results[name] = res
        with (out_dir / f"{name}.json").open("w") as f:
            json.dump(res, f, indent=2)

    _print_comparison(results)
    _print_per_class(results, "pr_auc_per_class", "Per-class PR-AUC (point [95% CI])  [random baseline = prevalence]")
    _print_prevalence(results)
    _print_per_class(results, "f1_per_class", "Per-class F1 (point [95% CI])")

    with (out_dir / "summary.json").open("w") as f:
        json.dump({n: r for n, r in results.items()}, f, indent=2)
    _write_csvs(results, out_dir)

    # --- Pairwise paired bootstrap comparison ---
    if len(results) >= 2:
        print("\n" + "=" * 80)
        print("PAIRWISE PAIRED BOOTSTRAP COMPARISON (A - B)")
        print("=" * 80)
        label_cols = next(iter(results.values()))["label_columns"]
        pair_dir = out_dir / "pairwise"
        pair_dir.mkdir(parents=True, exist_ok=True)
        all_pairs: dict[str, dict] = {}
        pairs = list(combinations(results.keys(), 2))
        for i, (name_a, name_b) in enumerate(pairs, 1):
            pair_key = f"{name_a}_vs_{name_b}"
            pair_path = pair_dir / f"{pair_key}.json"
            # Idempotent resume: reuse an existing pair JSON only if it is newer
            # than both input caches (so a stale pair auto-invalidates when a
            # cache is re-extracted). Lets a re-run after a mid-pairwise death
            # compute only the missing pairs and converge instead of redoing all
            # C(n,2) from scratch every time.
            if pair_path.exists():
                pair_mtime = pair_path.stat().st_mtime
                inputs_mtime = max(
                    Path(cache_paths[name_a]).stat().st_mtime,
                    Path(cache_paths[name_b]).stat().st_mtime,
                )
                if pair_mtime >= inputs_mtime:
                    with pair_path.open() as f:
                        all_pairs[pair_key] = json.load(f)
                    print(f"[skip] {pair_key} (cached, newer than inputs)")
                    continue
            print(f"[pair {i}/{len(pairs)}] computing {pair_key} "
                  f"(n_boot={eval_cfg.n_boot}, ~45s)...", flush=True)
            pa, ta = pooled[name_a]
            pb, tb = pooled[name_b]
            comp = compare_paired(
                pa, pb, ta, label_cols,
                threshold=eval_cfg.threshold, n_boot=eval_cfg.n_boot, seed=eval_cfg.seed,
            )
            all_pairs[pair_key] = comp
            with pair_path.open("w") as f:
                json.dump(comp, f, indent=2)
            _print_pairwise(name_a, name_b, comp, label_cols)
        _write_pairwise_csv(all_pairs, label_cols, pair_dir / "pairwise_pr_auc.csv", "pr_auc_per_class")
        _write_pairwise_csv(all_pairs, label_cols, pair_dir / "pairwise_f1.csv", "f1_per_class")
        _write_pairwise_macro_csv(all_pairs, pair_dir / "pairwise_macro.csv")

    print(f"\nresults -> {out_dir}")


# Human-readable names for internal metric keys.
_METRIC_LABEL = {
    "macro_f1": "macro F1",
    "micro_f1": "micro F1",
    "macro_pr_auc": "macro PR-AUC",
    "subset_accuracy": "subset accuracy",
    "hamming_accuracy": "hamming accuracy",
    "pr_auc_per_class": "PR-AUC",
    "f1_per_class": "F1",
}


def _write_csvs(results: dict[str, dict], out_dir: Path) -> None:
    _write_aggregate_csv(results, out_dir / "aggregate.csv")
    _write_per_class_csv(results, "pr_auc_per_class", out_dir / "per_class_pr_auc.csv")
    _write_per_class_csv(results, "f1_per_class", out_dir / "per_class_f1.csv")


def _write_aggregate_csv(results: dict[str, dict], path: Path) -> None:
    """Rows = models. Columns: model, then for each metric a (value, CI low, CI high) triplet."""
    model_names = list(results.keys())
    metric_keys = list(next(iter(results.values()))["aggregate"].keys())
    header = ["model"]
    for m in metric_keys:
        label = _METRIC_LABEL.get(m, m)
        header += [label, f"{label} CI low", f"{label} CI high"]
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for name in model_names:
            agg = results[name]["aggregate"]
            row: list = [name]
            for m in metric_keys:
                d = agg[m]
                row += [d["point"], d["ci95_low"], d["ci95_high"]]
            w.writerow(row)


def _write_per_class_csv(results: dict[str, dict], metric_key: str, path: Path) -> None:
    """Rows = labels. Columns: label, prevalence, then per model a (metric, CI low, CI high) triplet."""
    first = next(iter(results.values()))
    label_columns = first["label_columns"]
    model_names = list(results.keys())
    metric_label = _METRIC_LABEL.get(metric_key, metric_key)

    has_prev = "prevalence_per_class" in first.get("per_class", {})
    header = ["label"]
    if has_prev:
        header.append("prevalence")
    for n in model_names:
        header += [f"{n} {metric_label}", f"{n} {metric_label} CI low", f"{n} {metric_label} CI high"]

    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for c in label_columns:
            row: list = [c]
            if has_prev:
                row.append(first["per_class"]["prevalence_per_class"][c]["point"])
            for n in model_names:
                d = results[n]["per_class"][metric_key].get(c)
                if d is None:
                    row += ["", "", ""]
                else:
                    row += [d["point"], d["ci95_low"], d["ci95_high"]]
            w.writerow(row)


def _fmt_diff(d: dict) -> str:
    sign = "+" if d["diff"] >= 0 else ""
    sig = " *" if d["sig"] != "ns" else ""
    return f"{sign}{d['diff']:.3f} [{sign}{d['ci95_low']:.3f},{sign}{d['ci95_high']:.3f}]{sig}"


def _print_pairwise(name_a: str, name_b: str, comp: dict, label_cols: list[str]) -> None:
    print(f"\n--- {name_a}  vs  {name_b}  (positive = {name_a} better) ---")
    # Scalars
    for k, d in comp["scalar"].items():
        label = _METRIC_LABEL.get(k, k)
        print(f"  {label:>22}:  {_fmt_diff(d)}")
    # Per-class PR-AUC
    if "pr_auc_per_class" in comp["per_class"]:
        print(f"\n  {'Per-class PR-AUC diff':}")
        for lbl in label_cols:
            d = comp["per_class"]["pr_auc_per_class"].get(lbl)
            if d:
                print(f"    {lbl:<30} {_fmt_diff(d)}")


def _write_pairwise_csv(
    all_pairs: dict[str, dict],
    label_cols: list[str],
    path: Path,
    metric_key: str,
) -> None:
    """One CSV with per-class diff for each model pair."""
    metric_label = _METRIC_LABEL.get(metric_key, metric_key)
    header = ["label"]
    pair_keys = list(all_pairs.keys())
    for pk in pair_keys:
        header += [f"{pk} diff {metric_label}", f"{pk} CI low", f"{pk} CI high", f"{pk} sig"]
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for lbl in label_cols:
            row: list = [lbl]
            for pk in pair_keys:
                d = all_pairs[pk].get("per_class", {}).get(metric_key, {}).get(lbl)
                if d:
                    row += [d["diff"], d["ci95_low"], d["ci95_high"], d["sig"]]
                else:
                    row += ["", "", "", ""]
            w.writerow(row)


def _write_pairwise_macro_csv(all_pairs: dict[str, dict], path: Path) -> None:
    """One row per pair, columns = macro metric diffs."""
    pair_keys = list(all_pairs.keys())
    if not pair_keys:
        return
    metric_keys = list(all_pairs[pair_keys[0]]["scalar"].keys())
    header = ["pair"]
    for m in metric_keys:
        label = _METRIC_LABEL.get(m, m)
        header += [f"{label} diff", f"{label} CI low", f"{label} CI high", f"{label} sig"]
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for pk in pair_keys:
            row: list = [pk]
            for m in metric_keys:
                d = all_pairs[pk]["scalar"][m]
                row += [d["diff"], d["ci95_low"], d["ci95_high"], d["sig"]]
            w.writerow(row)


if __name__ == "__main__":
    main()
