"""Aggregate zero-shot per-model predictions, pair-compare, print + write CSVs.

Input: per-model prediction files written by `scripts.zero_shot_score`:
    <pred_dir>/<name>_predictions.pt   {probs, targets, label_columns}

Output: `<output_dir>/`
    summary.json, aggregate.csv, per_class_pr_auc.csv, per_class_f1.csv,
    pairwise/*.json + pairwise_macro.csv + pairwise_pr_auc.csv + pairwise_f1.csv.

Usage:
    uv run python -m scripts.zero_shot_compare --config-name compare
"""
from __future__ import annotations

import csv
import json
from itertools import combinations
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

from ctfm_eval.eval import _bootstrap_metrics, _merge_per_class, _merge_scalar, compare_paired
from ctfm_eval.knn import multilabel_metrics


_METRIC_LABEL = {
    "macro_f1": "macro F1",
    "micro_f1": "micro F1",
    "macro_pr_auc": "macro PR-AUC",
    "subset_accuracy": "subset accuracy",
    "hamming_accuracy": "hamming accuracy",
    "pr_auc_per_class": "PR-AUC",
    "f1_per_class": "F1",
}


def _fmt(entry: dict) -> str:
    return f"{entry['point']:.3f} [{entry['ci95_low']:.3f},{entry['ci95_high']:.3f}]"


def _fmt_diff(d: dict) -> str:
    sign = "+" if d["diff"] >= 0 else ""
    sig = " *" if d["sig"] != "ns" else ""
    return f"{sign}{d['diff']:.3f} [{sign}{d['ci95_low']:.3f},{sign}{d['ci95_high']:.3f}]{sig}"


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


def _print_pairwise(name_a: str, name_b: str, comp: dict, label_cols: list[str]) -> None:
    print(f"\n--- {name_a}  vs  {name_b}  (positive = {name_a} better) ---")
    for k, d in comp["scalar"].items():
        label = _METRIC_LABEL.get(k, k)
        print(f"  {label:>22}:  {_fmt_diff(d)}")
    if "pr_auc_per_class" in comp["per_class"]:
        print("\n  Per-class PR-AUC diff")
        for lbl in label_cols:
            d = comp["per_class"]["pr_auc_per_class"].get(lbl)
            if d:
                print(f"    {lbl:<30} {_fmt_diff(d)}")


def _write_aggregate_csv(results: dict[str, dict], path: Path) -> None:
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


def _write_pairwise_csv(
    all_pairs: dict[str, dict],
    label_cols: list[str],
    path: Path,
    metric_key: str,
) -> None:
    metric_label = _METRIC_LABEL.get(metric_key, metric_key)
    pair_keys = list(all_pairs.keys())
    header = ["label"]
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


def _load_prediction(path: Path) -> dict:
    d = torch.load(path, map_location="cpu", weights_only=False)
    return {
        "probs": d["probs"],
        "targets": d["targets"],
        "label_columns": d["label_columns"],
    }


@hydra.main(version_base=None, config_path="../configs/zero_shot", config_name=None)
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    threshold = float(cfg.get("threshold", 0.5))
    n_boot = int(cfg.get("n_boot", 1000))
    seed = int(cfg.get("seed", 42))

    # --- Per-model point + bootstrap ---
    results: dict[str, dict] = {}
    pooled: dict[str, tuple] = {}
    for entry in cfg.predictions:
        name = entry.name
        path = Path(entry.path)
        if not path.exists():
            print(f"[skip] missing predictions for {name}: {path}")
            continue
        pred = _load_prediction(path)
        probs, targets = pred["probs"], pred["targets"]
        pooled[name] = (probs, targets)

        point = multilabel_metrics(probs, targets, threshold=threshold)
        scalar_ci, per_class_ci = _bootstrap_metrics(
            probs, targets, threshold=threshold, n_boot=n_boot, seed=seed,
        )
        results[name] = {
            "n_samples": int(probs.shape[0]),
            "aggregate": _merge_scalar(point, scalar_ci),
            "per_class": _merge_per_class(point, per_class_ci, pred["label_columns"]),
            "label_columns": pred["label_columns"],
        }
        with (out_dir / f"{name}.json").open("w") as f:
            json.dump(results[name], f, indent=2)

    if not results:
        print("no predictions loaded; nothing to do")
        return

    _print_comparison(results)
    _print_per_class(results, "pr_auc_per_class", "Per-class PR-AUC  [random baseline = prevalence]")
    _print_prevalence(results)
    _print_per_class(results, "f1_per_class", "Per-class F1")

    with (out_dir / "summary.json").open("w") as f:
        json.dump(results, f, indent=2)
    _write_aggregate_csv(results, out_dir / "aggregate.csv")
    _write_per_class_csv(results, "pr_auc_per_class", out_dir / "per_class_pr_auc.csv")
    _write_per_class_csv(results, "f1_per_class", out_dir / "per_class_f1.csv")

    # --- Pairwise paired bootstrap ---
    if len(results) >= 2:
        print("\n" + "=" * 80)
        print("PAIRWISE PAIRED BOOTSTRAP (A - B)")
        print("=" * 80)
        label_cols = next(iter(results.values()))["label_columns"]
        pair_dir = out_dir / "pairwise"
        pair_dir.mkdir(parents=True, exist_ok=True)
        all_pairs: dict[str, dict] = {}
        for name_a, name_b in combinations(results.keys(), 2):
            pa, ta = pooled[name_a]
            pb, tb = pooled[name_b]
            if pa.shape != pb.shape or ta.shape != tb.shape:
                print(f"[skip pair] {name_a} vs {name_b}: shape mismatch")
                continue
            if not torch.equal(ta, tb):
                print(f"[warn] {name_a} and {name_b} targets differ; pairing uses {name_a}'s")
            comp = compare_paired(
                pa, pb, ta, label_cols,
                threshold=threshold, n_boot=n_boot, seed=seed,
            )
            pk = f"{name_a}_vs_{name_b}"
            all_pairs[pk] = comp
            with (pair_dir / f"{pk}.json").open("w") as f:
                json.dump(comp, f, indent=2)
            _print_pairwise(name_a, name_b, comp, label_cols)
        _write_pairwise_csv(all_pairs, label_cols, pair_dir / "pairwise_pr_auc.csv", "pr_auc_per_class")
        _write_pairwise_csv(all_pairs, label_cols, pair_dir / "pairwise_f1.csv", "f1_per_class")
        _write_pairwise_macro_csv(all_pairs, pair_dir / "pairwise_macro.csv")

    print(f"\nresults -> {out_dir}")


if __name__ == "__main__":
    main()
