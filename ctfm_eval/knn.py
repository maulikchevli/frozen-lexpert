from __future__ import annotations

import torch


def knn_multilabel_probs(
    train_emb: torch.Tensor,   # [N_tr, D]
    train_lab: torch.Tensor,   # [N_tr, C]  (0/1 floats)
    test_emb: torch.Tensor,    # [N_te, D]
    k: int,
    weighting: str = "uniform",  # "uniform" | "distance"
) -> torch.Tensor:
    """Cosine-similarity kNN for multilabel. Returns probs [N_te, C] in [0,1].

    The probability of class c for a test point = (weighted) fraction of its
    k nearest training neighbors that have c=1.
    """
    if k < 1:
        raise ValueError("k must be >= 1")
    k_eff = min(k, train_emb.shape[0])

    tr = torch.nn.functional.normalize(train_emb, dim=1)
    te = torch.nn.functional.normalize(test_emb, dim=1)
    sim = te @ tr.T                                  # [N_te, N_tr], in [-1, 1]
    top_sim, top_idx = sim.topk(k=k_eff, dim=1)      # [N_te, k]

    neigh_lab = train_lab[top_idx]                   # [N_te, k, C]
    if weighting == "uniform":
        return neigh_lab.mean(dim=1)
    if weighting == "distance":
        # Map similarity in [-1,1] -> weight in (0, 2]; +eps to avoid div-by-zero.
        w = (top_sim + 1.0).clamp_min(1e-6)          # [N_te, k]
        w = w / w.sum(dim=1, keepdim=True)
        return (neigh_lab * w.unsqueeze(-1)).sum(dim=1)
    raise ValueError(f"unknown weighting {weighting!r}")


def multilabel_metrics(
    probs: torch.Tensor,   # [N, C] in [0,1]
    target: torch.Tensor,  # [N, C] 0/1
    threshold: float = 0.5,
) -> dict[str, float | list[float]]:
    """Threshold-based multilabel metrics + threshold-free per-class PR-AUC.

    PR-AUC (= average precision) is the right summary under heavy class
    imbalance: ROC-AUC is dominated by the (huge) negative pool and looks
    artificially OK on rare classes; PR-AUC stays sensitive to false positives
    on the rare positive set. Random baseline for PR-AUC equals the class
    prevalence (not 0.5), so always read PR-AUC alongside the prevalence.
    """
    eps = 1e-8
    pred = (probs >= threshold).float()

    tp = (pred * target).sum(dim=0)
    fp = (pred * (1 - target)).sum(dim=0)
    fn = ((1 - pred) * target).sum(dim=0)

    f1_per = (2 * tp) / (2 * tp + fp + fn + eps)             # [C]
    macro_f1 = float(torch.nan_to_num(f1_per).mean())
    micro_f1 = float((2 * tp.sum()) / (2 * tp.sum() + fp.sum() + fn.sum() + eps))
    subset_acc = float((pred == target).all(dim=1).float().mean())
    hamming_acc = float((pred == target).float().mean())

    pr_auc_per = _average_precision_per_class(probs, target)  # [C], NaN where undefined
    macro_pr_auc = float(torch.nanmean(pr_auc_per))

    prevalence_per = target.mean(dim=0)                       # [C]

    return {
        "macro_f1": macro_f1,
        "micro_f1": micro_f1,
        "macro_pr_auc": macro_pr_auc,
        "subset_accuracy": subset_acc,
        "hamming_accuracy": hamming_acc,
        "f1_per_class": f1_per.tolist(),
        "pr_auc_per_class": pr_auc_per.tolist(),
        "prevalence_per_class": prevalence_per.tolist(),
    }


def _average_precision_per_class(scores: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Per-class average precision (PR-AUC), NaN if class has no positives.

    AP = sum_k (R_k - R_{k-1}) * P_k, with samples sorted by descending score.
    Ties are handled by sorting (stable enough for our use).
    """
    n_classes = scores.shape[1]
    out = torch.full((n_classes,), float("nan"))
    for c in range(n_classes):
        s = scores[:, c]
        y = target[:, c]
        n_pos = float(y.sum())
        if n_pos == 0:
            continue
        order = torch.argsort(s, descending=True)
        y_sorted = y[order]
        tp_cum = torch.cumsum(y_sorted, dim=0)                # [N]
        ks = torch.arange(1, len(y) + 1, dtype=s.dtype)
        precision = tp_cum / ks
        recall = tp_cum / n_pos
        recall_prev = torch.cat([torch.zeros(1, dtype=recall.dtype), recall[:-1]])
        out[c] = float(((recall - recall_prev) * precision).sum())
    return out
