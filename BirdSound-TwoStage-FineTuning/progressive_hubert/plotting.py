"""Figures: per-fold histories and 5-fold aggregate plots."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

STRATEGY_LABEL = {
    "single-species": "single",
    "overlap-only": "overlap",
    "progressive": "two-stage",
}

STYLE = {
    "two-stage": {"train": "black", "val": "red"},
    "single": {"train": "blue", "val": "green"},
    "overlap": {"train": "magenta", "val": "olive"},
}


def _load_eval_series(path: Path) -> tuple[list[int], list[float], list[float]]:
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    steps, losses, accs = [], [], []
    for r in data.get("step_history", []):
        if r.get("record_type") != "eval":
            continue
        steps.append(r["global_optimizer_step"])
        losses.append(r["eval_loss"])
        accs.append(r.get("eval_subset_accuracy", r.get("eval_micro_f1", 0)))
    return steps, losses, accs


def plot_fold_comparison(cv_root: Path, fold: int) -> None:
    fold_dir = cv_root / f"fold_{fold}"
    fig_dir = cv_root / "figures" / f"fold_{fold}"
    fig_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "two-stage": fold_dir / "progressive" / "history.json",
        "single": fold_dir / "single_species" / "history.json",
        "overlap": fold_dir / "overlap_only" / "history.json",
    }
    for metric, ylabel, fname in [
        ("loss", "Eval loss", "eval_loss_vs_updates.png"),
        ("acc", "Subset accuracy", "eval_accuracy_vs_updates.png"),
    ]:
        fig, ax = plt.subplots(figsize=(9, 5))
        for key, path in paths.items():
            if not path.exists():
                continue
            steps, losses, accs = _load_eval_series(path)
            y = losses if metric == "loss" else accs
            if steps:
                ax.plot(steps, y, label=f"{key}", color=STYLE[key]["val"], alpha=0.85)
        ax.set_xlabel("Optimizer updates")
        ax.set_ylabel(ylabel)
        ax.set_title(f"Fold {fold + 1}: {ylabel} vs updates")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(fig_dir / fname, dpi=150)
        plt.close(fig)


def generate_cv_figures(cv_root: Path, n_folds: int) -> None:
    for fold in range(n_folds):
        plot_fold_comparison(cv_root, fold)

    agg_path = cv_root / "cv_aggregate.json"
    if not agg_path.exists():
        return
    with agg_path.open(encoding="utf-8") as f:
        agg = json.load(f)["aggregate"]

    strategies = ["single-species", "overlap-only", "progressive"]
    labels = [STRATEGY_LABEL.get(s, s) for s in strategies]
    f1_mean = [agg[s]["test_micro_f1_mean"] for s in strategies]
    f1_std = [agg[s]["test_micro_f1_std"] for s in strategies]
    acc_mean = [agg[s].get("test_subset_accuracy_mean", 0) for s in strategies]
    acc_std = [agg[s].get("test_subset_accuracy_std", 0) for s in strategies]

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    x = np.arange(len(labels))
    axes[0].bar(x, f1_mean, yerr=f1_std, capsize=5, color=["#333", "#1f77b4", "#ff69b4"])
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels)
    axes[0].set_ylabel("Micro F1")
    axes[0].set_title("5-fold CV: test Micro F1 (mean ± std)")
    axes[0].grid(True, axis="y", alpha=0.3)

    axes[1].bar(x, acc_mean, yerr=acc_std, capsize=5, color=["#333", "#1f77b4", "#ff69b4"])
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels)
    axes[1].set_ylabel("Subset accuracy")
    axes[1].set_title("5-fold CV: test accuracy (mean ± std)")
    axes[1].grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    out = cv_root / "figures" / "cv5_aggregate_metrics.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved aggregate figure: {out}")
