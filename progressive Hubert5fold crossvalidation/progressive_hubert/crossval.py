"""5-fold cross-validation across three training strategies."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import KFold

from progressive_hubert.data import load_datasets
from progressive_hubert.experiments import run_overlap_fold, run_progressive_fold, run_single_fold
from progressive_hubert.plotting import generate_cv_figures
from progressive_hubert.train import save_label_map


def _split_fold(df: pd.DataFrame, train_idx: np.ndarray, test_idx: np.ndarray):
    return df.iloc[train_idx].copy(), df.iloc[test_idx].copy()


def run_5fold_cv(
    single_path: str,
    overlap_path: str,
    output_dir: Path,
    cfg: dict,
) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_folds = int(cfg.get("n_folds", 5))
    seed = int(cfg.get("seed", 42))

    single_df, overlap_df, label_space = load_datasets(single_path, overlap_path)
    save_label_map(label_space, output_dir / "label_space")

    kf_single = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    kf_overlap = KFold(n_splits=n_folds, shuffle=True, random_state=seed + 1)

    single_splits = list(kf_single.split(single_df))
    overlap_splits = list(kf_overlap.split(overlap_df))

    all_fold_summaries = []
    cv_root = output_dir / "cv_5fold"
    cv_root.mkdir(parents=True, exist_ok=True)

    for fold in range(n_folds):
        print(f"\n{'='*60}\nFOLD {fold + 1}/{n_folds}\n{'='*60}")
        fold_dir = cv_root / f"fold_{fold}"

        s_train_idx, s_val_idx = single_splits[fold]
        o_train_idx, o_test_idx = overlap_splits[fold]

        single_train, single_val = _split_fold(single_df, s_train_idx, s_val_idx)
        overlap_train, overlap_test = _split_fold(overlap_df, o_train_idx, o_test_idx)

        print(
            f"Sizes | single train/val: {len(single_train)}/{len(single_val)} | "
            f"overlap train/test: {len(overlap_train)}/{len(overlap_test)}"
        )

        fold_result = {"fold": fold, "strategies": {}}

        print("\n>>> single-species")
        r_single = run_single_fold(
            single_train, single_val, overlap_test, label_space, fold_dir, device, cfg
        )
        fold_result["strategies"]["single-species"] = {
            "test_loss": r_single["test_loss"],
            **{f"test_{k}": v for k, v in r_single["test_metrics"].items()},
        }

        print("\n>>> overlap-only")
        r_overlap = run_overlap_fold(
            overlap_train, overlap_test, single_train, label_space, fold_dir, device, cfg
        )
        fold_result["strategies"]["overlap-only"] = {
            "test_loss": r_overlap["test_loss"],
            **{f"test_{k}": v for k, v in r_overlap["test_metrics"].items()},
        }

        print("\n>>> progressive")
        r_prog = run_progressive_fold(
            single_train, single_val, overlap_train, overlap_test,
            label_space, fold_dir, device, cfg,
        )
        fold_result["strategies"]["progressive"] = {
            "test_loss": r_prog["test_loss"],
            **{f"test_{k}": v for k, v in r_prog["test_metrics"].items()},
        }

        all_fold_summaries.append(fold_result)
        with (fold_dir / "fold_summary.json").open("w", encoding="utf-8") as f:
            json.dump(fold_result, f, indent=2)

    aggregate = _aggregate_fold_metrics(all_fold_summaries)
    with (cv_root / "cv_aggregate.json").open("w", encoding="utf-8") as f:
        json.dump({"folds": all_fold_summaries, "aggregate": aggregate}, f, indent=2)

    _write_aggregate_csv(aggregate, cv_root / "cv_results_5fold.csv")
    generate_cv_figures(cv_root, n_folds)

    print("\n========== 5-Fold CV Aggregate (mean ± std) ==========")
    for strategy, stats in aggregate.items():
        print(
            f"{strategy}: test_micro_f1={stats['test_micro_f1_mean']:.4f} "
            f"± {stats['test_micro_f1_std']:.4f} | "
            f"test_subset_acc={stats['test_subset_accuracy_mean']:.4f} "
            f"± {stats['test_subset_accuracy_std']:.4f}"
        )
    print(f"\nResults: {cv_root}")
    return {"folds": all_fold_summaries, "aggregate": aggregate, "output_dir": str(cv_root)}


def _aggregate_fold_metrics(fold_summaries: list[dict]) -> dict:
    strategies = ["single-species", "overlap-only", "progressive"]
    metrics = ["test_loss", "test_micro_f1", "test_macro_f1", "test_subset_accuracy"]
    out = {}
    for strategy in strategies:
        out[strategy] = {}
        for metric in metrics:
            values = [
                f["strategies"][strategy][metric]
                for f in fold_summaries
                if metric in f["strategies"][strategy]
            ]
            if values:
                out[strategy][f"{metric}_mean"] = float(np.mean(values))
                out[strategy][f"{metric}_std"] = float(np.std(values))
    return out


def _write_aggregate_csv(aggregate: dict, path: Path) -> None:
    rows = []
    for strategy, stats in aggregate.items():
        row = {"strategy": strategy}
        row.update(stats)
        rows.append(row)
    pd.DataFrame(rows).to_csv(path, index=False)
