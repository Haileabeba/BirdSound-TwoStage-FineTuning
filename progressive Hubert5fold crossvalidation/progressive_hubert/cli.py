"""CLI: 5-fold CV and single-run experiments."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from progressive_hubert.crossval import run_5fold_cv


def load_config(path: Path | None) -> dict:
    if not path:
        return {}
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="ProComment 5-fold CV: fair HuBERT strategy comparison.",
    )
    parser.add_argument(
        "--experiment",
        required=True,
        choices=["cv5fold", "cv5fold-quick"],
        help="cv5fold = full 5-fold; cv5fold-quick = 2 folds for smoke test",
    )
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--single-path", type=str)
    parser.add_argument("--overlap-path", type=str)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--no-match-total-updates", action="store_true")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    single_path = args.single_path or cfg.get("single_species_path")
    overlap_path = args.overlap_path or cfg.get("overlapping_species_path")
    if not single_path or not overlap_path:
        parser.error("Set paths in --config or --single-path / --overlap-path")

    cfg.setdefault("match_total_updates", not args.no_match_total_updates)
    cfg.setdefault("n_folds", 5)
    cfg.setdefault("seed", 42)
    cfg.setdefault("phase1_epochs", 50)
    cfg.setdefault("phase2_epochs", 50)
    cfg.setdefault("batch_size", 16)
    cfg.setdefault("phase1_fraction", 0.5)
    cfg.setdefault("phase1_learning_rate", 1e-4)
    cfg.setdefault("phase2_learning_rate", 5e-6)
    cfg.setdefault("learning_rate", 1e-4)

    if args.experiment == "cv5fold-quick":
        cfg["n_folds"] = 2
        cfg["phase1_epochs"] = cfg.get("quick_phase1_epochs", 2)
        cfg["phase2_epochs"] = cfg.get("quick_phase2_epochs", 2)

    output_dir = Path(args.output_dir or cfg.get("output_dir", "outputs"))
    run_5fold_cv(single_path, overlap_path, output_dir, cfg)


if __name__ == "__main__":
    main()
