"""Command-line entry point for all experiments."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from progressive_hubert.experiments import run_overlap_only, run_progressive, run_single_species


def load_config(path: Path | None) -> dict:
    if path is None:
        return {}
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Progressive HuBERT: multi-label species classification experiments.",
    )
    parser.add_argument(
        "--experiment",
        required=True,
        choices=["single", "overlap", "progressive"],
        help="Training regime to run.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="YAML config file (see configs/default.yaml).",
    )
    parser.add_argument("--single-path", type=str, help="Path to single-species dataset root.")
    parser.add_argument("--overlap-path", type=str, help="Path to overlapping-species dataset root.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    cfg = load_config(args.config)

    single_path = args.single_path or cfg.get("single_species_path")
    overlap_path = args.overlap_path or cfg.get("overlapping_species_path")
    if not single_path or not overlap_path:
        parser.error("Provide --single-path and --overlap-path, or set them in --config.")

    output_dir = args.output_dir or Path(cfg.get("output_dir", "outputs"))
    common = {
        "single_path": single_path,
        "overlap_path": overlap_path,
        "output_dir": Path(output_dir),
        "batch_size": args.batch_size or cfg.get("batch_size", 16),
        "seed": args.seed or cfg.get("seed", 42),
        "encode_batch_size": cfg.get("encode_batch_size", 8),
        "train_frac": cfg.get("train_frac", 0.8),
    }

    if args.experiment == "single":
        run_single_species(
            epochs=args.epochs or cfg.get("epochs", 50),
            learning_rate=args.learning_rate or cfg.get("learning_rate", 1e-4),
            **common,
        )
    elif args.experiment == "overlap":
        run_overlap_only(
            epochs=args.epochs or cfg.get("epochs", 50),
            learning_rate=args.learning_rate or cfg.get("learning_rate", 1e-4),
            **common,
        )
    else:
        run_progressive(
            phase1_epochs=cfg.get("phase1_epochs", args.epochs or 50),
            phase2_epochs=cfg.get("phase2_epochs", args.epochs or 50),
            phase1_lr=cfg.get("phase1_learning_rate", args.learning_rate or 1e-4),
            phase2_lr=cfg.get("phase2_learning_rate", 5e-6),
            **common,
        )


if __name__ == "__main__":
    main()
