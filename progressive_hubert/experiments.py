"""Three training regimes: single-species, overlap-only, and progressive."""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader

from progressive_hubert.audio import encode_dataset, get_feature_extractor
from progressive_hubert.data import load_datasets
from progressive_hubert.dataset import AudioDataset
from progressive_hubert.train import (
    build_model,
    evaluate,
    fit,
    load_checkpoint,
    save_label_map,
)
import torch.nn as nn


def _make_loader(encoded, labels, batch_size: int, shuffle: bool = False) -> DataLoader:
    return DataLoader(
        AudioDataset(encoded, labels),
        batch_size=batch_size,
        shuffle=shuffle,
    )


def run_single_species(
    single_path: str,
    overlap_path: str,
    output_dir: Path,
    *,
    epochs: int = 50,
    batch_size: int = 16,
    learning_rate: float = 1e-4,
    train_frac: float = 0.8,
    seed: int = 42,
    encode_batch_size: int = 8,
) -> dict:
    """Train on single-species data; evaluate on overlapping test set."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    single_df, overlap_df, label_space = load_datasets(single_path, overlap_path)

    train_single = single_df.sample(frac=train_frac, random_state=seed)
    val_single = single_df.drop(train_single.index)
    test_overlap = overlap_df

    feature_extractor = get_feature_extractor()
    train_enc = encode_dataset(train_single, feature_extractor, encode_batch_size)
    val_enc = encode_dataset(val_single, feature_extractor, encode_batch_size)
    test_enc = encode_dataset(test_overlap, feature_extractor, encode_batch_size)

    model = build_model(len(label_space.species_names), device)
    out = output_dir / "single_species"
    save_label_map(label_space, out)

    result = fit(
        model,
        _make_loader(train_enc, train_single["binary_labels"], batch_size, shuffle=True),
        _make_loader(val_enc, val_single["binary_labels"], batch_size),
        device,
        epochs,
        learning_rate,
        out / "best_single_species.pt",
    )

    load_checkpoint(model, out / "best_single_species.pt")
    test_loader = _make_loader(test_enc, test_overlap["binary_labels"], batch_size)
    test_loss, test_metrics = evaluate(model, test_loader, nn.BCEWithLogitsLoss(), device)
    result["test_loss"] = test_loss
    result["test_metrics"] = test_metrics
    print(f"Test on overlapping | loss={test_loss:.4f} | micro_f1={test_metrics['micro_f1']:.4f}")
    return result


def run_overlap_only(
    single_path: str,
    overlap_path: str,
    output_dir: Path,
    *,
    epochs: int = 50,
    batch_size: int = 16,
    learning_rate: float = 1e-4,
    train_frac: float = 0.8,
    seed: int = 42,
    encode_batch_size: int = 8,
) -> dict:
    """Train and test only on overlapping multi-species data."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, overlap_df, label_space = load_datasets(single_path, overlap_path)

    train_overlap = overlap_df.sample(frac=train_frac, random_state=seed)
    test_overlap = overlap_df.drop(train_overlap.index)

    feature_extractor = get_feature_extractor()
    train_enc = encode_dataset(train_overlap, feature_extractor, encode_batch_size)
    test_enc = encode_dataset(test_overlap, feature_extractor, encode_batch_size)

    model = build_model(len(label_space.species_names), device)
    out = output_dir / "overlap_only"
    save_label_map(label_space, out)

    result = fit(
        model,
        _make_loader(train_enc, train_overlap["binary_labels"], batch_size, shuffle=True),
        _make_loader(test_enc, test_overlap["binary_labels"], batch_size),
        device,
        epochs,
        learning_rate,
        out / "best_overlap_only.pt",
    )
    result["test_metrics"] = result["history"][-1] if result["history"] else {}
    return result


def run_progressive(
    single_path: str,
    overlap_path: str,
    output_dir: Path,
    *,
    phase1_epochs: int = 50,
    phase2_epochs: int = 50,
    batch_size: int = 16,
    phase1_lr: float = 1e-4,
    phase2_lr: float = 5e-6,
    train_frac: float = 0.8,
    seed: int = 42,
    encode_batch_size: int = 8,
) -> dict:
    """Phase 1: single-species pretraining. Phase 2: fine-tune on overlapping audio."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    single_df, overlap_df, label_space = load_datasets(single_path, overlap_path)

    train_single = single_df.sample(frac=train_frac, random_state=seed)
    val_single = single_df.drop(train_single.index)
    train_overlap = overlap_df.sample(frac=train_frac, random_state=seed)
    test_overlap = overlap_df.drop(train_overlap.index)

    feature_extractor = get_feature_extractor()
    train_single_enc = encode_dataset(train_single, feature_extractor, encode_batch_size)
    val_single_enc = encode_dataset(val_single, feature_extractor, encode_batch_size)
    train_overlap_enc = encode_dataset(train_overlap, feature_extractor, encode_batch_size)
    test_overlap_enc = encode_dataset(test_overlap, feature_extractor, encode_batch_size)

    model = build_model(len(label_space.species_names), device)
    out = output_dir / "progressive"
    save_label_map(label_space, out)
    phase1_ckpt = out / "best_single_species.pt"

    print("=== Phase 1: single-species pretraining ===")
    phase1 = fit(
        model,
        _make_loader(train_single_enc, train_single["binary_labels"], batch_size, shuffle=True),
        _make_loader(val_single_enc, val_single["binary_labels"], batch_size),
        device,
        phase1_epochs,
        phase1_lr,
        phase1_ckpt,
    )

    print("=== Phase 2: overlapping fine-tuning ===")
    load_checkpoint(model, phase1_ckpt)
    phase2 = fit(
        model,
        _make_loader(train_overlap_enc, train_overlap["binary_labels"], batch_size, shuffle=True),
        _make_loader(test_overlap_enc, test_overlap["binary_labels"], batch_size),
        device,
        phase2_epochs,
        phase2_lr,
        out / "best_progressive.pt",
    )

    return {"phase1": phase1, "phase2": phase2, "checkpoint": str(out / "best_progressive.pt")}
