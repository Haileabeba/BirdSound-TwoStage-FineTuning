"""Per-fold training for three strategies (fair optimizer-step budget)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from progressive_hubert.audio import encode_dataset, get_feature_extractor
from progressive_hubert.budget import build_budget
from progressive_hubert.data import LabelSpace
from progressive_hubert.dataset import AudioDataset
from progressive_hubert.train import (
    build_model,
    evaluate,
    fit,
    load_checkpoint,
    save_step_history,
)


def _loader(encoded, labels, batch_size: int, shuffle: bool = False) -> DataLoader:
    return DataLoader(AudioDataset(encoded, labels), batch_size=batch_size, shuffle=shuffle)


def _merge_phases(parts: list[dict], strategy: str) -> dict:
    steps = []
    for p in parts:
        steps.extend(p.get("step_history", []))
    return {
        "strategy": strategy,
        "step_history": steps,
        "global_step_end": parts[-1]["global_step_end"] if parts else 0,
    }


def run_single_fold(
    single_train: pd.DataFrame,
    single_val: pd.DataFrame,
    overlap_test: pd.DataFrame,
    label_space: LabelSpace,
    fold_dir: Path,
    device: torch.device,
    cfg: dict,
) -> dict:
    fe = get_feature_extractor()
    enc_bs = cfg.get("encode_batch_size", 8)
    bs = cfg.get("batch_size", 16)
    match = cfg.get("match_total_updates", True)

    train_enc = encode_dataset(single_train, fe, enc_bs)
    val_enc = encode_dataset(single_val, fe, enc_bs)
    test_enc = encode_dataset(overlap_test, fe, enc_bs)

    train_loader = _loader(train_enc, single_train["binary_labels"], bs, True)
    overlap_dummy = _loader(train_enc, single_train["binary_labels"], bs, True)
    budget = build_budget(train_loader, overlap_dummy, cfg["phase1_epochs"], cfg["phase2_epochs"])
    max_steps = budget.progressive_total_steps if match else None

    model = build_model(len(label_space.species_names), device)
    out = fold_dir / "single_species"
    result = fit(
        model,
        train_loader,
        _loader(val_enc, single_val["binary_labels"], bs),
        device,
        cfg.get("learning_rate", 1e-4),
        out / "best.pt",
        strategy="single-species",
        phase="single_train",
        max_optimizer_steps=max_steps,
        eval_every_n_steps=cfg.get("eval_every_n_steps"),
    )
    load_checkpoint(model, out / "best.pt")
    test_loss, test_m = evaluate(
        model,
        _loader(test_enc, overlap_test["binary_labels"], bs),
        nn.BCEWithLogitsLoss(),
        device,
    )
    result["test_loss"] = test_loss
    result["test_metrics"] = test_m
    result["budget"] = budget.summary(match, cfg.get("phase1_fraction", 0.5))
    save_step_history(result, out / "history.json")
    return result


def run_overlap_fold(
    overlap_train: pd.DataFrame,
    overlap_test: pd.DataFrame,
    single_train: pd.DataFrame,
    label_space: LabelSpace,
    fold_dir: Path,
    device: torch.device,
    cfg: dict,
) -> dict:
    fe = get_feature_extractor()
    enc_bs = cfg.get("encode_batch_size", 8)
    bs = cfg.get("batch_size", 16)
    match = cfg.get("match_total_updates", True)

    train_enc = encode_dataset(overlap_train, fe, enc_bs)
    test_enc = encode_dataset(overlap_test, fe, enc_bs)
    single_enc = encode_dataset(single_train, fe, enc_bs)

    train_loader = _loader(train_enc, overlap_train["binary_labels"], bs, True)
    single_loader = _loader(single_enc, single_train["binary_labels"], bs, True)
    budget = build_budget(single_loader, train_loader, cfg["phase1_epochs"], cfg["phase2_epochs"])
    max_steps = budget.progressive_total_steps if match else None

    model = build_model(len(label_space.species_names), device)
    out = fold_dir / "overlap_only"
    result = fit(
        model,
        train_loader,
        _loader(test_enc, overlap_test["binary_labels"], bs),
        device,
        cfg.get("learning_rate", 1e-4),
        out / "best.pt",
        strategy="overlap-only",
        phase="overlap_train",
        max_optimizer_steps=max_steps,
        eval_every_n_steps=cfg.get("eval_every_n_steps"),
    )
    load_checkpoint(model, out / "best.pt")
    test_loss, test_m = evaluate(
        model, _loader(test_enc, overlap_test["binary_labels"], bs), nn.BCEWithLogitsLoss(), device
    )
    result["test_loss"] = test_loss
    result["test_metrics"] = test_m
    result["budget"] = budget.summary(match, cfg.get("phase1_fraction", 0.5))
    save_step_history(result, out / "history.json")
    return result


def run_progressive_fold(
    single_train: pd.DataFrame,
    single_val: pd.DataFrame,
    overlap_train: pd.DataFrame,
    overlap_test: pd.DataFrame,
    label_space: LabelSpace,
    fold_dir: Path,
    device: torch.device,
    cfg: dict,
) -> dict:
    fe = get_feature_extractor()
    enc_bs = cfg.get("encode_batch_size", 8)
    bs = cfg.get("batch_size", 16)
    match = cfg.get("match_total_updates", True)
    frac = cfg.get("phase1_fraction", 0.5)

    train_s_enc = encode_dataset(single_train, fe, enc_bs)
    val_s_enc = encode_dataset(single_val, fe, enc_bs)
    train_o_enc = encode_dataset(overlap_train, fe, enc_bs)
    test_o_enc = encode_dataset(overlap_test, fe, enc_bs)

    single_loader = _loader(train_s_enc, single_train["binary_labels"], bs, True)
    overlap_loader = _loader(train_o_enc, overlap_train["binary_labels"], bs, True)
    budget = build_budget(single_loader, overlap_loader, cfg["phase1_epochs"], cfg["phase2_epochs"])
    p1_steps, p2_steps = budget.phase_steps_for_progressive(match, frac)

    model = build_model(len(label_space.species_names), device)
    out = fold_dir / "progressive"

    phase1 = fit(
        model,
        single_loader,
        _loader(val_s_enc, single_val["binary_labels"], bs),
        device,
        cfg.get("phase1_learning_rate", 1e-4),
        out / "phase1_best.pt",
        strategy="progressive",
        phase="phase1_single",
        max_optimizer_steps=p1_steps,
        eval_every_n_steps=cfg.get("eval_every_n_steps"),
    )
    load_checkpoint(model, out / "phase1_best.pt")
    phase2 = fit(
        model,
        overlap_loader,
        _loader(test_o_enc, overlap_test["binary_labels"], bs),
        device,
        cfg.get("phase2_learning_rate", 5e-6),
        out / "best.pt",
        strategy="progressive",
        phase="phase2_overlap",
        max_optimizer_steps=p2_steps,
        global_step_offset=phase1["global_step_end"],
        eval_every_n_steps=cfg.get("eval_every_n_steps"),
    )
    combined = _merge_phases([phase1, phase2], "progressive")
    load_checkpoint(model, out / "best.pt")
    test_loss, test_m = evaluate(
        model, _loader(test_o_enc, overlap_test["binary_labels"], bs), nn.BCEWithLogitsLoss(), device
    )
    combined["test_loss"] = test_loss
    combined["test_metrics"] = test_m
    combined["budget"] = budget.summary(match, frac)
    save_step_history(combined, out / "history.json")
    return combined
