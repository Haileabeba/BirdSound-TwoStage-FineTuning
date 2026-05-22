"""Training with per-optimizer-step logging."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, hamming_loss
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import HubertForSequenceClassification

from progressive_hubert.audio import HUBERT_MODEL_ID
from progressive_hubert.data import LabelSpace


def build_model(num_labels: int, device: torch.device) -> HubertForSequenceClassification:
    model = HubertForSequenceClassification.from_pretrained(
        HUBERT_MODEL_ID,
        num_labels=num_labels,
        problem_type="multi_label_classification",
        ignore_mismatched_sizes=True,
    )
    return model.to(device)


def compute_metrics(logits: torch.Tensor, labels: torch.Tensor, threshold: float = 0.5) -> dict:
    y_true = labels.cpu().numpy().astype(int)
    y_pred = (torch.sigmoid(logits).cpu().numpy() >= threshold).astype(int)
    return {
        "hamming_loss": float(hamming_loss(y_true, y_pred)),
        "subset_accuracy": float(np.mean(np.all(y_true == y_pred, axis=1))) if y_true.size else 0.0,
        "micro_f1": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


@torch.no_grad()
def evaluate(model, dataloader, criterion, device, threshold=0.5):
    model.eval()
    total_loss = 0.0
    metric_sums = {"hamming_loss": 0.0, "micro_f1": 0.0, "macro_f1": 0.0, "subset_accuracy": 0.0}
    n = 0
    for batch in tqdm(dataloader, desc="eval", leave=False):
        inputs = batch["input_values"].to(device)
        mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        outputs = model(input_values=inputs, attention_mask=mask)
        loss = criterion(outputs.logits, labels)
        total_loss += loss.item()
        m = compute_metrics(outputs.logits, labels, threshold)
        for k in metric_sums:
            metric_sums[k] += m[k]
        n += 1
    avg = {k: v / max(n, 1) for k, v in metric_sums.items()}
    return total_loss / max(n, 1), avg


def fit(
    model,
    train_loader,
    eval_loader,
    device,
    learning_rate,
    checkpoint_path: Path,
    *,
    strategy: str,
    phase: str = "train",
    max_epochs: int | None = None,
    max_optimizer_steps: int | None = None,
    global_step_offset: int = 0,
    eval_every_n_steps: int | None = None,
) -> dict:
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    best_score = float("inf")
    step_history = []
    global_step = global_step_offset
    epoch = 0
    stop = False

    while not stop:
        epoch += 1
        model.train()
        batch_count = 0
        epoch_train_loss = 0.0

        for batch in tqdm(train_loader, desc=f"{strategy}/{phase} e{epoch}", leave=False):
            inputs = batch["input_values"].to(device)
            mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            optimizer.zero_grad()
            outputs = model(input_values=inputs, attention_mask=mask)
            loss = criterion(outputs.logits, labels)
            loss.backward()
            optimizer.step()
            global_step += 1
            batch_count += 1
            epoch_train_loss += loss.item()

            end_epoch = batch_count == len(train_loader)
            run_eval = (eval_every_n_steps and global_step % eval_every_n_steps == 0) or (
                not eval_every_n_steps and end_epoch
            )
            if run_eval:
                train_loss, train_m = evaluate(model, train_loader, criterion, device)
                eval_loss, eval_m = evaluate(model, eval_loader, criterion, device)
                step_history.append(
                    {
                        "record_type": "eval",
                        "strategy": strategy,
                        "phase": phase,
                        "epoch": epoch,
                        "global_optimizer_step": global_step,
                        "train_loss": train_loss,
                        "eval_loss": eval_loss,
                        **{f"train_{k}": v for k, v in train_m.items()},
                        **{f"eval_{k}": v for k, v in eval_m.items()},
                    }
                )
                if eval_loss < best_score:
                    best_score = eval_loss
                    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                    torch.save(
                        {"model_state_dict": model.state_dict(), "global_optimizer_step": global_step},
                        checkpoint_path,
                    )

            if max_optimizer_steps and global_step >= max_optimizer_steps:
                stop = True
                break

        if max_epochs and epoch >= max_epochs:
            stop = True
        if max_optimizer_steps and global_step >= max_optimizer_steps:
            stop = True

    return {
        "strategy": strategy,
        "phase": phase,
        "global_step_end": global_step,
        "step_history": step_history,
        "best_checkpoint": str(checkpoint_path),
    }


def save_step_history(result: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)


def load_checkpoint(model, path: Path) -> None:
    state = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model_state_dict"] if isinstance(state, dict) else state)


def save_label_map(label_space: LabelSpace, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "species_labels.txt").open("w", encoding="utf-8") as f:
        for i, name in enumerate(label_space.species_names):
            f.write(f"{i}\t{name}\n")
