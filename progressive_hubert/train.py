"""Training loop, evaluation, and checkpointing."""

from __future__ import annotations

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


def _predictions(logits: torch.Tensor, threshold: float = 0.5) -> np.ndarray:
    probs = torch.sigmoid(logits).cpu().numpy()
    return (probs >= threshold).astype(int)


def compute_metrics(logits: torch.Tensor, labels: torch.Tensor, threshold: float = 0.5) -> dict:
    y_true = labels.cpu().numpy().astype(int)
    y_pred = _predictions(logits, threshold)
    return {
        "hamming_loss": float(hamming_loss(y_true, y_pred)),
        "micro_f1": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def train_epoch(
    model: HubertForSequenceClassification,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    for batch in tqdm(dataloader, desc="train", leave=False):
        inputs = batch["input_values"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()
        outputs = model(input_values=inputs, attention_mask=attention_mask)
        loss = criterion(outputs.logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / max(len(dataloader), 1)


@torch.no_grad()
def evaluate(
    model: HubertForSequenceClassification,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    threshold: float = 0.5,
) -> tuple[float, dict]:
    model.eval()
    total_loss = 0.0
    metric_sums = {"hamming_loss": 0.0, "micro_f1": 0.0, "macro_f1": 0.0}
    n_batches = 0

    for batch in tqdm(dataloader, desc="eval", leave=False):
        inputs = batch["input_values"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(input_values=inputs, attention_mask=attention_mask)
        loss = criterion(outputs.logits, labels)
        total_loss += loss.item()

        batch_metrics = compute_metrics(outputs.logits, labels, threshold)
        for key in metric_sums:
            metric_sums[key] += batch_metrics[key]
        n_batches += 1

    avg_loss = total_loss / max(n_batches, 1)
    avg_metrics = {k: v / max(n_batches, 1) for k, v in metric_sums.items()}
    return avg_loss, avg_metrics


def fit(
    model: HubertForSequenceClassification,
    train_loader: DataLoader,
    eval_loader: DataLoader,
    device: torch.device,
    epochs: int,
    learning_rate: float,
    checkpoint_path: Path,
    monitor: str = "val_loss",
) -> dict:
    """Train with early best-checkpoint saving based on eval loss or micro F1."""
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    best_score = float("inf") if monitor == "val_loss" else -float("inf")
    history = []

    for epoch in range(1, epochs + 1):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        eval_loss, eval_metrics = evaluate(model, eval_loader, criterion, device)

        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "eval_loss": eval_loss,
            **{f"eval_{k}": v for k, v in eval_metrics.items()},
        }
        history.append(record)

        print(
            f"Epoch {epoch}/{epochs} | "
            f"train_loss={train_loss:.4f} | eval_loss={eval_loss:.4f} | "
            f"micro_f1={eval_metrics['micro_f1']:.4f} | macro_f1={eval_metrics['macro_f1']:.4f}"
        )

        if monitor == "val_loss":
            improved = eval_loss < best_score
            if improved:
                best_score = eval_loss
        else:
            improved = eval_metrics["micro_f1"] > best_score
            if improved:
                best_score = eval_metrics["micro_f1"]

        if improved:
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "eval_loss": eval_loss,
                    "eval_metrics": eval_metrics,
                },
                checkpoint_path,
            )

    return {"history": history, "best_checkpoint": str(checkpoint_path)}


def load_checkpoint(model: HubertForSequenceClassification, checkpoint_path: Path) -> None:
    state = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(state, dict) and "model_state_dict" in state:
        model.load_state_dict(state["model_state_dict"])
    else:
        model.load_state_dict(state)


def save_label_map(label_space: LabelSpace, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    mapping_path = output_dir / "species_labels.txt"
    with mapping_path.open("w", encoding="utf-8") as f:
        for idx, name in enumerate(label_space.species_names):
            f.write(f"{idx}\t{name}\n")
