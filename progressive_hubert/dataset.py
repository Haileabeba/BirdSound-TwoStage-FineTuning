"""PyTorch dataset wrapper for encoded audio batches."""

from __future__ import annotations

import torch
from torch.utils.data import Dataset


class AudioDataset(Dataset):
    def __init__(self, encoded_dataset, labels):
        self.encoded = encoded_dataset
        self.labels = labels.reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict:
        input_values = self.encoded["input_values"][idx]
        attention_mask = self.encoded["attention_mask"][idx]
        if not torch.is_tensor(input_values):
            input_values = torch.tensor(input_values, dtype=torch.float)
        if not torch.is_tensor(attention_mask):
            attention_mask = torch.tensor(attention_mask, dtype=torch.long)

        return {
            "input_values": input_values,
            "attention_mask": attention_mask,
            "labels": torch.tensor(self.labels.iloc[idx], dtype=torch.float),
        }
