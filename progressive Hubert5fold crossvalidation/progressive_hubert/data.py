"""Dataset discovery and multi-label encoding."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterable

import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer


AUDIO_EXTENSIONS = (".wav", ".flac")


def load_single_species(base_path: str) -> pd.DataFrame:
    """Load recordings with one species per file (folder name = species)."""
    data = []
    if not os.path.isdir(base_path):
        raise FileNotFoundError(f"Single-species path not found: {base_path}")

    for species in sorted(os.listdir(base_path)):
        species_path = os.path.join(base_path, species)
        if not os.path.isdir(species_path):
            continue
        for file in sorted(os.listdir(species_path)):
            if file.lower().endswith(AUDIO_EXTENSIONS):
                data.append(
                    {"path": os.path.join(species_path, file), "species": [species]}
                )
    return pd.DataFrame(data)


def load_overlapping(base_path: str) -> pd.DataFrame:
    """Load multi-species recordings (folder name encodes species, e.g. A_B)."""
    data = []
    if not os.path.isdir(base_path):
        raise FileNotFoundError(f"Overlapping-species path not found: {base_path}")

    for folder in sorted(os.listdir(base_path)):
        folder_path = os.path.join(base_path, folder)
        if not os.path.isdir(folder_path):
            continue
        species = [s.split("(")[0].strip() for s in folder.split("_")]
        for file in sorted(os.listdir(folder_path)):
            if file.lower().endswith(AUDIO_EXTENSIONS):
                data.append(
                    {"path": os.path.join(folder_path, file), "species": species}
                )
    return pd.DataFrame(data)


@dataclass
class LabelSpace:
    species_names: list[str]
    species_to_idx: dict[str, int]
    mlb: MultiLabelBinarizer

    def encode(self, species_lists: Iterable[list[str]]) -> list:
        return [
            self.mlb.transform([[self.species_to_idx[sp] for sp in sp_list]])[0]
            for sp_list in species_lists
        ]


def build_label_space(single_df: pd.DataFrame, overlap_df: pd.DataFrame) -> LabelSpace:
    """Build a shared multi-label space from both dataset splits."""
    all_species = sorted(
        {
            sp
            for sublist in single_df["species"].tolist() + overlap_df["species"].tolist()
            for sp in sublist
        }
    )
    if not all_species:
        raise ValueError("No species found. Check dataset paths and folder layout.")

    species_to_idx = {sp: i for i, sp in enumerate(all_species)}
    mlb = MultiLabelBinarizer(classes=list(range(len(all_species))))
    mlb.fit([[i] for i in range(len(all_species))])
    return LabelSpace(all_species, species_to_idx, mlb)


def attach_binary_labels(df: pd.DataFrame, label_space: LabelSpace) -> pd.DataFrame:
    out = df.copy()
    out["binary_labels"] = label_space.encode(out["species"])
    return out


def load_datasets(single_path: str, overlap_path: str) -> tuple[pd.DataFrame, pd.DataFrame, LabelSpace]:
    single_df = load_single_species(single_path)
    overlap_df = load_overlapping(overlap_path)
    label_space = build_label_space(single_df, overlap_df)
    single_df = attach_binary_labels(single_df, label_space)
    overlap_df = attach_binary_labels(overlap_df, label_space)
    return single_df, overlap_df, label_space
