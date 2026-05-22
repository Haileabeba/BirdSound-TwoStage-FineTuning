"""Audio loading and HuBERT feature extraction."""

from __future__ import annotations

import librosa
from datasets import Dataset
from transformers import Wav2Vec2FeatureExtractor

HUBERT_MODEL_ID = "facebook/hubert-base-ls960"
SAMPLE_RATE = 16000


def load_audio(example: dict) -> dict:
    speech, _ = librosa.load(example["path"], sr=SAMPLE_RATE, mono=True)
    example["speech"] = speech
    return example


def get_feature_extractor() -> Wav2Vec2FeatureExtractor:
    return Wav2Vec2FeatureExtractor.from_pretrained(HUBERT_MODEL_ID)


def encode_dataset(
    df,
    feature_extractor: Wav2Vec2FeatureExtractor,
    batch_size: int = 8,
) -> Dataset:
    """Load waveforms and compute HuBERT input features."""
    dataset = Dataset.from_pandas(df[["path", "species", "binary_labels"]])

    def tokenize_batch(batch):
        return feature_extractor(
            batch["speech"],
            sampling_rate=SAMPLE_RATE,
            padding=True,
            return_tensors="pt",
        )

    dataset = dataset.map(load_audio)
    encoded = dataset.map(
        tokenize_batch,
        batched=True,
        batch_size=batch_size,
        remove_columns=["path", "species", "speech"],
    )
    return encoded
