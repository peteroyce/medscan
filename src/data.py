"""
Dataset loading and preprocessing for PubMed 200k RCT.

Handles downloading via HuggingFace datasets, label encoding,
tokenization, and stratified splits.
"""

import logging
from pathlib import Path

import torch
import yaml
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)

LABEL_MAP = {
    "BACKGROUND": 0,
    "OBJECTIVE": 1,
    "METHODS": 2,
    "RESULTS": 3,
    "CONCLUSIONS": 4,
}

ID_TO_LABEL = {v: k for k, v in LABEL_MAP.items()}


def load_config(config_path: str = "configs/train_config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_pubmed_rct(config: dict | None = None):
    """
    Load PubMed 200k RCT dataset.

    Falls back to the smaller 20k version if 200k isn't available.
    Returns train/val/test splits as HuggingFace datasets.
    """
    config = config or load_config()

    logger.info("Loading PubMed RCT dataset...")
    try:
        ds = load_dataset("pubmed_rct", "200k")
    except Exception:
        logger.warning("200k not available, falling back to 20k")
        ds = load_dataset("pubmed_rct", "20k")

    return ds["train"], ds["validation"], ds["test"]


def encode_labels(example):
    """Map string labels to integers."""
    example["label"] = LABEL_MAP.get(example["label"], 0)
    return example


class SentenceDataset(torch.utils.data.Dataset):
    """PyTorch dataset wrapping tokenized PubMed sentences."""

    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


def prepare_datasets(config: dict | None = None):
    """
    Full data preparation pipeline.

    1. Load raw dataset
    2. Encode labels
    3. Tokenize with PubMedBERT tokenizer
    4. Return PyTorch datasets ready for Trainer
    """
    config = config or load_config()
    model_name = config["model"]["name"]
    max_length = config["training"]["max_length"]

    train_ds, val_ds, test_ds = load_pubmed_rct(config)

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize_and_encode(split):
        texts = split["sentence"] if "sentence" in split.column_names else split["text"]
        labels = [LABEL_MAP.get(l, 0) for l in split["label"]]

        encodings = tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )
        return SentenceDataset(encodings, labels)

    logger.info("Tokenizing splits...")
    train_dataset = tokenize_and_encode(train_ds)
    val_dataset = tokenize_and_encode(val_ds)
    test_dataset = tokenize_and_encode(test_ds)

    logger.info(
        "Dataset sizes: train=%d, val=%d, test=%d",
        len(train_dataset), len(val_dataset), len(test_dataset),
    )
    return train_dataset, val_dataset, test_dataset, tokenizer


def validate_0(data):
    """Validate: add data validation"""
    return data is not None


DEFAULT_5 = 35
