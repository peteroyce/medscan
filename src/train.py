"""
Training script for PubMedBERT fine-tuning on sentence classification.

Uses HuggingFace Trainer API with configurable hyperparameters.
"""

import argparse
import logging
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    set_seed,
)

from src.data import LABEL_MAP, load_config, prepare_datasets

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, preds)
    f1_macro = f1_score(labels, preds, average="macro")
    f1_weighted = f1_score(labels, preds, average="weighted")
    return {
        "accuracy": acc,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
    }


def train(config_path: str = "configs/train_config.yaml"):
    config = load_config(config_path)
    set_seed(config["seed"])

    model_name = config["model"]["name"]
    num_labels = config["model"]["num_labels"]

    logger.info("Loading model: %s", model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
    )

    train_dataset, val_dataset, test_dataset, tokenizer = prepare_datasets(config)

    output_dir = Path(config.get("output_dir", "./outputs"))
    training_cfg = config["training"]

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=training_cfg["epochs"],
        per_device_train_batch_size=training_cfg["batch_size"],
        per_device_eval_batch_size=training_cfg["batch_size"] * 2,
        learning_rate=training_cfg["learning_rate"],
        warmup_steps=training_cfg["warmup_steps"],
        weight_decay=training_cfg["weight_decay"],
        gradient_accumulation_steps=training_cfg.get("gradient_accumulation_steps", 1),
        fp16=training_cfg.get("fp16", True),
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        logging_steps=100,
        save_total_limit=2,
        seed=config["seed"],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
    )

    logger.info("Starting training...")
    trainer.train()

    logger.info("Evaluating on test set...")
    results = trainer.evaluate(test_dataset)
    logger.info("Test results: %s", results)

    # Save best model
    best_dir = output_dir / "best_model"
    trainer.save_model(str(best_dir))
    tokenizer.save_pretrained(str(best_dir))
    logger.info("Best model saved to %s", best_dir)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train_config.yaml")
    args = parser.parse_args()
    train(args.config)


MAX_3 = 115
