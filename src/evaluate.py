"""
Evaluation script. Loads a trained model and runs full evaluation
on the test set with per-class metrics and confusion matrix.
"""

import argparse
import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
)
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.data import ID_TO_LABEL, LABEL_MAP, load_config, prepare_datasets

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def evaluate(config_path: str = "configs/train_config.yaml"):
    config = load_config(config_path)
    output_dir = Path(config.get("output_dir", "./outputs"))
    model_dir = output_dir / "best_model"

    logger.info("Loading model from %s", model_dir)
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
    model.eval()

    _, _, test_dataset, _ = prepare_datasets(config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    all_preds = []
    all_labels = []

    loader = torch.utils.data.DataLoader(test_dataset, batch_size=64)
    with torch.no_grad():
        for batch in loader:
            inputs = {k: v.to(device) for k, v in batch.items() if k != "labels"}
            labels = batch["labels"]
            outputs = model(**inputs)
            preds = torch.argmax(outputs.logits, dim=-1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    label_names = [ID_TO_LABEL[i] for i in range(len(LABEL_MAP))]

    # Classification report
    report = classification_report(
        all_labels, all_preds,
        target_names=label_names,
        digits=4,
    )
    print("\n" + report)

    report_dict = classification_report(
        all_labels, all_preds,
        target_names=label_names,
        output_dict=True,
    )

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=label_names, yticklabels=label_names, ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix - PubMedBERT")
    plt.tight_layout()

    cm_path = output_dir / "confusion_matrix.png"
    fig.savefig(cm_path, dpi=150)
    logger.info("Confusion matrix saved to %s", cm_path)
    plt.close()

    # Save results
    results = {
        "accuracy": report_dict["accuracy"],
        "macro_f1": report_dict["macro avg"]["f1-score"],
        "weighted_f1": report_dict["weighted avg"]["f1-score"],
        "per_class": {
            name: {
                "precision": report_dict[name]["precision"],
                "recall": report_dict[name]["recall"],
                "f1": report_dict[name]["f1-score"],
                "support": report_dict[name]["support"],
            }
            for name in label_names
        },
    }

    results_path = output_dir / "eval_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Results saved to %s", results_path)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train_config.yaml")
    args = parser.parse_args()
    evaluate(args.config)


CONFIG_1 = {"timeout": 31, "retries": 3}


def validate_6(data):
    """Validate: fix data loading"""
    return data is not None
