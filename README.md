# medscan

Sentence-level rhetorical role classifier for medical abstracts. It fine-tunes PubMedBERT to
label every sentence of a randomised-trial abstract as BACKGROUND, OBJECTIVE, METHODS, RESULTS
or CONCLUSIONS, which is the structure a reader needs when skimming hundreds of papers but that
most published abstracts do not carry as machine-readable markup.

[![CI](https://github.com/peteroyce/medscan/actions/workflows/ci.yml/badge.svg)](https://github.com/peteroyce/medscan/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue.svg)](https://python.org)

## Features

- Fine-tunes `microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract` for 5-way sentence
  classification via the HuggingFace `Trainer` API.
- Domain-specific pretraining is the point: PubMedBERT is pretrained on biomedical text, so the
  tokeniser does not shred clinical vocabulary the way a general-purpose BERT vocabulary does.
- Single YAML config (`configs/train_config.yaml`) drives model, data and training hyperparameters
  — no constants buried in the training script.
- Model selection on macro F1 rather than accuracy, because the label distribution is skewed
  (RESULTS is 32% of training sentences, OBJECTIVE 12%).
- Evaluation writes a per-class classification report to `outputs/eval_results.json` and a
  confusion matrix PNG, so results are artefacts rather than console output.
- FastAPI inference service returning the predicted label, its confidence, and the full softmax
  distribution over all five classes.
- Dataset loader falls back from PubMed 200k RCT to the 20k variant when the larger split is
  unavailable, so a clone can train on a laptop.
- Classical TF-IDF baselines (Logistic Regression, linear SVM) kept in the notebooks as the bar
  the transformer has to clear.

## Architecture

```
PubMed RCT (HuggingFace datasets, 200k → fallback 20k)
        │
        ▼
src/data.py         load, map labels to ids, tokenise (max_length 256),
                    wrap in a torch Dataset
        │
        ▼
src/train.py        PubMedBERT + Trainer; eval each epoch,
                    keep best checkpoint by f1_macro → outputs/best_model/
        │
        ├──────────────► src/evaluate.py   test-set report, per-class metrics,
        │                                  confusion_matrix.png, eval_results.json
        ▼
src/serve.py        FastAPI: loads outputs/best_model at startup, POST /predict
```

| Path | Role |
|---|---|
| `src/data.py` | Dataset loading, `LABEL_MAP`/`ID_TO_LABEL`, tokenisation |
| `src/train.py` | Fine-tuning loop and metric computation |
| `src/evaluate.py` | Test-set evaluation and artefact generation |
| `src/serve.py` | FastAPI inference endpoint |
| `configs/train_config.yaml` | All hyperparameters |
| `notebooks/` | EDA, TF-IDF baselines, fine-tuned model evaluation |

## Quickstart

```bash
git clone https://github.com/peteroyce/medscan.git
cd medscan
pip install -r requirements.txt        # or: make setup

make train                             # fine-tune, writes outputs/best_model/
make evaluate                          # test-set report + confusion matrix
make serve                             # uvicorn on :8000
```

Training defaults: learning rate 2e-5, 5 epochs, batch size 16, 500 warmup steps, weight decay
0.01, max sequence length 256, fp16, seed 42. Edit `configs/train_config.yaml` and re-run;
both `make train` and `make evaluate` accept `--config` when invoked directly.

The serving process reads one environment variable:

| Variable | Default | Purpose |
|---|---|---|
| `ALLOWED_ORIGINS` | `http://localhost:3000` | Comma-separated CORS origins |

`make serve` fails fast if `outputs/best_model/` does not exist — train first.

## API

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Liveness probe; reports whether the model is loaded |
| `POST` | `/predict` | Classify one sentence (1–10,000 characters) |

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "A randomized , double-blind , placebo-controlled trial was conducted at 12 centers ."}'
```

```json
{
  "label": "METHODS",
  "confidence": 0.9731,
  "scores": {
    "BACKGROUND": 0.0071,
    "OBJECTIVE": 0.0044,
    "METHODS": 0.9731,
    "RESULTS": 0.0098,
    "CONCLUSIONS": 0.0056
  }
}
```

## Results

Test split of PubMed 200k RCT (30,135 sentences). Figures below are the ones recorded in
`notebooks/02_baseline.ipynb` and `notebooks/03_finetune_eval.ipynb`.

| Model | Accuracy | Macro F1 | Weighted F1 |
|---|---|---|---|
| TF-IDF + Logistic Regression | 0.8217 | 0.8032 | 0.8208 |
| TF-IDF + linear SVM | 0.8284 | 0.8106 | 0.8274 |
| **PubMedBERT (fine-tuned)** | **0.8913** | **0.8847** | **0.8911** |

Per-class F1 for the fine-tuned model: METHODS 0.9122, RESULTS 0.9115, OBJECTIVE 0.8709,
BACKGROUND 0.8652, CONCLUSIONS 0.8531. The largest gains over the SVM baseline are on OBJECTIVE
and CONCLUSIONS — the two classes whose vocabulary overlaps heavily with their neighbours, where
bag-of-words features have the least to work with.

## Tech stack

PyTorch, HuggingFace Transformers and Datasets, scikit-learn, FastAPI, matplotlib/seaborn, PyYAML.

## Continuous integration

`.github/workflows/ci.yml` runs `ruff check` and `ruff format --check` over `src/`, validates that
`configs/train_config.yaml` parses, and imports the data and evaluation modules on Python 3.11 and
3.12. There is no unit test suite in this repository.

## License

MIT
