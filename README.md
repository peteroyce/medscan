# medscan

Fine-tuned PubMedBERT for classifying sentences in medical abstracts into their rhetorical role (Background, Objective, Methods, Results, Conclusions).

Trained on the [PubMed 200k RCT](https://github.com/Franck-Dernoncourt/pubmed-rct) dataset.

## Results

| Model | Accuracy | Macro F1 | Weighted F1 |
|-------|----------|----------|-------------|
| TF-IDF + LogReg | 82.2% | 0.803 | 0.821 |
| TF-IDF + SVM | 82.8% | 0.811 | 0.827 |
| **PubMedBERT** | **89.1%** | **0.885** | **0.891** |

Per-class F1 (PubMedBERT):
- BACKGROUND: 0.865
- OBJECTIVE: 0.871
- METHODS: 0.912
- RESULTS: 0.912
- CONCLUSIONS: 0.853

## Setup

```bash
pip install -r requirements.txt
```

## Training

```bash
# edit configs/train_config.yaml if needed
make train
```

## Evaluation

```bash
make evaluate
```

Generates a classification report and confusion matrix in `outputs/`.

## Inference

```bash
make serve
# POST to localhost:8000/predict with {"text": "..."}
```

## Notebooks

- `01_eda.ipynb` - data exploration and class distribution
- `02_baseline.ipynb` - TF-IDF baselines (LogReg, SVM)
- `03_finetune_eval.ipynb` - PubMedBERT evaluation and comparison

## Tech

PyTorch, HuggingFace Transformers (PubMedBERT), scikit-learn, FastAPI
