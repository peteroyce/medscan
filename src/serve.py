"""
FastAPI inference endpoint for sentence classification.
"""

import os
from pathlib import Path

import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.data import ID_TO_LABEL

ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000").split(",")

app = FastAPI(title="medscan", description="Medical abstract sentence classifier")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_DIR = Path("outputs/best_model")

model = None
tokenizer = None


class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=10000)


class PredictResponse(BaseModel):
    label: str
    confidence: float
    scores: dict[str, float]


@app.on_event("startup")
def load_model():
    global model, tokenizer
    if not MODEL_DIR.exists():
        raise RuntimeError(f"Model not found at {MODEL_DIR}. Run training first.")
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
    model = AutoModelForSequenceClassification.from_pretrained(str(MODEL_DIR))
    model.eval()


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if model is None:
        raise HTTPException(503, "Model not loaded")

    inputs = tokenizer(
        req.text,
        truncation=True,
        padding="max_length",
        max_length=256,
        return_tensors="pt",
    )

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1).squeeze()

    pred_idx = probs.argmax().item()
    scores = {ID_TO_LABEL[i]: round(probs[i].item(), 4) for i in range(len(probs))}

    return PredictResponse(
        label=ID_TO_LABEL[pred_idx],
        confidence=round(probs[pred_idx].item(), 4),
        scores=scores,
    )
