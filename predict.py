"""Classify chatbot messages with the fine-tuned DistilBERT model.

Usage:  python predict.py "how much money do I have in checking" "who won the world cup"
Needs models/distilbert-clinc/ from `python -m src.evaluate`.
"""
import json
import sys
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_DIR = Path(__file__).parent / "models" / "distilbert-clinc"


def main(texts):
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR).eval()
    threshold = json.loads((MODEL_DIR / "threshold.json").read_text())["threshold"]
    classes = json.loads((MODEL_DIR / "classes.json").read_text())
    with torch.no_grad():
        probs = model(**tok(texts, padding=True, return_tensors="pt")).logits.softmax(-1)
    for text, p in zip(texts, probs):
        conf, idx = p.max(0)
        intent = classes[int(idx)]
        if intent == "oos" or conf < threshold:
            intent = "out_of_scope -> hand over to a human agent"
        print(f"{conf:5.0%}  {intent:45s}  {text}")


if __name__ == "__main__":
    main(sys.argv[1:] or ["what's my checking balance", "book a table for two at 8pm", "who won the cricket match"])
