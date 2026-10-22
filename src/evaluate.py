"""Train and evaluate every model x out-of-scope strategy, saving one JSON per run.

Strategies:
  threshold   - train on the 150 in-scope intents only; reject low-confidence queries as out-of-scope
  oos_class   - also train on the 250 labelled out-of-scope examples as a 151st class, plus the threshold

Run from the project root:
    python -m src.evaluate                        # all models (DistilBERT ~25 min on an M2 laptop)
    python -m src.evaluate --models tfidf embed   # fast models only
Outputs: reports/runs/<model>_<strategy>.json, reports/predictions/<model>_<strategy>.csv
"""
import argparse
import json
import time
from pathlib import Path

import pandas as pd

from src.data import OOS, load_splits
from src.models import DistilBert, EmbeddingLR, TfidfLR
from src.oos import decide, metrics, threshold_curve, tune_threshold

ROOT = Path(__file__).resolve().parents[1]
MODELS = {"tfidf": TfidfLR, "embed": EmbeddingLR, "bert": DistilBert}


def run(key: str, strategy: str, splits: dict) -> dict:
    train = splits["train"] if strategy == "oos_class" else splits["train"][~splits["train"].is_oos]
    model = MODELS[key]()
    start = time.time()
    model.fit(list(train.text), list(train.label))
    train_sec = time.time() - start

    val, test = splits["validation"], splits["test"]
    p_val, classes = model.predict_proba(val.text)
    threshold = tune_threshold(p_val, classes, val.label)

    start = time.time()
    p_test, _ = model.predict_proba(test.text)
    ms_per_query = (time.time() - start) / len(test) * 1000
    pred = decide(p_test, classes, threshold)

    result = {
        "model": model.name, "key": key, "strategy": strategy, "threshold": round(threshold, 3),
        "train_seconds": round(train_sec, 1), "ms_per_query": round(ms_per_query, 2),
        "test": metrics(test.label, pred),
        "test_no_threshold": metrics(test.label, decide(p_test, classes, 0.0)),
        "test_curve": threshold_curve(p_test, classes, test.label),
    }
    name = f"{key}_{strategy}"
    for sub in ["runs", "predictions"]:
        (ROOT / "reports" / sub).mkdir(parents=True, exist_ok=True)
    (ROOT / "reports" / "runs" / f"{name}.json").write_text(json.dumps(result, indent=2))
    pd.DataFrame({"text": test.text, "label": test.label, "pred": pred, "confidence": p_test.max(1).round(4)}) \
        .to_csv(ROOT / "reports" / "predictions" / f"{name}.csv", index=False)
    if key == "bert" and strategy == "oos_class":
        model.save(ROOT / "models" / "distilbert-clinc")
        (ROOT / "models" / "distilbert-clinc" / "threshold.json").write_text(json.dumps({"threshold": threshold}))
    print(name, json.dumps(result["test"]), f"threshold={threshold:.2f}")
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=list(MODELS), choices=list(MODELS))
    args = ap.parse_args()
    splits = load_splits()
    assert splits["test"].label.eq(OOS).sum() == 1000
    for key in args.models:
        for strategy in ["threshold", "oos_class"]:
            run(key, strategy, splits)


if __name__ == "__main__":
    main()
