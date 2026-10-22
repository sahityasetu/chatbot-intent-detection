import numpy as np
import pytest

from src.oos import decide, metrics, tune_threshold

CLASSES = np.array(["balance", "oos", "transfer"])


def test_decide_rejects_low_confidence():
    probs = np.array([[0.9, 0.05, 0.05], [0.4, 0.3, 0.3], [0.1, 0.8, 0.1]])
    assert list(decide(probs, CLASSES, 0.5)) == ["balance", "oos", "oos"]
    assert list(decide(probs, CLASSES, 0.0)) == ["balance", "balance", "oos"]


def test_metrics():
    y_true = ["balance", "transfer", "oos", "oos"]
    y_pred = ["balance", "oos", "oos", "transfer"]
    m = metrics(y_true, y_pred)
    assert m["in_scope_accuracy"] == 0.5
    assert m["oos_recall"] == 0.5
    assert m["oos_precision"] == 0.5
    assert m["overall_accuracy"] == 0.5
    assert m["wrong_answer_rate"] == 0.0  # the in-scope miss was sent to a human, not answered wrongly


def test_tune_threshold_separates_confident_from_unsure():
    probs = np.array([[0.95, 0.0, 0.05]] * 10 + [[0.4, 0.0, 0.6]] * 10)
    y = ["balance"] * 10 + ["oos"] * 10
    t = tune_threshold(probs, CLASSES, y)
    assert 0.6 < t <= 0.95
    assert metrics(y, decide(probs, CLASSES, t))["overall_accuracy"] == pytest.approx(1.0)
