"""Out-of-scope (OOS) detection and the metrics used in the CLINC150 paper (Larson et al., 2019).

A query is sent to "oos" (hand over to a human / say "I can't help with that") when the model's top
probability is below a threshold, or when the model itself predicts the "oos" class.
"""
import numpy as np

from src.data import OOS


def decide(probs: np.ndarray, classes: np.ndarray, threshold: float) -> np.ndarray:
    top = probs.argmax(1)
    pred = classes[top].astype(object)
    pred[probs.max(1) < threshold] = OOS
    return pred


def metrics(y_true, y_pred) -> dict:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    in_scope = y_true != OOS
    oos_pred = y_pred == OOS
    return {
        "in_scope_accuracy": round(float((y_pred[in_scope] == y_true[in_scope]).mean()), 4),
        "oos_recall": round(float(oos_pred[~in_scope].mean()), 4),
        "oos_precision": round(float((~in_scope[oos_pred]).mean()) if oos_pred.any() else 0.0, 4),
        "overall_accuracy": round(float((y_pred == y_true).mean()), 4),
        # in-scope queries the bot answered with the WRONG intent: the costly error for users
        "wrong_answer_rate": round(float(((y_pred != y_true) & ~oos_pred)[in_scope].mean()), 4),
    }


def tune_threshold(probs, classes, y_true, grid=np.linspace(0, 0.95, 96)) -> float:
    """Pick the threshold that maximises the average of in-scope accuracy and OOS recall on validation data.

    Averaging the two stops the tuner from ignoring OOS just because it is rare in the validation split.
    """
    def score(t):
        m = metrics(y_true, decide(probs, classes, t))
        return (m["in_scope_accuracy"] + m["oos_recall"]) / 2
    return float(max(grid, key=score))


def threshold_curve(probs, classes, y_true, grid=np.linspace(0, 0.95, 39)) -> list:
    return [{"threshold": round(float(t), 3), **metrics(y_true, decide(probs, classes, t))} for t in grid]
