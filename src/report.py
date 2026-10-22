"""Compare all runs, plot the threshold trade-off and analyse the best model's errors.

Run from the project root after src.evaluate:  python -m src.report
Outputs: reports/results.csv, reports/error_analysis.json, reports/figures/*.png
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.data import OOS, domain_map

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "reports" / "figures"
BLUE, ORANGE, GREY = "#2a78d6", "#eb6834", "#898781"
STRATEGY = {"threshold": "threshold only", "oos_class": "OOS class + threshold"}
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "font.size": 10})


def load_runs() -> list:
    return [json.loads(p.read_text()) for p in sorted((ROOT / "reports" / "runs").glob("*.json"))]


def results_table(runs) -> pd.DataFrame:
    rows = [{"model": r["model"], "strategy": STRATEGY[r["strategy"]], "run": f"{r['key']}_{r['strategy']}",
             "threshold": r["threshold"], **r["test"],
             "in_scope_acc_no_threshold": r["test_no_threshold"]["in_scope_accuracy"],
             "oos_recall_no_threshold": r["test_no_threshold"]["oos_recall"],
             "train_seconds": r["train_seconds"], "ms_per_query": r["ms_per_query"]} for r in runs]
    return pd.DataFrame(rows).sort_values("overall_accuracy")


def plot_comparison(t: pd.DataFrame):
    labels = [f"{m}\n({s})" for m, s in zip(t.model, t.strategy)]
    y = range(len(t))
    fig, ax = plt.subplots(figsize=(8, 0.6 * len(t) + 1.5))
    ax.barh([i + 0.2 for i in y], t.in_scope_accuracy * 100, height=0.38, color=BLUE, label="In-scope accuracy")
    ax.barh([i - 0.2 for i in y], t.oos_recall * 100, height=0.38, color=ORANGE, label="Out-of-scope recall")
    for i, (a, b) in enumerate(zip(t.in_scope_accuracy, t.oos_recall)):
        ax.text(a * 100 + 0.5, i + 0.2, f"{a:.1%}", va="center", fontsize=8)
        ax.text(b * 100 + 0.5, i - 0.2, f"{b:.1%}", va="center", fontsize=8)
    ax.set_yticks(list(y), labels, fontsize=8)
    ax.set_xlim(50, 102)
    ax.set_xlabel("% of test queries (4,500 in-scope, 1,000 out-of-scope)")
    ax.set_title("Model comparison on the CLINC150 test set", loc="left", fontweight="bold")
    ax.legend(frameon=False, loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "model_comparison.png", dpi=160)


def plot_threshold(run: dict):
    c = pd.DataFrame(run["test_curve"])
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(c.threshold, c.in_scope_accuracy * 100, color=BLUE, lw=2, label="In-scope accuracy")
    ax.plot(c.threshold, c.oos_recall * 100, color=ORANGE, lw=2, label="Out-of-scope recall")
    ax.plot(c.threshold, c.wrong_answer_rate * 100, color=GREY, lw=2, ls="--", label="Wrong answers (in-scope)")
    ax.axvline(run["threshold"], color="#0b0b0b", lw=1)
    ax.annotate(f"chosen on validation: {run['threshold']:.2f}", (run["threshold"], 50), xytext=(6, 0),
                textcoords="offset points", fontsize=8)
    ax.set_xlabel("Confidence threshold (below it, the bot says \"I can't help with that\")")
    ax.set_ylabel("%")
    ax.set_title(f"The threshold trade-off: {run['model']}", loc="left", fontweight="bold")
    ax.legend(frameon=False, fontsize=8, loc="center left")
    fig.tight_layout()
    fig.savefig(FIG / "threshold_tradeoff.png", dpi=160)


def error_analysis(run_name: str) -> dict:
    p = pd.read_csv(ROOT / "reports" / "predictions" / f"{run_name}.csv")
    dom = domain_map()
    wrong = p[p.label != p.pred]
    in_scope_wrong = wrong[(wrong.label != OOS) & (wrong.pred != OOS)]
    same_domain = (in_scope_wrong.label.map(dom) == in_scope_wrong.pred.map(dom)).mean()
    pairs = in_scope_wrong.groupby(["label", "pred"]).size().sort_values(ascending=False).head(8)
    examples = wrong.sample(min(12, len(wrong)), random_state=0)[["text", "label", "pred", "confidence"]]
    return {
        "run": run_name,
        "errors": len(wrong),
        "in_scope_sent_to_oos": int(((p.label != OOS) & (p.pred == OOS)).sum()),
        "oos_answered_as_intent": int(((p.label == OOS) & (p.pred != OOS)).sum()),
        "intent_confusions": len(in_scope_wrong),
        "intent_confusions_within_same_domain": round(float(same_domain), 4),
        "top_confused_pairs": [{"actual": a, "predicted": b, "count": int(n)} for (a, b), n in pairs.items()],
        "example_errors": examples.to_dict("records"),
    }


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    runs = load_runs()
    t = results_table(runs)
    t.to_csv(ROOT / "reports" / "results.csv", index=False)
    plot_comparison(t)
    best = max(runs, key=lambda r: r["test"]["overall_accuracy"])
    plot_threshold(best)
    ea = error_analysis(f"{best['key']}_{best['strategy']}")
    (ROOT / "reports" / "error_analysis.json").write_text(json.dumps(ea, indent=2))
    print(t.drop(columns=["run"]).to_string(index=False))
    print(json.dumps({k: v for k, v in ea.items() if k != "example_errors"}, indent=2))


if __name__ == "__main__":
    main()
