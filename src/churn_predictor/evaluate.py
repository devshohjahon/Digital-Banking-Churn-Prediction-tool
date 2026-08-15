# evaluate.py
#
# Runs the final selected model on the held-out test set (only once), and
# saves the metrics, plots, and misclassified rows for error analysis.
# Also saves the final fitted pipeline so it can be reused for prediction.

import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay, classification_report, confusion_matrix,
    f1_score, precision_score, recall_score, roc_auc_score,
)

from churn_predictor.preprocessing import build_pipeline
from churn_predictor.train import candidate_models, get_splits

REPORTS_DIR = Path(__file__).resolve().parents[2] / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
ARTIFACTS_DIR = Path(__file__).resolve().parents[2] / "artifacts"


def evaluate_final_model(selected_name):
    X_train, X_val, X_test, y_train, y_val, y_test = get_splits()

    # Train the final model on train + val combined (test is still untouched)
    X_fit = pd.concat([X_train, X_val])
    y_fit = pd.concat([y_train, y_val])

    model = candidate_models()[selected_name]
    pipeline = build_pipeline(model)
    pipeline.fit(X_fit, y_fit)

    # Predict on the test set, only once
    test_predictions = pipeline.predict(X_test)
    test_probabilities = pipeline.predict_proba(X_test)[:, 1]

    metrics = {
        "selected_model": selected_name,
        "test_recall_churn": round(recall_score(y_test, test_predictions), 4),
        "test_precision_churn": round(precision_score(y_test, test_predictions), 4),
        "test_f1_churn": round(f1_score(y_test, test_predictions), 4),
        "test_roc_auc": round(roc_auc_score(y_test, test_probabilities), 4),
        "n_test_rows": len(y_test),
        "classification_report": classification_report(y_test, test_predictions, output_dict=True),
    }

    REPORTS_DIR.mkdir(exist_ok=True)
    with open(REPORTS_DIR / "final_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # Confusion matrix plot
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    cm = confusion_matrix(y_test, test_predictions)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Stayed", "Exited"])
    fig, ax = plt.subplots(figsize=(5, 4))
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"Confusion Matrix - {selected_name} (test set)")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "confusion_matrix.png", dpi=150)
    plt.close(fig)

    # Save the rows the model got wrong, for error analysis
    test_df = X_test.copy()
    test_df["y_true"] = y_test.values
    test_df["y_pred"] = test_predictions
    test_df["p_churn"] = test_probabilities

    misclassified = test_df[test_df["y_true"] != test_df["y_pred"]]
    misclassified.to_csv(REPORTS_DIR / "misclassified_rows.csv", index=False)

    false_negatives = misclassified[(misclassified["y_true"] == 1) & (misclassified["y_pred"] == 0)]
    false_positives = misclassified[(misclassified["y_true"] == 0) & (misclassified["y_pred"] == 1)]

    cols_to_check = ["CreditScore", "Age", "Balance", "NumOfProducts", "IsActiveMember"]

    error_notes = "# Error Analysis - " + selected_name + "\n\n"
    error_notes += f"## Held-out test set: {len(y_test)} customers\n\n"
    error_notes += f"- False negatives (missed at-risk customers): {len(false_negatives)}\n"
    error_notes += f"- False positives (flagged customers who stayed): {len(false_positives)}\n\n"
    error_notes += "## False-negative pattern (predicted stay, actually left)\n"
    error_notes += false_negatives[cols_to_check].mean().round(2).to_string() + "\n\n"
    error_notes += "## False-positive pattern (predicted leave, actually stayed)\n"
    error_notes += false_positives[cols_to_check].mean().round(2).to_string() + "\n\n"
    error_notes += (
        "## Threshold note\n"
        "The model outputs a churn probability, not just a 0/1 label. "
        "The default cutoff is 0.5. Since missing an at-risk customer is "
        "the costlier mistake, a lower cutoff (e.g. 0.35-0.4) trades some "
        "extra false positives for higher recall, and can be adjusted "
        "without retraining the model.\n"
    )

    with open(REPORTS_DIR / "error_analysis.md", "w") as f:
        f.write(error_notes)

    # Save the full fitted pipeline (preprocessing + model together)
    ARTIFACTS_DIR.mkdir(exist_ok=True)
    joblib.dump(pipeline, ARTIFACTS_DIR / "churn_pipeline.joblib")

    return metrics


if __name__ == "__main__":
    with open(REPORTS_DIR / "selected_candidate.json") as f:
        selected = json.load(f)["selected_candidate"]

    result = evaluate_final_model(selected)
    result_to_print = {k: v for k, v in result.items() if k != "classification_report"}
    print(json.dumps(result_to_print, indent=2))
