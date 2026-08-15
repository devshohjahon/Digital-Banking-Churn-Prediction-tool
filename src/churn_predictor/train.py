# train.py
#
# Splits the data once (train/val/test), trains a dummy baseline plus two
# real models on the training set, and compares them on the validation
# set. The test set is not touched here at all -- that happens later in
# evaluate.py, after we've already picked a winner.

import json
import time
from pathlib import Path

import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split

from churn_predictor.data import load_raw_data
from churn_predictor.preprocessing import build_pipeline, split_features_target

REPORTS_DIR = Path(__file__).resolve().parents[2] / "reports"
RANDOM_STATE = 42

# Minimum recall a model needs to even be considered for selection
MIN_ACCEPTABLE_RECALL = 0.65


def get_splits(random_state=RANDOM_STATE):
    df = load_raw_data()
    X, y = split_features_target(df)

    # First split off 60% train, 40% temp (stratified so churn rate stays
    # the same in every split)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.4, stratify=y, random_state=random_state
    )
    # Then split temp in half -> 20% validation, 20% test
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=random_state
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


def candidate_models():
    # The three models we compare
    models = {
        "dummy_baseline": DummyClassifier(strategy="most_frequent"),
        "logistic_regression_balanced": LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE
        ),
        "random_forest_balanced": RandomForestClassifier(
            n_estimators=300, max_depth=8, class_weight="balanced",
            random_state=RANDOM_STATE, n_jobs=-1
        ),
    }
    return models


def run_experiments():
    X_train, X_val, X_test, y_train, y_val, y_test = get_splits()

    results = []

    for name, model in candidate_models().items():
        pipeline = build_pipeline(model)

        start_time = time.perf_counter()
        pipeline.fit(X_train, y_train)
        train_seconds = time.perf_counter() - start_time

        val_predictions = pipeline.predict(X_val)

        # Not every model has predict_proba, but all three of ours do
        val_probabilities = pipeline.predict_proba(X_val)[:, 1]

        row = {
            "model": name,
            "val_recall_churn": round(recall_score(y_val, val_predictions), 4),
            "val_precision_churn": round(precision_score(y_val, val_predictions), 4),
            "val_f1_churn": round(f1_score(y_val, val_predictions), 4),
            "val_roc_auc": round(roc_auc_score(y_val, val_probabilities), 4),
            "train_seconds": round(train_seconds, 3),
        }
        results.append(row)

    results_df = pd.DataFrame(results).sort_values("val_recall_churn", ascending=False)

    REPORTS_DIR.mkdir(exist_ok=True)
    results_df.to_csv(REPORTS_DIR / "experiments.csv", index=False)

    return results_df


def select_candidate(results_df):
    # We care most about recall (catching at-risk customers), but a model
    # that flags everyone isn't useful either. So: only look at models
    # that reach a minimum recall, then pick the best F1 among those.
    trained_only = results_df[results_df["model"] != "dummy_baseline"]
    good_enough = trained_only[trained_only["val_recall_churn"] >= MIN_ACCEPTABLE_RECALL]

    if len(good_enough) == 0:
        good_enough = trained_only

    best_row = good_enough.sort_values("val_f1_churn", ascending=False).iloc[0]
    return best_row["model"]


if __name__ == "__main__":
    results = run_experiments()
    print(results.to_string(index=False))

    winner = select_candidate(results)
    print(f"\nSelected candidate: {winner}")

    with open(REPORTS_DIR / "selected_candidate.json", "w") as f:
        json.dump({"selected_candidate": winner}, f, indent=2)
