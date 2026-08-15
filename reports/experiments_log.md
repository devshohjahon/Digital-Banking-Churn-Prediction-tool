# Experiment Log

All models below were trained and validated on the **same fixed 60/20/20
train/validation/test split** (stratified on `Exited`, `random_state=42`),
using the leakage-safe preprocessing pipeline in
`src/churn_predictor/preprocessing.py`. The test split was not touched
during this stage.

## Results (validation set, 2,000 customers)

| model | recall (churn) | precision (churn) | F1 (churn) | ROC-AUC | train time (s) |
|---|---|---|---|---|---|
| dummy_baseline | 0.0000 | 0.0000 | 0.0000 | 0.5000 | 0.013 |
| logistic_regression_balanced | 0.7451 | 0.3990 | 0.5197 | 0.7938 | 0.020 |
| random_forest_balanced | 0.7157 | 0.5692 | 0.6341 | 0.8732 | 1.546 |

Raw values: see `reports/experiments.csv`.

## Selection reasoning

The dummy baseline (always predicts "stayed") confirms the ~80/20 class
imbalance -- it scores 0 recall on the churn class by construction and is
the floor every real model must beat.

Logistic Regression (balanced) reaches slightly higher recall (0.745 vs
0.716), but at a real cost: precision of only 0.40 means **6 out of 10
customers it flags as at-risk actually would have stayed** -- a retention
team acting on that list wastes most of its outreach effort.

Random Forest (balanced) trades a small amount of recall for a much better
precision/recall balance (F1 0.634 vs 0.520) and a clearly stronger ROC-AUC
(0.873 vs 0.794), meaning it ranks at-risk customers more reliably overall.

**Selected candidate: `random_forest_balanced`**, using the rule in
`src/churn_predictor/train.py::select_candidate`: among models clearing a
minimum recall bar (0.65), pick the best F1. This was decided from
validation evidence only, before the test set was touched.

## Final held-out test result (2,000 customers, untouched until this point)

| metric | value |
|---|---|
| recall (churn) | 0.6806 |
| precision (churn) | 0.5496 |
| F1 (churn) | 0.6081 |
| ROC-AUC | 0.8597 |

Test metrics are close to validation metrics for the same model
(recall 0.68 vs 0.72, F1 0.61 vs 0.63), which is evidence the model is not
overfit to the validation split. Full numbers: `reports/final_metrics.json`.
See `reports/error_analysis.md` for false-negative/false-positive patterns.
