# Project Brief — Digital Banking Customer Churn Prediction

## Problem

A retail bank wants to know, for each current customer, how likely they
are to leave the bank ("churn"), so a retention team can proactively reach
out to the customers most likely to leave.

## Stakeholder

A digital banking retention/customer-success team who would use the
model's output as a prioritized worklist, not an automated decision.

## Target

Binary classification: `Exited` (1 = churned, 0 = retained).

## Prediction point

Each row is a snapshot of a customer's account attributes (credit score,
geography, demographics, tenure, balance, product usage, engagement
flags, salary). The model predicts churn from that snapshot — it does not
use any information about what happens after the snapshot (there is no
such "future" information in this dataset beyond the `Exited` label
itself).

## Inputs / Outputs

**Inputs:** CreditScore, Geography, Gender, Age, Tenure, Balance,
NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary.

**Excluded from inputs:** RowNumber, CustomerId, Surname (identifiers with
no generalizable signal — see `data/README.md`).

**Output:** a churn probability (0-1), a `high_risk` / `low_risk` label at
the 0.5 threshold, and a plain-language message. See
`src/churn_predictor/predict.py`.

## Metrics

Recall on the churn class is the primary metric: missing an at-risk
customer (a false negative) means no retention outreach happens, which is
the costlier mistake for the business. Precision and F1 are tracked
alongside recall so the flagged list stays usable (see
`reports/experiments_log.md` for why recall alone was not sufficient to
pick the final model). ROC-AUC is tracked as an overall ranking-quality
metric.

## Leakage rules

1. `RowNumber`, `CustomerId`, `Surname` are never model inputs.
2. `Exited` is only ever the label, never a feature.
3. Preprocessing (scaling, encoding) is fit only on the training split.
4. No engineered feature uses information unavailable at the prediction
   snapshot.

## Fairness considerations

`Geography` and `Gender` are used as model inputs because they show a
real association with churn in this data (see `notebooks/01_eda.ipynb`).
Using demographic fields in a churn model is common, but any operational
use of geography/gender-correlated churn scores should be reviewed for
disparate-impact concerns before being tied to differential treatment of
customers — see `docs/RESPONSIBLE_AI.md`.

## Scope and non-goals

**In scope:** a leakage-safe, tested, reproducible churn classifier and a
notebook demo that loads the saved pipeline and scores new customer
records.

**Out of scope:** a production API/deployment, a live retraining pipeline,
causal analysis of *why* a customer churns, and any automated action
(offers, account changes) triggered directly by the model's output.
