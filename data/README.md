# Dataset: Bank Customer Churn ("Churn Modelling")

## Source

This project uses the **"Churn Modelling"** bank customer churn dataset,
the same dataset used in the Kaggle notebook
[kdsharma/banking-churn-analysis-modeling](https://www.kaggle.com/code/kdsharma/banking-churn-analysis-modeling).

A local copy is checked into `data/raw/Churn_Modelling.csv` so the project
runs and reproduces without a Kaggle account or API key. The file was
verified to have the exact schema and row count described below (10,000
rows, 14 columns) against multiple public mirrors of the same dataset
before being used here.

> If you have Kaggle API access and prefer to re-download the source
> yourself: `kaggle kernels output kdsharma/banking-churn-analysis-modeling
> -p data/raw` (or download `Churn_Modelling.csv` directly from the
> notebook's Input Data section) and overwrite `data/raw/Churn_Modelling.csv`.
> The loader in `src/churn_predictor/data.py` validates the schema on load,
> so a mismatched file will fail loudly instead of training silently on
> the wrong data.

## Schema

| Column | Type | Meaning |
|---|---|---|
| RowNumber | int | Row index. Not a customer attribute — dropped before modeling. |
| CustomerId | int | Unique customer identifier. Dropped before modeling. |
| Surname | str | Customer surname. Dropped before modeling (identifier + fairness risk). |
| CreditScore | int | Customer's credit score. |
| Geography | str | Country: France, Spain, or Germany. |
| Gender | str | Female or Male. |
| Age | int | Customer age in years. |
| Tenure | int | Years the customer has held an account with the bank. |
| Balance | float | Account balance. |
| NumOfProducts | int | Number of bank products the customer uses (1-4). |
| HasCrCard | int (0/1) | Whether the customer has a credit card. |
| IsActiveMember | int (0/1) | Whether the customer is an active member. |
| EstimatedSalary | float | Estimated annual salary. |
| **Exited** | int (0/1) | **Target.** 1 = customer churned (left the bank), 0 = stayed. |

## Data quality (from an actual run — see `reports/eda_issue_log.md`)

- 10,000 rows, 14 columns
- 0 duplicate rows, 0 duplicate `CustomerId` values
- 0 missing values in any column
- Target is imbalanced: ~79.6% stayed, ~20.4% exited

## Prediction point & leakage

There is no explicit timestamp column in this dataset — every row is a
single snapshot of a customer's account state, with `Exited` as the
outcome. Because of that:

- **RowNumber, CustomerId, Surname are excluded from model inputs.** They
  are identifiers, not attributes; leaving them in would let a model
  latch onto row order or specific names instead of learning generalizable
  churn patterns.
- No other column represents information from *after* the point the bank
  would want a prediction (there is no future balance, no "months to
  close", etc.), so beyond removing identifiers there is no additional
  temporal-leakage risk in this dataset.
- All scaling/encoding is fit only on the training split and reused
  unchanged on validation/test/inference (see
  `src/churn_predictor/preprocessing.py`).

## Limitations

This is a small, public, illustrative dataset (10,000 customers across two
"synthetic-style" geographies commonly used for churn-modeling tutorials).
It should not be treated as representative of any real bank's actual
customer base, and a model trained on it should not be deployed on a
different institution's customers without re-validation.
