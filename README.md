# Digital Banking Customer Churn Prediction

A leakage-safe, tested machine learning project that predicts which bank
customers are likely to churn (`Exited`), so a retention team can
prioritize outreach. Built as an early-warning **decision-support signal**
— not an automated decision system.

## Problem & dataset

- **Target:** `Exited` (1 = customer left the bank, 0 = stayed)
- **Dataset:** the "Churn Modelling" bank customer dataset (10,000 rows),
  the same dataset used in the Kaggle notebook
  [kdsharma/banking-churn-analysis-modeling](https://www.kaggle.com/code/kdsharma/banking-churn-analysis-modeling).
  A verified local copy is included at `data/raw/Churn_Modelling.csv` — see
  `data/README.md` for schema, quality audit, and how to re-download it
  yourself if you have Kaggle access.
- **Columns:** `RowNumber, CustomerId, Surname, CreditScore, Geography,
  Gender, Age, Tenure, Balance, NumOfProducts, HasCrCard, IsActiveMember,
  EstimatedSalary, Exited`

## Why this is leakage-safe

- `RowNumber`, `CustomerId`, `Surname` are **excluded from model inputs** —
  they are identifiers, not customer attributes, and would let the model
  memorize rows instead of learning generalizable patterns.
- `Exited` is only ever used as the label, never as a feature.
- Preprocessing (scaling, one-hot encoding) is fit **only on the training
  split**, inside a single `sklearn.Pipeline`, and reused unchanged on
  validation, test, and inference data — so there is no way for the saved
  model to process a new customer differently than it was trained.
- The test set is touched exactly once, after a final model is already
  selected from validation results.
- All of the above is enforced by automated tests, not just documentation
  — see `tests/test_preprocessing.py::test_no_leakage_columns_reach_the_model`.

Full reasoning: `docs/PROJECT_BRIEF.md` and `data/README.md`.

## Project structure

```
bank-churn-prediction/
├── data/
│   ├── README.md              # dataset source, schema, quality audit
│   └── raw/Churn_Modelling.csv
├── src/churn_predictor/
│   ├── data.py                # reproducible loading + schema validation
│   ├── preprocessing.py       # leakage-safe Pipeline/ColumnTransformer
│   ├── train.py                # baseline + candidate model comparison
│   ├── evaluate.py             # final held-out test evaluation
│   └── predict.py              # predict_churn_risk() with input validation
├── notebooks/
│   ├── 01_eda.ipynb            # data audit, plots, leakage analysis
│   ├── 02_demo.ipynb           # step-by-step pipeline walkthrough + visual risk gauge
│   └── 03_churn_insights.ipynb # who is leaving: churn rate by segment + feature importance
├── reports/                    # experiments.csv, final_metrics.json,
│                                # error_analysis.md, figures/
├── artifacts/churn_pipeline.joblib   # saved fitted pipeline (preprocess + model)
├── tests/                      # pytest suite (9 tests)
└── docs/                       # PROJECT_BRIEF.md, RESPONSIBLE_AI.md
```

## Results

Three models were compared on one fixed, stratified 60/20/20 train/val/test
split (see `reports/experiments_log.md` for the full reasoning):

| model | val recall (churn) | val precision | val F1 | val ROC-AUC |
|---|---|---|---|---|
| Dummy baseline | 0.000 | 0.000 | 0.000 | 0.500 |
| Logistic Regression (balanced) | 0.745 | 0.399 | 0.520 | 0.794 |
| **Random Forest (balanced) — selected** | 0.716 | 0.569 | 0.634 | 0.873 |

**Final held-out test results** (2,000 customers, evaluated once):

| metric | value |
|---|---|
| Recall (churn) | 0.681 |
| Precision (churn) | 0.550 |
| F1 (churn) | 0.608 |
| ROC-AUC | 0.860 |

Test performance closely tracks validation performance, indicating the
model generalizes rather than having been overfit to one split.

## Quickstart

```bash
# 1. install
pip install -r requirements.txt

# 2. run the model comparison (fits Dummy / LogisticRegression / RandomForest
#    on train, evaluates on validation, writes reports/experiments.csv)
PYTHONPATH=src python -m churn_predictor.train

# 3. train the final selected model on train+val and evaluate ONCE on the
#    held-out test set; saves artifacts/churn_pipeline.joblib
PYTHONPATH=src python -m churn_predictor.evaluate

# 4. run tests
PYTHONPATH=src pytest tests/ -v

# 5. try a prediction
PYTHONPATH=src python -m churn_predictor.predict
```

Or open the notebooks directly — all three already contain executed outputs:
- `notebooks/01_eda.ipynb` — data quality audit and leakage analysis
- `notebooks/02_demo.ipynb` — step-by-step walkthrough of the pipeline (validate → preprocess → predict) plus a visual risk gauge
- `notebooks/03_churn_insights.ipynb` — churn rate broken down by age, tenure, balance, credit score, activity, and geography, plus which features the trained model relies on most

Prefer plain Python over Jupyter? `notebooks/scripts/` has a standalone `.py` version of each notebook (auto-converted, headless-safe, no Jupyter required):
```bash
cd notebooks/scripts
python 01_eda.py
python 02_demo.py
python 03_churn_insights.py
```

## Using the model on a new customer

```python
from churn_predictor.predict import predict_churn_risk

predict_churn_risk({
    "CreditScore": 619,
    "Geography": "France",
    "Gender": "Female",
    "Age": 42,
    "Tenure": 2,
    "Balance": 0.0,
    "NumOfProducts": 1,
    "HasCrCard": 1,
    "IsActiveMember": 1,
    "EstimatedSalary": 101348.88,
})
# -> {'churn_probability': 0.62, 'risk_label': 'high_risk',
#     'message': 'High churn risk. Recommend proactive retention outreach.'}
```

`predict_churn_risk` rejects records that are missing a required field,
include an unexpected field (including `Exited`, `RowNumber`, `CustomerId`,
or `Surname`), or contain an out-of-range value (e.g. `Age=250`) — see
`tests/test_inference.py`.

## Responsible use

This model is an **early-warning signal**, not an automatic decision. High
-risk flags should go to a human reviewer before any action is taken with
a customer. See `docs/RESPONSIBLE_AI.md` for fairness notes on `Geography`
and `Gender`, and dataset limitations.

## License

Code in this repository: MIT (see individual file headers or add a
`LICENSE` file as needed for your submission). The dataset is used under
its original public-tutorial license — see `data/README.md` for the
source link.
