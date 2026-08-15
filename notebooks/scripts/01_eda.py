#!/usr/bin/env python
# coding: utf-8
"""Standalone script version of notebooks/01_eda.ipynb.

Generated from the notebook so this same analysis can be run without Jupyter:
    cd bank-churn-prediction/notebooks/scripts
    python 01_eda.py
"""

import os
import matplotlib
matplotlib.use("Agg")  # headless backend so this runs from a plain terminal
import matplotlib.pyplot as plt

# Jupyter-only helpers (Image/display) get lightweight stand-ins so the same
# notebook code runs fine from a plain terminal: instead of rendering inline,
# they just print where the pre-generated figure lives on disk.
def Image(filename=None, **kwargs):
    return filename
def display(*args, **kwargs):
    for a in args:
        if a is not None:
            print(f"[Figure available at: {a}]")

_fig_counter = [0]
def _show_and_save(*args, **kwargs):
    out_dir = "../../reports/figures/script_output"
    os.makedirs(out_dir, exist_ok=True)
    _fig_counter[0] += 1
    out_path = f"{out_dir}/01_eda_fig{_fig_counter[0]}.png"
    plt.gcf().savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved figure -> {out_path}")
plt.show = _show_and_save

# # 01 - EDA & Data Quality Audit
# ## Digital Banking Customer Churn Prediction
# 
# Dataset: "Churn Modelling" bank customer churn dataset (10,000 rows), the same
# dataset referenced in the Kaggle notebook
# https://www.kaggle.com/code/kdsharma/banking-churn-analysis-modeling.
# A local copy is stored at `data/raw/Churn_Modelling.csv` so this notebook
# runs without Kaggle credentials.
# 
# Target: `Exited` (1 = customer churned, 0 = customer stayed).
# 
# Add the src/ folder to the path so we can import our own churn_predictor package
import sys
sys.path.insert(0, "../../src")

import pandas as pd
from churn_predictor.data import load_raw_data, quality_report

# Load the raw dataset and take a first look at the rows
df = load_raw_data()
df.head()


# ## Shape, types, and quality checks
# Check the shape (rows, columns) and the data type of each column
import json

print(df.shape)
print(df.dtypes)

# Run our reusable data-quality audit: duplicates, missing values, target balance
print(json.dumps(quality_report(df), indent=2))


# **Findings:** 10,000 rows, 14 columns, no missing values, no duplicate rows,
# no duplicate `CustomerId` values. The target is imbalanced (~80% stayed,
# ~20% exited) -- this is why the model comparison in notebook/experiments
# uses recall/F1/ROC-AUC rather than plain accuracy, and why a Dummy baseline
# is included for comparison.

# ## Target distribution
# Display the pre-generated target distribution chart (Exited vs stayed)

Image(filename="../../reports/figures/target_distribution.png")


# ## Churn rate by Geography and Gender
# Show churn rate broken down by Geography and by Gender, chart + exact numbers
display(Image(filename="../../reports/figures/churn_by_geography.png"))
display(Image(filename="../../reports/figures/churn_by_gender.png"))

# Print the precise churn rate per group so the chart above can be double-checked
print(df.groupby("Geography")["Exited"].mean().round(3))
print(df.groupby("Gender")["Exited"].mean().round(3))


# **Observation:** Germany shows a noticeably higher churn rate than France
# or Spain, and female customers churn somewhat more than male customers in
# this dataset. These are descriptive patterns in the data, not causal
# claims -- see `docs/RESPONSIBLE_AI.md` for fairness considerations before
# acting on demographic splits like Gender.

# ## Age distribution by churn status, and Number of Products
# Show how churn relates to Age and to the number of products a customer holds
display(Image(filename="../../reports/figures/age_by_churn.png"))
display(Image(filename="../../reports/figures/churn_by_numproducts.png"))


# **Observation:** Churned customers skew somewhat older. Customers with 3-4
# products churn far more than customers with 1-2 -- worth flagging for the
# business, though the small sample sizes in the 3-4 product bins mean this
# pattern should be treated as directional, not precise.

# ## Leakage analysis
# 
# | Column | Keep as model input? | Reason |
# |---|---|---|
# | RowNumber | No | Row index, not a customer attribute |
# | CustomerId | No | Unique identifier, no predictive meaning, would let the model memorize instead of generalize |
# | Surname | No | Unique identifier (also a fairness risk if used) |
# | CreditScore, Geography, Gender, Age, Tenure, Balance, NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary | Yes | Snapshot account attributes available at prediction time |
# | Exited | N/A | Target only, never a feature |
# 
# No column in this dataset encodes information from *after* the point the
# bank would actually want a churn prediction (there is no future-period
# balance, no "months until close", etc.), so beyond dropping identifiers
# there is no additional temporal-leakage risk to guard against here.

# ## Issue log
# See `reports/eda_issue_log.md` for the full text version of the findings above (generated by this same audit).
