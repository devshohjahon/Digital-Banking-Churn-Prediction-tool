# data.py
# Loads the raw churn dataset and checks it has the columns we expect.

from pathlib import Path
import pandas as pd

# Path to the raw CSV file
RAW_DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "raw" / "Churn_Modelling.csv"

# These columns just identify a customer, they don't describe behavior.
# We will not use them as model inputs.
ID_COLUMNS = ["RowNumber", "CustomerId", "Surname"]

# This is what we are trying to predict
TARGET_COLUMN = "Exited"

# All the columns the dataset should have
EXPECTED_COLUMNS = [
    "RowNumber", "CustomerId", "Surname", "CreditScore", "Geography",
    "Gender", "Age", "Tenure", "Balance", "NumOfProducts", "HasCrCard",
    "IsActiveMember", "EstimatedSalary", "Exited",
]


def load_raw_data(path=RAW_DATA_PATH):
    # Read the CSV
    df = pd.read_csv(path)

    # Make sure none of the expected columns are missing
    for col in EXPECTED_COLUMNS:
        if col not in df.columns:
            raise ValueError(f"Missing expected column: {col}")

    return df


def quality_report(df):
    # Basic facts about the data, nothing invented
    report = {}
    report["n_rows"] = len(df)
    report["n_cols"] = df.shape[1]
    report["n_duplicate_rows"] = int(df.duplicated().sum())
    report["n_duplicate_customer_ids"] = int(df["CustomerId"].duplicated().sum())

    missing_counts = df.isna().sum()
    report["missing_by_column"] = {
        col: int(count) for col, count in missing_counts.items() if count > 0
    }

    report["target_balance"] = df[TARGET_COLUMN].value_counts(normalize=True).round(4).to_dict()

    return report


if __name__ == "__main__":
    data = load_raw_data()
    print(quality_report(data))
