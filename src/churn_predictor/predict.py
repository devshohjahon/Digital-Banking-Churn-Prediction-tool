# predict.py
#
# Loads the saved pipeline and predicts churn risk for one customer.
# The pipeline already has the preprocessing built in, so a plain,
# raw customer record goes in and a risk score comes out.

from pathlib import Path

import joblib
import pandas as pd

from churn_predictor.preprocessing import FEATURE_COLUMNS

ARTIFACT_PATH = Path(__file__).resolve().parents[2] / "artifacts" / "churn_pipeline.joblib"

# Fields we don't want the caller to pass in, with why
REJECTED_FIELDS = {
    "Exited": "Exited is the thing we're predicting, not an input.",
}

VALID_GEOGRAPHY = {"France", "Spain", "Germany"}
VALID_GENDER = {"Female", "Male"}

# Realistic min/max ranges for each numeric field
VALIDATION_RULES = {
    "CreditScore": (300, 900),
    "Age": (18, 100),
    "Tenure": (0, 15),
    "Balance": (0, 300000),
    "NumOfProducts": (1, 4),
    "HasCrCard": (0, 1),
    "IsActiveMember": (0, 1),
    "EstimatedSalary": (0, 1000000),
}


class InvalidCustomerData(ValueError):
    # Raised when the input is missing a field, has an extra field, or
    # has a value that doesn't make sense
    pass


def _load_pipeline():
    if not ARTIFACT_PATH.exists():
        raise FileNotFoundError(
            f"No trained pipeline found at {ARTIFACT_PATH}. "
            "Run evaluate.py first to train and save it."
        )
    return joblib.load(ARTIFACT_PATH)


def validate_customer_data(customer_data):
    # Check for fields that shouldn't be there
    for field, reason in REJECTED_FIELDS.items():
        if field in customer_data:
            raise InvalidCustomerData(reason)

    # Check for missing fields
    missing = [col for col in FEATURE_COLUMNS if col not in customer_data]
    if missing:
        raise InvalidCustomerData(f"Missing required field(s): {missing}")

    # Check for extra fields we don't use
    extra = [key for key in customer_data if key not in FEATURE_COLUMNS]
    if extra:
        raise InvalidCustomerData(f"Unexpected field(s): {extra}")

    # Check categorical values
    if customer_data["Geography"] not in VALID_GEOGRAPHY:
        raise InvalidCustomerData(f"Geography must be one of {sorted(VALID_GEOGRAPHY)}")

    if customer_data["Gender"] not in VALID_GENDER:
        raise InvalidCustomerData(f"Gender must be one of {sorted(VALID_GENDER)}")

    # Check numeric ranges
    for field, bounds in VALIDATION_RULES.items():
        low, high = bounds
        value = customer_data[field]

        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise InvalidCustomerData(f"{field} must be a number")

        if value < low or value > high:
            raise InvalidCustomerData(f"{field}={value} is outside the realistic range [{low}, {high}]")


def _risk_message(probability):
    if probability >= 0.6:
        return "High churn risk. Recommend proactive retention outreach."
    elif probability >= 0.35:
        return "Moderate churn risk. Consider monitoring or a light-touch check-in."
    else:
        return "Low churn risk. No action needed."


def predict_churn_risk(customer_data):
    # customer_data should be a dict with the 10 model input fields.
    # This is a support signal for a human reviewer, not an automatic
    # decision.
    validate_customer_data(customer_data)

    pipeline = _load_pipeline()

    row = pd.DataFrame([{col: customer_data[col] for col in FEATURE_COLUMNS}])
    probability = float(pipeline.predict_proba(row)[0, 1])

    if probability >= 0.5:
        label = "high_risk"
    else:
        label = "low_risk"

    return {
        "churn_probability": round(probability, 4),
        "risk_label": label,
        "message": _risk_message(probability),
    }


if __name__ == "__main__":
    example_customer = {
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
    }
    print(predict_churn_risk(example_customer))
