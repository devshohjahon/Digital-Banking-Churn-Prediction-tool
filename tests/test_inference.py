import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest

from churn_predictor.predict import ARTIFACT_PATH, InvalidCustomerData, predict_churn_risk

VALID_RECORD = {
    "CreditScore": 650,
    "Geography": "Germany",
    "Gender": "Male",
    "Age": 35,
    "Tenure": 5,
    "Balance": 75000.0,
    "NumOfProducts": 2,
    "HasCrCard": 1,
    "IsActiveMember": 0,
    "EstimatedSalary": 60000.0,
}

requires_artifact = pytest.mark.skipif(
    not ARTIFACT_PATH.exists(), reason="Trained pipeline artifact not built yet"
)


@requires_artifact
def test_valid_input_returns_probability_and_message():
    result = predict_churn_risk(VALID_RECORD)
    assert 0.0 <= result["churn_probability"] <= 1.0
    assert result["risk_label"] in {"high_risk", "low_risk"}
    assert isinstance(result["message"], str) and len(result["message"]) > 0


@requires_artifact
def test_missing_field_is_rejected():
    bad = VALID_RECORD.copy()
    del bad["Balance"]
    with pytest.raises(InvalidCustomerData):
        predict_churn_risk(bad)


@requires_artifact
def test_target_column_is_rejected():
    bad = VALID_RECORD.copy()
    bad["Exited"] = 1
    with pytest.raises(InvalidCustomerData):
        predict_churn_risk(bad)


@requires_artifact
def test_impossible_value_is_rejected():
    bad = VALID_RECORD.copy()
    bad["Age"] = 250
    with pytest.raises(InvalidCustomerData):
        predict_churn_risk(bad)


@requires_artifact
def test_invalid_geography_is_rejected():
    bad = VALID_RECORD.copy()
    bad["Geography"] = "Narnia"
    with pytest.raises(InvalidCustomerData):
        predict_churn_risk(bad)
