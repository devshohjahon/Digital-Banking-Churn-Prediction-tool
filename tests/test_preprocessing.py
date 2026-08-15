import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd
import pytest

from churn_predictor.data import ID_COLUMNS, TARGET_COLUMN, load_raw_data
from churn_predictor.preprocessing import FEATURE_COLUMNS, build_pipeline, split_features_target
from sklearn.linear_model import LogisticRegression


def test_raw_data_loads_and_has_expected_shape():
    df = load_raw_data()
    assert len(df) == 10000
    assert TARGET_COLUMN in df.columns


def test_split_features_target_drops_identifiers_and_target():
    df = load_raw_data()
    X, y = split_features_target(df)

    for col in ID_COLUMNS:
        assert col not in X.columns, f"{col} is an identifier and must not be a model input"
    assert TARGET_COLUMN not in X.columns
    assert list(X.columns) == FEATURE_COLUMNS
    assert len(X) == len(y) == len(df)


def test_pipeline_fits_and_predicts_on_one_raw_record():
    df = load_raw_data()
    X, y = split_features_target(df)

    pipeline = build_pipeline(LogisticRegression(max_iter=200))
    pipeline.fit(X.iloc[:500], y.iloc[:500])

    one_record = X.iloc[[0]]
    pred = pipeline.predict(one_record)
    proba = pipeline.predict_proba(one_record)

    assert pred.shape == (1,)
    assert proba.shape == (1, 2)
    assert 0.0 <= proba[0, 1] <= 1.0


def test_no_leakage_columns_reach_the_model():
    """Explicit guard: RowNumber/CustomerId/Surname/Exited must never be
    inside the feature columns the ColumnTransformer touches."""
    leakage_columns = set(ID_COLUMNS) | {TARGET_COLUMN}
    assert leakage_columns.isdisjoint(set(FEATURE_COLUMNS))
