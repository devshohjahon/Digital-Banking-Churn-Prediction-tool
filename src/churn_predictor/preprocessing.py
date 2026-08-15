# preprocessing.py
#
# Turns the raw dataframe into features (X) and target (y), and builds the
# scikit-learn pipeline that scales/encodes the features and fits a model.
#
# Leakage rules we follow here:
#   1. RowNumber, CustomerId, Surname are just IDs, not real attributes,
#      so we never give them to the model.
#   2. Exited is the target, it is never a feature.
#   3. Scaling and encoding are done inside a Pipeline, so they are only
#      "fit" on training data and reused as-is on validation/test/new data.

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from churn_predictor.data import ID_COLUMNS, TARGET_COLUMN

# Numeric columns we scale
NUMERIC_FEATURES = ["CreditScore", "Age", "Tenure", "Balance", "NumOfProducts", "EstimatedSalary"]

# Already 0/1 flags
BINARY_FEATURES = ["HasCrCard", "IsActiveMember"]

# Text categories we one-hot encode
CATEGORICAL_FEATURES = ["Geography", "Gender"]

FEATURE_COLUMNS = NUMERIC_FEATURES + BINARY_FEATURES + CATEGORICAL_FEATURES


def split_features_target(df):
    # Drop ID columns and the target, keep everything else as features
    columns_to_drop = ID_COLUMNS + [TARGET_COLUMN]
    X = df.drop(columns=columns_to_drop)
    y = df[TARGET_COLUMN].astype(int)

    # Only keep the columns we actually want to use, in a fixed order
    X = X[FEATURE_COLUMNS]

    return X, y


def build_preprocessor():
    # Scale numeric/binary columns, one-hot encode categorical columns
    preprocessor = ColumnTransformer(transformers=[
        ("num", StandardScaler(), NUMERIC_FEATURES + BINARY_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore", drop="if_binary"), CATEGORICAL_FEATURES),
    ])
    return preprocessor


def build_pipeline(model):
    # Bundle preprocessing + model together so they always travel as one
    # unit (same transform used in training and at prediction time)
    pipeline = Pipeline(steps=[
        ("preprocess", build_preprocessor()),
        ("model", model),
    ])
    return pipeline
