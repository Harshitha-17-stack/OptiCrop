"""
preprocessing.py
-----------------
Data loading, cleaning, and preprocessing utilities for the OptiCrop
Smart Agricultural Production Optimization Engine.

This module is shared by model_training.py (offline training) and can
also be re-used by app.py if you want to preprocess raw uploads at
inference time. The live Flask app currently receives already-numeric
form values, so it uses the saved scaler directly (see app.py).
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

DATA_PATH = "dataset/Crop_recommendation.csv"

FEATURE_COLUMNS = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
TARGET_COLUMN = "label"


def load_dataset(path: str = DATA_PATH) -> pd.DataFrame:
    """Load the raw crop recommendation dataset from disk."""
    df = pd.read_csv(path)
    return df


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Basic cleaning steps:
      - Drop exact duplicate rows.
      - Drop rows with missing values in the feature/target columns.
      - Guard against negative or clearly invalid sensor readings.
    """
    df = df.drop_duplicates()
    df = df.dropna(subset=FEATURE_COLUMNS + [TARGET_COLUMN])

    # Sensor readings should never be negative; guard against bad rows.
    for col in FEATURE_COLUMNS:
        df = df[df[col] >= 0]

    df = df.reset_index(drop=True)
    return df


def encode_labels(df: pd.DataFrame):
    """Fit a LabelEncoder on the crop label column."""
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(df[TARGET_COLUMN])
    return y_encoded, encoder


def split_and_scale(df: pd.DataFrame, y_encoded, test_size: float = 0.2, random_state: int = 42):
    """
    Split into train/test (stratified so every crop is represented
    proportionally in both splits), then fit a StandardScaler on the
    training partition only, to avoid test-set leakage.
    """
    X = df[FEATURE_COLUMNS]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded,
        test_size=test_size,
        random_state=random_state,
        stratify=y_encoded,
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


def prepare_data(path: str = DATA_PATH):
    """
    Convenience wrapper that runs the full pipeline:
    load -> clean -> encode -> split -> scale.

    Returns a dict with everything model_training.py needs.
    """
    df = load_dataset(path)
    df = clean_dataset(df)
    y_encoded, encoder = encode_labels(df)
    X_train, X_test, y_train, y_test, scaler = split_and_scale(df, y_encoded)

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "scaler": scaler,
        "encoder": encoder,
        "feature_columns": FEATURE_COLUMNS,
        "raw_df": df,
    }


if __name__ == "__main__":
    data = prepare_data()
    print(f"Loaded {len(data['raw_df'])} clean rows across "
          f"{len(data['encoder'].classes_)} crop classes.")
    print("Training set:", data["X_train"].shape, " Test set:", data["X_test"].shape)
