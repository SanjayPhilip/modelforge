"""Utility functions for data preprocessing and feature engineering."""
import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
from pathlib import Path


def load_california_housing(test_size=0.2, random_state=42):
    """Load and split California housing dataset."""
    data = fetch_california_housing(as_frame=True)
    X = data.data
    y = data.target

    feature_names = list(data.feature_names)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    return X_train, X_test, y_train, y_test, feature_names


def preprocess_data(X_train, X_test, scaler_path="models/scaler.pkl"):
    """Standardize features using StandardScaler."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Save scaler for inference
    Path(scaler_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, scaler_path)

    return X_train_scaled, X_test_scaled, scaler


def save_processed_data(X_train, X_test, y_train, y_test, output_dir="data/processed"):
    """Save processed datasets to CSV."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    pd.DataFrame(X_train).to_csv(f"{output_dir}/X_train.csv", index=False)
    pd.DataFrame(X_test).to_csv(f"{output_dir}/X_test.csv", index=False)
    pd.DataFrame(y_train).to_csv(f"{output_dir}/y_train.csv", index=False)
    pd.DataFrame(y_test).to_csv(f"{output_dir}/y_test.csv", index=False)
