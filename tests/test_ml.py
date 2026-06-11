"""Tests for ML pipeline components."""
import pytest
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from ml.utils import load_california_housing, preprocess_data
from ml.predict import ModelPredictor
import joblib
from pathlib import Path


def test_load_california_housing():
    """Test data loading function."""
    X_train, X_test, y_train, y_test, feature_names = load_california_housing()

    assert len(X_train) > 0
    assert len(X_test) > 0
    assert len(X_train) > len(X_test)  # 80/20 split
    assert len(feature_names) == 8
    assert "MedInc" in feature_names


def test_preprocess_data():
    """Test data preprocessing."""
    X_train, X_test, y_train, y_test, _ = load_california_housing()
    X_train_scaled, X_test_scaled, scaler = preprocess_data(X_train, X_test)

    # Check shapes preserved
    assert X_train_scaled.shape == X_train.shape
    assert X_test_scaled.shape == X_test.shape

    # Check scaler was saved
    assert Path("models/scaler.pkl").exists()


def test_model_predictor_init_without_model():
    """Test predictor raises error when model missing."""
    with pytest.raises(FileNotFoundError):
        predictor = ModelPredictor(model_path="nonexistent.pkl")


def test_random_forest_training():
    """Test basic model training."""
    X_train, X_test, y_train, y_test, _ = load_california_housing()

    model = RandomForestRegressor(n_estimators=10, random_state=42)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    assert len(predictions) == len(y_test)
    assert all(np.isfinite(predictions))
