"""Inference wrapper for model predictions."""
import joblib
import numpy as np
from pathlib import Path
from typing import List, Union


class ModelPredictor:
    """Wrapper for loading model and making predictions."""

    def __init__(self, model_path: str = "models/model.pkl", scaler_path: str = "models/scaler.pkl"):
        self.model_path = model_path
        self.scaler_path = scaler_path
        self.model = None
        self.scaler = None
        self._load_model()

    def _load_model(self):
        """Load model and scaler from disk."""
        if not Path(self.model_path).exists():
            raise FileNotFoundError(f"Model not found at {self.model_path}. Run training first.")

        self.model = joblib.load(self.model_path)

        if Path(self.scaler_path).exists():
            self.scaler = joblib.load(self.scaler_path)

    def predict(self, features: Union[List[float], List[List[float]]]) -> List[float]:
        """Make prediction for given features."""
        if isinstance(features[0], (int, float)):
            features = [features]

        features_array = np.array(features)

        # Apply scaling if scaler exists
        if self.scaler is not None:
            features_array = self.scaler.transform(features_array)

        predictions = self.model.predict(features_array)
        return predictions.tolist()

    def get_model_info(self) -> dict:
        """Return model metadata."""
        return {
            "model_type": type(self.model).__name__,
            "model_path": self.model_path,
            "scaler_loaded": self.scaler is not None,
            "n_features": self.model.n_features_in_ if hasattr(self.model, "n_features_in_") else None,
        }
