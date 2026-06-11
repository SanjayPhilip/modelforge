"""Model evaluation utilities."""
import json
from pathlib import Path
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np


def evaluate_model(model, X_test, y_test):
    """Evaluate model and return comprehensive metrics."""
    y_pred = model.predict(X_test)

    metrics = {
        "rmse": float(mean_squared_error(y_test, y_pred, squared=False)),
        "mae": float(mean_absolute_error(y_test, y_pred)),
        "r2_score": float(r2_score(y_test, y_pred)),
        "mape": float(np.mean(np.abs((y_test - y_pred) / y_test)) * 100),
    }

    return metrics, y_pred


def save_metrics(metrics, output_path="models/metrics.json"):
    """Save metrics to JSON file."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"📊 Metrics saved to {output_path}")
