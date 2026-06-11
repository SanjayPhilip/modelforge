"""Training script with MLflow experiment tracking."""
import os
import sys
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
from pathlib import Path
from dotenv import load_dotenv

from ml.utils import load_california_housing, preprocess_data

# Load environment variables
load_dotenv()


def train(
    n_estimators: int = 100,
    max_depth: int = 10,
    min_samples_split: int = 2,
    random_state: int = 42,
    model_output_path: str = "models/model.pkl",
):
    """Train a RandomForest model and log to MLflow."""

    # Load data
    print("📊 Loading California Housing dataset...")
    X_train, X_test, y_train, y_test, feature_names = load_california_housing()

    # Preprocess
    print("🔧 Preprocessing data...")
    X_train_scaled, X_test_scaled, scaler = preprocess_data(X_train, X_test)

    # Set MLflow tracking
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME", "california-housing")

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name="random-forest-training"):
        # Log parameters
        mlflow.log_param("model_type", "RandomForestRegressor")
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("min_samples_split", min_samples_split)
        mlflow.log_param("random_state", random_state)
        mlflow.log_param("dataset", "California Housing")
        mlflow.log_param("features", len(feature_names))
        mlflow.log_param("feature_names", feature_names)

        # Train model
        print("🚀 Training RandomForest model...")
        model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            random_state=random_state,
            n_jobs=-1,
        )
        model.fit(X_train_scaled, y_train)

        # Evaluate
        print("📈 Evaluating model...")
        y_pred = model.predict(X_test_scaled)

        rmse = mean_squared_error(y_test, y_pred, squared=False)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        # Log metrics
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2_score", r2)

        # Log model
        mlflow.sklearn.log_model(model, "model")

        # Save model locally
        Path(model_output_path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, model_output_path)
        mlflow.log_artifact(model_output_path)

        # Save scaler too
        mlflow.log_artifact("models/scaler.pkl")

        print(f"\n✅ Training Complete!")
        print(f"   RMSE: {rmse:.4f}")
        print(f"   MAE:  {mae:.4f}")
        print(f"   R²:   {r2:.4f}")
        print(f"\n📁 Model saved to: {model_output_path}")
        print(f"🔗 MLflow Run ID: {mlflow.active_run().info.run_id}")

        return {
            "rmse": rmse,
            "mae": mae,
            "r2": r2,
            "model_path": model_output_path,
            "run_id": mlflow.active_run().info.run_id,
        }


if __name__ == "__main__":
    # Allow command-line args for hyperparameters
    import argparse

    parser = argparse.ArgumentParser(description="Train California Housing Model")
    parser.add_argument("--n-estimators", type=int, default=100)
    parser.add_argument("--max-depth", type=int, default=10)
    parser.add_argument("--min-samples-split", type=int, default=2)

    args = parser.parse_args()

    train(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        min_samples_split=args.min_samples_split,
    )
