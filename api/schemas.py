"""Pydantic models for API request/response validation."""
from pydantic import BaseModel, Field
from typing import List, Optional


class PredictionRequest(BaseModel):
    """Request model for single prediction."""
    features: List[float] = Field(
        ...,
        min_length=8,
        max_length=8,
        description="8 feature values for California Housing prediction",
        example=[8.3252, 41.0, 6.98412698, 1.02380952, 322.0, 2.55555556, 37.88, -122.23]
    )


class BatchPredictionRequest(BaseModel):
    """Request model for batch predictions."""
    features: List[List[float]] = Field(
        ...,
        description="List of feature vectors for batch prediction",
        example=[
            [8.3252, 41.0, 6.98412698, 1.02380952, 322.0, 2.55555556, 37.88, -122.23],
            [8.3014, 21.0, 6.23813708, 0.97188049, 2401.0, 2.10984183, 37.86, -122.22]
        ]
    )


class PredictionResponse(BaseModel):
    """Response model for prediction."""
    prediction: float
    model_version: str = "1.0.0"
    model_type: str = "RandomForestRegressor"


class BatchPredictionResponse(BaseModel):
    """Response model for batch predictions."""
    predictions: List[float]
    model_version: str = "1.0.0"
    count: int


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model_loaded: bool
    model_type: Optional[str] = None
    scaler_loaded: bool = False


class ModelInfoResponse(BaseModel):
    """Model information response."""
    model_name: str = "California Housing Price Predictor"
    model_type: str = "RandomForestRegressor"
    model_version: str = "1.0.0"
    dataset: str = "California Housing"
    n_features: int = 8
    feature_names: List[str] = [
        "MedInc", "HouseAge", "AveRooms", "AveBedrms",
        "Population", "AveOccup", "Latitude", "Longitude"
    ]
    description: str = "Predicts median house value in California districts"
