"""API route handlers for prediction endpoints."""
from fastapi import APIRouter, HTTPException
from api.schemas import (
    PredictionRequest, PredictionResponse,
    BatchPredictionRequest, BatchPredictionResponse,
    HealthResponse, ModelInfoResponse
)
from ml.predict import ModelPredictor
import os

router = APIRouter()

# Initialize predictor (lazy loading handled in class)
predictor = None


def get_predictor():
    """Get or initialize model predictor."""
    global predictor
    if predictor is None:
        model_path = os.getenv("MODEL_PATH", "models/model.pkl")
        scaler_path = os.getenv("SCALER_PATH", "models/scaler.pkl")
        predictor = ModelPredictor(model_path=model_path, scaler_path=scaler_path)
    return predictor


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint to verify model status."""
    try:
        pred = get_predictor()
        info = pred.get_model_info()
        return HealthResponse(
            status="healthy",
            model_loaded=True,
            model_type=info.get("model_type"),
            scaler_loaded=info.get("scaler_loaded", False)
        )
    except Exception as e:
        return HealthResponse(
            status="unhealthy",
            model_loaded=False,
            scaler_loaded=False
        )


@router.get("/info", response_model=ModelInfoResponse)
async def model_info():
    """Return model metadata and information."""
    return ModelInfoResponse()


@router.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Make a single prediction."""
    try:
        pred = get_predictor()
        predictions = pred.predict(request.features)

        return PredictionResponse(
            prediction=predictions[0],
            model_version="1.0.0",
            model_type=pred.get_model_info().get("model_type", "Unknown")
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@router.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(request: BatchPredictionRequest):
    """Make batch predictions."""
    try:
        pred = get_predictor()
        predictions = pred.predict(request.features)

        return BatchPredictionResponse(
            predictions=predictions,
            model_version="1.0.0",
            count=len(predictions)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")
