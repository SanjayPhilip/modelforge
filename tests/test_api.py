"""Tests for FastAPI endpoints."""
import pytest
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)


def test_root():
    """Test root endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["message"] == "Welcome to ModelForge API"
    assert "version" in response.json()


def test_health_without_model():
    """Test health endpoint when model is not loaded."""
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    # Will be unhealthy if model not trained yet
    assert "status" in data
    assert "model_loaded" in data


def test_model_info():
    """Test model info endpoint."""
    response = client.get("/api/v1/info")
    assert response.status_code == 200
    data = response.json()
    assert data["model_name"] == "California Housing Price Predictor"
    assert data["n_features"] == 8
    assert len(data["feature_names"]) == 8


def test_readiness():
    """Test readiness probe."""
    response = client.get("/ready")
    assert response.status_code == 200
    assert response.json()["status"] == "ready"


def test_liveness():
    """Test liveness probe."""
    response = client.get("/live")
    assert response.status_code == 200
    assert response.json()["status"] == "alive"
