"""FastAPI application entry point."""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.router import router
import os

# Create FastAPI app
app = FastAPI(
    title="ModelForge API",
    description="End-to-End MLOps Pipeline - California Housing Price Prediction",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(router, prefix="/api/v1", tags=["predictions"])


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Welcome to ModelForge API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/api/v1/health",
        "info": "/api/v1/info",
    }


@app.get("/ready")
async def readiness():
    """Kubernetes-style readiness probe."""
    return {"status": "ready"}


@app.get("/live")
async def liveness():
    """Kubernetes-style liveness probe."""
    return {"status": "alive"}


if __name__ == "__main__":
    import uvicorn
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", "8000"))
    uvicorn.run(app, host=host, port=port)
