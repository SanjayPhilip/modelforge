# 🚀 ModelForge — End-to-End MLOps Pipeline

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=flat&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![MLflow](https://img.shields.io/badge/MLflow-0194E2?style=flat&logo=mlflow&logoColor=white)](https://mlflow.org)
[![Docker](https://img.shields.io/badge/Docker-2496ED?style=flat&logo=docker&logoColor=white)](https://docker.com)
[![GitHub Actions](https://img.shields.io/badge/GitHub_Actions-2088FF?style=flat&logo=github-actions&logoColor=white)](https://github.com/features/actions)

A production-ready MLOps pipeline that demonstrates the complete lifecycle of a machine learning model — from training and experiment tracking to containerization and cloud deployment.

## 📋 Table of Contents

- [Architecture](#architecture)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Quick Start](#quick-start)
- [API Documentation](#api-documentation)
- [CI/CD Pipeline](#cicd-pipeline)
- [Deployment](#deployment)
- [Project Structure](#project-structure)
- [Screenshots](#screenshots)
- [Roadmap](#roadmap)

## 🏗️ Architecture

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   GitHub    │────▶│ GitHub      │────▶│   Docker    │────▶│   Render/   │
│   Repo      │     │ Actions     │     │   Hub       │     │   AWS       │
└─────────────┘     └─────────────┘     └─────────────┘     └─────────────┘
       │                   │                   │                   │
       ▼                   ▼                   ▼                   ▼
  ┌─────────┐        ┌─────────┐        ┌─────────┐        ┌─────────┐
  │  Code   │        │  Tests  │        │  Build  │        │  Deploy │
  │  Push   │        │  Train  │        │  Image  │        │  API    │
  └─────────┘        └─────────┘        └─────────┘        └─────────┘
                                                          ▲
                                                          │
                                                   ┌─────────────┐
                                                   │   MLflow    │
                                                   │  (Tracking) │
                                                   └─────────────┘
```

## ✨ Features

- 🎯 **ML Model Training** — RandomForest regression on California Housing dataset
- 📊 **Experiment Tracking** — Full MLflow integration for parameters, metrics, and artifacts
- 🐳 **Containerization** — Multi-stage Docker build for optimized production images
- 🔄 **CI/CD Pipeline** — Automated training, testing, and deployment via GitHub Actions
- 🌐 **REST API** — FastAPI with auto-generated docs, batch predictions, and health checks
- 📈 **Monitoring Ready** — Prometheus/Grafana hooks for production observability
- ☁️ **Cloud Deployed** — One-click deployment to Render, AWS, or Railway

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| **API Framework** | FastAPI |
| **ML Framework** | Scikit-learn, XGBoost |
| **Experiment Tracking** | MLflow |
| **Containerization** | Docker, Docker Compose |
| **CI/CD** | GitHub Actions |
| **Database** | PostgreSQL (MLflow backend) |
| **Cloud** | Render / AWS EC2 / Railway |
| **Testing** | Pytest |

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- Docker & Docker Compose
- Git

### 1. Clone & Setup

```bash
git clone https://github.com/SanjayPhilip/modelforge.git
cd modelforge

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Environment Configuration

```bash
cp .env.example .env
# Edit .env with your credentials (optional for local development)
```

### 3. Train the Model

```bash
# Train with default parameters
python ml/train.py

# Or with custom hyperparameters
python ml/train.py --n-estimators 200 --max-depth 15
```

### 4. Run API Locally

```bash
# Option A: Direct Python
uvicorn api.main:app --reload

# Option B: Docker Compose (Full stack with MLflow)
cd docker
docker-compose up -d

# Access API: http://localhost:8000
# Access MLflow UI: http://localhost:5000
# Access API Docs: http://localhost:8000/docs
```

### 5. Make Predictions

```bash
# Health check
curl http://localhost:8000/api/v1/health

# Single prediction
curl -X POST "http://localhost:8000/api/v1/predict" \
  -H "Content-Type: application/json" \
  -d '{"features": [8.3252, 41.0, 6.9841, 1.0238, 322.0, 2.5555, 37.88, -122.23]}'

# Batch prediction
curl -X POST "http://localhost:8000/api/v1/predict/batch" \
  -H "Content-Type: application/json" \
  -d '{"features": [[8.3252, 41.0, 6.9841, 1.0238, 322.0, 2.5555, 37.88, -122.23], [8.3014, 21.0, 6.2381, 0.9718, 2401.0, 2.1098, 37.86, -122.22]]}'
```

## 📚 API Documentation

Once running, interactive API documentation is available at:

- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

### Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | API info and links |
| GET | `/health` | Health check with model status |
| GET | `/ready` | Kubernetes readiness probe |
| GET | `/live` | Kubernetes liveness probe |
| GET | `/api/v1/health` | Detailed health check |
| GET | `/api/v1/info` | Model metadata |
| POST | `/api/v1/predict` | Single prediction |
| POST | `/api/v1/predict/batch` | Batch predictions |

## 🔄 CI/CD Pipeline

### Pull Request Flow
```
Push to PR branch
    │
    ▼
GitHub Actions Triggered
    │
    ├──▶ Install Dependencies
    ├──▶ Run Pytest (Unit + Integration)
    ├──▶ Train Model
    └──▶ Upload Model Artifact
```

### Main Branch Deployment
```
Push to main branch
    │
    ▼
GitHub Actions Triggered
    │
    ├──▶ Build Docker Image
    ├──▶ Push to Docker Hub
    └──▶ Deploy to Render (via webhook)
```

## 🌐 Deployment

### Option A: Render (Recommended — Free Tier)

1. Create a Web Service on [Render](https://render.com)
2. Connect your Docker Hub image: `sanjayphilip/modelforge:latest`
3. Set environment variables in Render dashboard
4. Add `RENDER_DEPLOY_HOOK` to GitHub Secrets for auto-deployment

### Option B: AWS EC2

```bash
# SSH into your EC2 instance
ssh -i your-key.pem ubuntu@your-ec2-ip

# Install Docker
curl -fsSL https://get.docker.com | sh
sudo usermod -aG docker ubuntu

# Pull and run
docker pull sanjayphilip/modelforge:latest
docker run -d -p 80:8000 --name modelforge sanjayphilip/modelforge:latest
```

### Option C: Railway (Your Existing Stack)

1. Connect GitHub repo to Railway
2. Railway auto-detects Dockerfile
3. Add PostgreSQL plugin for MLflow backend

## 📁 Project Structure

```
modelforge/
├── .github/
│   └── workflows/
│       ├── train.yml          # PR: Train + Test + Evaluate
│       └── deploy.yml         # Main: Build + Push + Deploy
├── api/
│   ├── main.py                # FastAPI app entry point
│   ├── router.py              # API route handlers
│   └── schemas.py             # Pydantic request/response models
├── ml/
│   ├── train.py               # Training script with MLflow
│   ├── evaluate.py            # Model evaluation utilities
│   ├── predict.py             # Inference wrapper
│   └── utils.py               # Data preprocessing
├── models/                    # Trained models (gitignored)
├── data/
│   ├── raw/                   # Raw datasets
│   └── processed/             # Cleaned datasets
├── notebooks/
│   └── eda.ipynb              # Exploratory Data Analysis
├── tests/
│   ├── test_api.py            # API endpoint tests
│   └── test_ml.py             # ML pipeline tests
├── docker/
│   ├── Dockerfile             # Multi-stage build
│   └── docker-compose.yml     # Full stack orchestration
├── requirements.txt
├── setup.py
├── .env.example
└── README.md
```

## 📸 Screenshots

*Add screenshots of:*
- *MLflow UI with experiment runs*
- *FastAPI Swagger docs*
- *GitHub Actions pipeline*
- *Docker containers running*

## 🗺️ Roadmap

- [ ] Add Prometheus + Grafana monitoring
- [ ] Implement model versioning and A/B testing
- [ ] Add data drift detection
- [ ] Integrate with AWS SageMaker
- [ ] Add authentication to API endpoints
- [ ] Build Streamlit frontend for interactive predictions
- [ ] Add model explainability (SHAP values)

## 👤 Author

**Sanjay Philip**
- GitHub: [@SanjayPhilip](https://github.com/SanjayPhilip)
- MCA Final Year | Saintgits College of Engineering, Kerala, India
- Passionate about Web Development, AI/ML, and Cybersecurity

## 📄 License

This project is licensed under the MIT License.

---

⭐ **Star this repo if you find it useful!**
