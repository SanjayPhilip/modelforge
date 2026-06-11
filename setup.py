from setuptools import setup, find_packages

setup(
    name="modelforge",
    version="1.0.0",
    description="End-to-End MLOps Pipeline with FastAPI, MLflow, and Docker",
    author="Sanjay Philip",
    packages=find_packages(),
    install_requires=[
        "fastapi>=0.115.0",
        "uvicorn>=0.30.0",
        "scikit-learn>=1.5.0",
        "mlflow>=2.15.0",
        "joblib>=1.4.0",
        "numpy>=1.26.0",
        "pandas>=2.2.0",
    ],
    python_requires=">=3.9",
)