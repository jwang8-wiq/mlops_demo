from fastapi import FastAPI, Request
from application.src.create_service import load_model
from application.src.predict import router as predict_router
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
import numpy as np
import pandas as pd
import boto3
import os
from botocore.exceptions import NoCredentialsError
import requests

# Environment variables for MinIO
MINIO_URL = os.getenv("MINIO_URL", "http://localhost:9000")
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID", "minioadmin")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY", "minioadmin")
REFERENCE_DATA_BUCKET = "models"
REFERENCE_DATA_KEY = "reference_data.csv"

def load_reference_data_from_minio():
    """
    Load reference data from MinIO.
    """
    s3_client = boto3.client(
        "s3",
        endpoint_url=MINIO_URL,
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    )
    try:
        # Download reference data from MinIO
        local_reference_path = "/tmp/reference_data.csv"
        s3_client.download_file(REFERENCE_DATA_BUCKET, REFERENCE_DATA_KEY, local_reference_path)
        reference_data = pd.read_csv(local_reference_path)
        print("Reference data loaded successfully from MinIO.")
        return reference_data
    except NoCredentialsError:
        raise RuntimeError("Invalid MinIO credentials")
    except Exception as e:
        raise RuntimeError(f"Failed to load reference data from MinIO: {e}")


def create_app() -> FastAPI:
    """
    Create and configure the FastAPI application.
    """
    app = FastAPI(title="Churn Prediction API", version="1.0.0")

    # Load the best model (ensures model is available globally)
    app.state.model = load_model()

    # Load reference data from MinIO
    reference_data = load_reference_data_from_minio()

    # Include prediction routes
    app.include_router(predict_router, prefix="/api", tags=["predictions"])

    # Add health check route
    @app.get("/health")
    def health_check():
        return {"status": "healthy"}

    return app
