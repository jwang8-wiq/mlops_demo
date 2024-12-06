from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import Response, FileResponse
from pydantic import BaseModel, Field
import boto3
import pandas as pd
from botocore.exceptions import NoCredentialsError
from fastapi.logger import logger
import logging
import requests

import os
import sys


# Add the project root directory to the PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

# Now import the drift detection module
from application.src.drift_detection import  detect_drift_and_generate_report

from application.src.create_service import load_model
from application.src.predict import router as predict_router


from prometheus_client import Gauge, Counter, generate_latest, CONTENT_TYPE_LATEST

# Configure the logger
logging.basicConfig(level=logging.INFO)

# Environment variables for MinIO
MINIO_URL = os.getenv("MINIO_URL", "http://localhost:9000")
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID", "minioadmin")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY", "minioadmin")
MODEL_BUCKET = "models"
MODEL_KEY_PREFIX = "best_model"
CURRENT_DATA_BUCKET = "data"
CURRENT_DATA_KEY = "app/data/processed/X_test.csv"
WEBHOOK_URL = os.getenv("WEBHOOK_URL", "http://argo-events-service.argo-events.svc.cluster.local:12000/drift")

# Prometheus metrics
data_drift_score = Gauge("data_drift_score", "Latest data drift score")
drift_detected = Counter("drift_detected", "Number of times drift was detected")

minio_client = boto3.client(
    "s3",
    endpoint_url=MINIO_URL,
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
)

# Input schema for drift simulation
class DriftSimulationInput(BaseModel):
    drift_type: str = Field(
        ...,
        example="numerical_shift",
        description="Type of drift to simulate. Options: 'numerical_shift', 'category_mismatch', 'drop_column'",
    )

# Helper to fetch the latest model file from MinIO
def fetch_latest_model_info():

    try:
        response = minio_client.list_objects_v2(Bucket=MODEL_BUCKET, Prefix=MODEL_KEY_PREFIX)
        if 'Contents' not in response:
            raise RuntimeError("No models found in MinIO bucket.")
        
        # Find the latest model by timestamp
        latest_model = max(response['Contents'], key=lambda obj: obj['LastModified'])
        return latest_model['Key']
    except NoCredentialsError:
        raise RuntimeError("Invalid MinIO credentials")
    except Exception as e:
        raise RuntimeError(f"Failed to fetch latest model info: {e}")

# Load model from MinIO
def load_model_from_minio():
    latest_model_key = fetch_latest_model_info()
    local_model_path = f"/tmp/{latest_model_key.split('/')[-1]}"
    
    try:
        minio_client.download_file(MODEL_BUCKET, latest_model_key, local_model_path)
        print(f"Loaded model from MinIO: {latest_model_key}")
        # return load_model(local_model_path)
        return load_model()
    except Exception as e:
        raise RuntimeError(f"Failed to load model from MinIO: {e}")

# Create FastAPI app
def create_app() -> FastAPI:
    app = FastAPI(title="Churn Prediction API", version="1.0.0")

    # Load the initial model
    app.state.model = load_model_from_minio()

    # Include prediction routes
    app.include_router(predict_router, prefix="/api", tags=["predictions"])


    @app.post("/simulate-drift")
    def simulate_drift(input: DriftSimulationInput):
        """
        Simulate drift in the current dataset and store the drifted dataset in MinIO.
        Drift Types:
        - numerical_shift: Increases numerical values in a column (e.g., MonthlyCharges).
        - category_mismatch: Replaces categorical values with new ones (e.g., PaymentMethod).
        - drop_column: Removes a specific column from the dataset (e.g., Contract).
        """
        drift_type = input.drift_type
        try:
            # Fetch the current dataset from MinIO
            current_data_path = "/tmp/current_data.csv"
            minio_client.download_file(CURRENT_DATA_BUCKET, CURRENT_DATA_KEY, current_data_path)
            data = pd.read_csv(current_data_path)

            # Simulate drift
            if drift_type == "numerical_shift":
                data["MonthlyCharges"] *= 1.5
            elif drift_type == "category_mismatch":
                data["PaymentMethod"] = data["PaymentMethod"].replace("Electronic check", "Crypto")
            elif drift_type == "drop_column":
                if "Contract" in data.columns:
                    data.drop(columns=["Contract"], inplace=True)
            else:
                raise HTTPException(status_code=400, detail=f"Unknown drift type: {drift_type}")

            # Save drifted dataset back to MinIO
            drifted_data_path = "/tmp/current_data.csv"
            data.to_csv(drifted_data_path, index=False)
            minio_client.upload_file(drifted_data_path, CURRENT_DATA_BUCKET, CURRENT_DATA_KEY)

            return {"status": "success", "message": "Drift simulated"}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))



    @app.post("/detect-drift")
    def detect_drift_endpoint():
        """
        Trigger drift detection, generate an HTML report, and return the drift score.
        """
        try:
            logger.info("Starting drift detection process.")
            
            # Perform drift detection and generate report
            score, html_report_path = detect_drift_and_generate_report()
            logger.info(f"Drift detection completed. Drift score: {score}. Report path: {html_report_path}")
            
            # Update Prometheus metrics
            data_drift_score.set(score)
            logger.info("Updated Prometheus metric: data_drift_score.")
            
            # Check drift threshold and take action if needed
            if score > 0.3:  # Example threshold
                drift_detected.inc()
                logger.info("Drift detected. Incremented Prometheus metric: drift_detected.")
                
                # Trigger Argo Events webhook
                webhook_url = os.getenv("WEBHOOK_URL", "http://drift-detection-eventsource-svc.argo-events.svc.cluster.local:12000/drift-detected")
                response = requests.post(webhook_url, json={"drift_score": score})
                logger.info(f"Webhook sent to {webhook_url}. Response: {response.status_code} - {response.text}")
            
            return {
                "status": "success",
                "drift_score": score,
                "report_path": html_report_path
            }
        except Exception as e:
            logger.error(f"Error during drift detection: {str(e)}", exc_info=True)
            return {"status": "error", "message": str(e)}

    @app.get("/drift-report")
    def get_drift_report():
        """
        Serve the drift report HTML file.
        """
        html_report_path = "/tmp/data_drift_report.html"
        if os.path.exists(html_report_path):
            return FileResponse(html_report_path, media_type="text/html")
        return {"status": "error", "message": "Report not found"}

    @app.get("/metrics")
    def metrics():
        """
        Expose metrics for Prometheus scraping.
        """
        return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)

    # Health check
    @app.get("/health")
    def health_check():
        return {"status": "healthy"}

    # Endpoint to reload the model
    @app.post("/reload-model")
    def reload_model():
        try:
            app.state.model = load_model_from_minio()
            return {"status": "success", "message": "Model reloaded successfully"}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    return app
