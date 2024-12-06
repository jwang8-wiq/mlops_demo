# drift_detection.py
import pandas as pd
import requests
import os
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

# Environment variables for MinIO and Webhook
MINIO_URL = os.getenv("MINIO_URL", "http://localhost:9000")
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID", "minioadmin")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY", "minioadmin")
REFERENCE_DATA_BUCKET = "models"
REFERENCE_DATA_KEY = "reference_data.csv"
CURRENT_DATA_BUCKET = "data"
CURRENT_DATA_KEY = "app/data/processed/X_test.csv"  # In Prod swap this with data saved from your api calls
WEBHOOK_URL = os.getenv("WEBHOOK_URL", "http://argo-events-service.argo-events.svc.cluster.local:12000/drift")

def detect_drift_and_generate_report():
    try:
        # Load reference and current data
        reference_data_path = "/tmp/reference_data.csv"
        reference_data = download_from_minio(REFERENCE_DATA_BUCKET, REFERENCE_DATA_KEY, reference_data_path)
        current_data_path = "/tmp/current_data.csv"
        current_data = download_from_minio(CURRENT_DATA_BUCKET, CURRENT_DATA_KEY, current_data_path)

        # Ensure column type consistency
        for col in reference_data.columns:
            if col in current_data:
                expected_dtype = reference_data[col].dtype
                current_data[col] = current_data[col].astype(expected_dtype)

        # Validate schema consistency
        missing_columns = set(reference_data.columns) - set(current_data.columns)
        if missing_columns:
            raise ValueError(f"Missing columns in input data: {missing_columns}")

        # Perform drift detection
        report = Report(metrics=[DataDriftPreset()])
        report.run(reference_data=reference_data, current_data=current_data)

        # Save the report as HTML
        html_report_path = "/tmp/data_drift_report.html"
        report.save_html(html_report_path)
        print(f"Drift report saved to {html_report_path}")

        # Extract drift score for additional processing
        drift_metrics = report.as_dict()
        drift_score = drift_metrics["metrics"][0]["result"]["drift_share"]
        print(f"Drift Score: {drift_score}")

        return drift_score, html_report_path
    except Exception as e:
        print(f"Error during drift detection: {e}")
        raise

# Function to download data from MinIO
def download_from_minio(bucket_name, object_name, local_path):
    import boto3
    from botocore.exceptions import NoCredentialsError

    s3_client = boto3.client(
        "s3",
        endpoint_url=MINIO_URL,
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    )

    try:
        # Download the file from MinIO
        s3_client.download_file(bucket_name, object_name, local_path)
        print(f"Downloaded {object_name} from MinIO bucket {bucket_name} to {local_path}")
        return pd.read_csv(local_path)
    except NoCredentialsError:
        raise RuntimeError("Invalid MinIO credentials")
    except Exception as e:
        raise RuntimeError(f"Failed to download {object_name} from MinIO: {e}")


def detect_drift_and_notify():
    try:
        # Load reference and current data
        reference_data_path = "/tmp/reference_data.csv"
        reference_data = download_from_minio(REFERENCE_DATA_BUCKET, REFERENCE_DATA_KEY, reference_data_path)
        current_data_path = "/tmp/current_data.csv"
        current_data = download_from_minio(CURRENT_DATA_BUCKET, CURRENT_DATA_KEY, current_data_path)

        # Ensure column type consistency
        for col in reference_data.columns:
            if col in current_data:
                expected_dtype = reference_data[col].dtype
                current_data[col] = current_data[col].astype(expected_dtype)

        # Validate schema consistency
        missing_columns = set(reference_data.columns) - set(current_data.columns)
        if missing_columns:
            raise ValueError(f"Missing columns in input data: {missing_columns}")

        # Perform drift detection
        report = Report(metrics=[DataDriftPreset()])
        report.run(reference_data=reference_data, current_data=current_data)

        # Extract drift score
        drift_metrics = report.as_dict()
        drift_score = drift_metrics["metrics"][0]["result"]["drift_share"]
        print(f"Drift Score: {drift_score}")

        return drift_score
    except Exception as e:
        print(f"Error during drift detection: {e}")
        raise

