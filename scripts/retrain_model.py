# retrain_model.py
import os
import sys
import pandas as pd
import boto3
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
from botocore.exceptions import NoCredentialsError

# Add the project root directory to the PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

from scripts.model_training import train_and_save_model, upload_model_to_minio, upload_reference_data_to_minio

# MinIO/S3 configuration
S3_CLIENT = boto3.client(
    "s3",
    endpoint_url=os.getenv("MINIO_URL", "http://localhost:9000"),
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID", "minioadmin"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY", "minioadmin"),
)

def download_from_s3(bucket, key, local_path):
    """
    Download a file from MinIO/S3 to a local path.
    """
    # Ensure the local directory exists
    local_dir = os.path.dirname(local_path)
    os.makedirs(local_dir, exist_ok=True)

    try:
        # Download file to the specified path
        S3_CLIENT.download_file(bucket, key, local_path)
        print(f"Downloaded {key} from bucket {bucket} to {local_path}")
    except NoCredentialsError:
        raise RuntimeError("Invalid MinIO/S3 credentials")
    except Exception as e:
        raise RuntimeError(f"Failed to download {key} from bucket {bucket}: {e}")


def upload_to_s3(local_path, bucket, key):
    """
    Upload a file from a local path to MinIO/S3.
    """
    try:
        S3_CLIENT.upload_file(local_path, bucket, key)
        print(f"Uploaded {local_path} to bucket {bucket} with key {key}")
    except Exception as e:
        raise RuntimeError(f"Failed to upload {local_path} to bucket {bucket}: {e}")

def assess_drift_for_retraining(reference_data_path, current_data_path):
    """
    Assess data drift between reference and current data to determine if retraining is needed.
    """
    reference_data = pd.read_csv(reference_data_path)
    current_data = pd.read_csv(current_data_path)

    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=reference_data, current_data=current_data)

    drift_score = report.as_dict()['metrics'][0]['result']['drift_share']
    print(f"Drift Share: {drift_score}")

    # Save drift report
    drift_report_path = os.path.join("models/evaluation_reports", "retraining_drift_report.html")
    os.makedirs(os.path.dirname(drift_report_path), exist_ok=True)
    report.save_html(drift_report_path)

    if drift_score > 0.5:  # Threshold for triggering retraining
        print("Significant drift detected. Proceeding with retraining.")
        return True
    else:
        print("No significant drift detected. Skipping retraining.")
        return False

def train_and_save_model_with_upload(data_dir, best_model_dir, bucket_name="models"):
    """
    Train and save the model, and upload to MinIO.
    """
    # Ensure required directories exist
    os.makedirs(best_model_dir, exist_ok=True)

    # Call the existing train_and_save_model function
    train_and_save_model(data_dir, best_model_dir)

    # Upload the best model
    model_uri = upload_model_to_minio(best_model_dir, bucket_name, "best_model")
    print(f"Model uploaded to: {model_uri}")

    # Upload reference data again for consistency
    reference_data_path = os.path.join(data_dir, "reference_data.csv")
    upload_reference_data_to_minio(data_dir, bucket_name, "reference_data.csv")
    print(f"Reference data re-uploaded to MinIO.")

if __name__ == "__main__":
    DATA_DIR = "data/processed"
    BEST_MODEL_DIR = "models/best_model"
    BUCKET_NAME = "models"
    BUCKET_NAME_DATA = "data"

    # Download datasets from S3
    reference_data_local = os.path.join(DATA_DIR, "X_train.csv")
    current_data_local = os.path.join(DATA_DIR, "X_test.csv")
    download_from_s3(BUCKET_NAME_DATA, "app/data/processed/X_train.csv", reference_data_local)
    download_from_s3(BUCKET_NAME_DATA, "app/data/processed/X_test.csv", current_data_local)

    # Assess drift before retraining
    if assess_drift_for_retraining(reference_data_local, current_data_local):
        print("Starting retraining process...")
        train_and_save_model_with_upload(DATA_DIR, BEST_MODEL_DIR, BUCKET_NAME)
        print("Retraining completed successfully.")
    else:
        print("Retraining skipped due to no significant drift.")
