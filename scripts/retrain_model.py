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

    # Updated retraining condition: lower threshold and column logging for demo
    if drift_score > 0.3:  # Lowered threshold for demo purposes
        print("Significant dataset drift detected. Proceeding with retraining.")
        return True
    else:
        print("No significant drift detected. Skipping retraining.")
        return False


def train_and_save_model_with_upload(data_dir, best_model_dir, model_bucket, data_bucket):
    """
    Train and save the model, and upload to MinIO.
    """
    # Ensure required directories exist
    os.makedirs(best_model_dir, exist_ok=True)

    # Train and save the model
    train_and_save_model(data_dir, best_model_dir)

    # Upload the best model to the `models` bucket
    model_uri = upload_model_to_minio(best_model_dir, model_bucket, "best_model")
    print(f"Model uploaded to: {model_uri}")

    # Upload reference data to the `models` bucket
    reference_data_path = os.path.join(data_dir, "reference_data.csv")
    upload_to_s3(reference_data_path, model_bucket, "reference_data.csv")
    print(f"Reference data uploaded to MinIO: s3://{model_bucket}/reference_data.csv")

    # Upload processed data back to the `data` bucket for consistency
    for filename in ["X_train.csv", "X_test.csv", "y_train.csv", "y_test.csv", "X_val.csv", "y_val.csv"]:
        local_path = os.path.join(data_dir, filename)
        upload_to_s3(local_path, data_bucket, f"app/data/processed/{filename}")
        print(f"Processed data {filename} uploaded to MinIO: s3://{data_bucket}/app/data/processed/{filename}")

def ensure_local_directories():
    """
    Ensure that necessary local directories exist.
    """
    data_processed_dir = "data/processed"
    models_dir = "models/best_model"

    os.makedirs(data_processed_dir, exist_ok=True)  # Ensure data/processed exists
    os.makedirs(models_dir, exist_ok=True)         # Ensure models/best_model exists

    print(f"Directories ensured: {data_processed_dir}, {models_dir}")


def prepare_reference_data(data_dir, model_bucket):
    """
    Ensure reference_data.csv is available locally. 
    Download from MinIO if it exists; otherwise, generate it.
    """
    reference_data_path = os.path.join(data_dir, "reference_data.csv")
    
    # Check if file already exists locally
    if os.path.exists(reference_data_path):
        print(f"Reference data already exists locally at {reference_data_path}")
        return reference_data_path
    
    # Attempt to download from MinIO models bucket
    try:
        download_from_s3(model_bucket, "reference_data.csv", reference_data_path)
        print(f"Reference data downloaded from MinIO to {reference_data_path}")
    except Exception as e:
        # If download fails, generate reference_data.csv
        print(f"Failed to download reference data: {e}")
        print("Generating reference_data.csv locally as fallback...")
        # Example: Generate a placeholder CSV if needed
        pd.DataFrame({"example_column": [1, 2, 3]}).to_csv(reference_data_path, index=False)

    return reference_data_path


if __name__ == "__main__":
    # Ensure directories exist
    ensure_local_directories()

    DATA_DIR = "data/processed"
    BEST_MODEL_DIR = "models/best_model"
    MODEL_BUCKET = "models"
    DATA_BUCKET = "data"

    # Prepare reference_data.csv (download from MinIO or generate if missing)
    reference_data_path = prepare_reference_data(DATA_DIR, MODEL_BUCKET)


    # Download datasets from MinIO
    datasets = [
        ("app/data/processed/X_train.csv", "X_train.csv"),
        ("app/data/processed/X_test.csv", "X_test.csv"),
        ("app/data/processed/y_train.csv", "y_train.csv"),
        ("app/data/processed/y_test.csv", "y_test.csv"),
        ("app/data/processed/X_val.csv", "X_val.csv"),
        ("app/data/processed/y_val.csv", "y_val.csv"),
    ]
    for remote_path, local_filename in datasets:
        download_from_s3(DATA_BUCKET, remote_path, os.path.join(DATA_DIR, local_filename))

    # Assess drift and retrain if needed
    # reference_data_local = os.path.join(DATA_DIR, "X_train.csv")
    current_data_local = os.path.join(DATA_DIR, "X_test.csv")

    if assess_drift_for_retraining(reference_data_path, current_data_local):
        print("Starting retraining process...")
        train_and_save_model_with_upload(DATA_DIR, BEST_MODEL_DIR, MODEL_BUCKET, DATA_BUCKET)
        print("Retraining completed successfully.")
    else:
        print("Retraining skipped due to no significant drift.")