import os
import joblib
import boto3
from botocore.exceptions import NoCredentialsError

# Path to the saved model
BEST_MODEL_PATH = os.getenv("BEST_MODEL_PATH", "models/best_model/model.pkl")

def download_model_from_minio():
    """
    Download the model from MinIO if it doesn't exist locally.
    """
    minio_url = os.getenv("MINIO_URL", "http://minio-service.minio.svc.cluster.local:9000")
    minio_access_key = os.getenv("AWS_ACCESS_KEY_ID", "minioadmin")
    minio_secret_key = os.getenv("AWS_SECRET_ACCESS_KEY", "minioadmin")
    bucket_name = "models"
    model_key = "best_model/model.pkl"

    s3_client = boto3.client(
        "s3",
        endpoint_url=minio_url,
        aws_access_key_id=minio_access_key,
        aws_secret_access_key=minio_secret_key,
    )

    try:
        os.makedirs(os.path.dirname(BEST_MODEL_PATH), exist_ok=True)
        s3_client.download_file(bucket_name, model_key, BEST_MODEL_PATH)
        print(f"Model downloaded successfully from MinIO to {BEST_MODEL_PATH}")
    except NoCredentialsError:
        raise RuntimeError("Invalid MinIO credentials")
    except Exception as e:
        raise RuntimeError(f"Failed to download model from MinIO: {e}")

def load_model():
    """
    Load the best model from the saved path.

    Returns:
        Loaded model object.
    """
    if not os.path.exists(BEST_MODEL_PATH):
        print(f"Model not found locally at {BEST_MODEL_PATH}. Attempting to download from MinIO.")
        download_model_from_minio()

    try:
        model = joblib.load(BEST_MODEL_PATH)
        print(f"Model loaded successfully from {BEST_MODEL_PATH}")
        return model
    except Exception as e:
        raise RuntimeError(f"Failed to load the model. Reason: {e}")
