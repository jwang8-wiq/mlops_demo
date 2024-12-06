# model_training.py
import os
import joblib
import pandas as pd
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import boto3
from botocore.exceptions import NoCredentialsError

def upload_reference_data_to_minio(data_dir, bucket_name, object_name, save_to_models_bucket=False):
    """
    Upload reference data to MinIO for drift detection.
    """
    minio_url = os.getenv("MINIO_URL", "http://localhost:9000")
    minio_access_key = os.getenv("AWS_ACCESS_KEY_ID", "minioadmin")
    minio_secret_key = os.getenv("AWS_SECRET_ACCESS_KEY", "minioadmin")

    # Use the 'models' bucket if specified
    if save_to_models_bucket:
        bucket_name = "models"
        reference_data_path = os.path.join("models", object_name)  # Use the models path
    else:
        reference_data_path = os.path.join(data_dir, object_name)  # Use data_dir path

    s3_client = boto3.client(
        "s3",
        endpoint_url=minio_url,
        aws_access_key_id=minio_access_key,
        aws_secret_access_key=minio_secret_key,
    )

    reference_data_path = os.path.join(data_dir, "reference_data.csv")
    try:
        # Ensure bucket exists
        try:
            s3_client.head_bucket(Bucket=bucket_name)
        except s3_client.exceptions.ClientError:
            print(f"Bucket {bucket_name} does not exist. Creating it...")
            s3_client.create_bucket(Bucket=bucket_name)

        # Upload the reference data
        print(f"Uploading {reference_data_path} to MinIO bucket {bucket_name} with key {object_name}...")
        s3_client.upload_file(reference_data_path, bucket_name, object_name)
        print("Reference data upload successful!")

        return f"s3://{bucket_name}/{object_name}"
    except Exception as e:
        raise RuntimeError(f"Failed to upload reference data to MinIO: {e}")


def upload_model_to_minio(model_dir, bucket_name, model_name):
    """
    Uploads the best model to MinIO.
    """
    minio_url = os.getenv("MINIO_URL", "http://localhost:9000")
    minio_access_key = os.getenv("AWS_ACCESS_KEY_ID", "minioadmin")
    minio_secret_key = os.getenv("AWS_SECRET_ACCESS_KEY", "minioadmin")

    s3_client = boto3.client(
        "s3",
        endpoint_url=minio_url,
        aws_access_key_id=minio_access_key,
        aws_secret_access_key=minio_secret_key,
    )

    try:
        model_path = os.path.join(model_dir, "model.pkl")
        s3_key = f"{model_name}/model.pkl"

        # Ensure bucket exists
        try:
            s3_client.head_bucket(Bucket=bucket_name)
        except NoCredentialsError:
            raise RuntimeError("Invalid MinIO credentials")
        except s3_client.exceptions.ClientError:
            print(f"Bucket {bucket_name} does not exist. Creating it...")
            s3_client.create_bucket(Bucket=bucket_name)

        # Upload the model
        print(f"Uploading {model_path} to MinIO bucket {bucket_name} with key {s3_key}...")
        s3_client.upload_file(model_path, bucket_name, s3_key)
        print("Upload successful!")

        return f"s3://{bucket_name}/{s3_key}"
    except Exception as e:
        raise RuntimeError(f"Failed to upload model to MinIO: {e}")



def train_and_save_model(data_dir, best_model_dir):
    # Paths for processed datasets
    X_train_path = os.path.join(data_dir, "X_train.csv")
    y_train_path = os.path.join(data_dir, "y_train.csv")
    X_val_path = os.path.join(data_dir, "X_val.csv")
    y_val_path = os.path.join(data_dir, "y_val.csv")

    # Load datasets
    X_train = pd.read_csv(X_train_path)
    y_train = pd.read_csv(y_train_path)
    X_val = pd.read_csv(X_val_path)
    y_val = pd.read_csv(y_val_path)

    # Preprocessing configuration
    numeric_features = ["tenure", "MonthlyCharges", "TotalCharges"]
    categorical_features = [
        "gender", "Partner", "Dependents", "PhoneService", 
        "InternetService", "Contract", "PaymentMethod", "Tenure_Bin"
    ]
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown="ignore")

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    # Configure MLflow
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:8080"))
    mlflow.set_experiment("Churn_Prediction")

    # Define models
    models = {
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
        "XGBoost": XGBClassifier(n_estimators=100, random_state=42, eval_metric="logloss"),
    }

    best_model = None
    best_auc = 0

    for model_name, model in models.items():
        with mlflow.start_run(run_name=model_name):
            clf_pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", model)])
            clf_pipeline.fit(X_train, y_train.values.ravel())
            y_val_proba = clf_pipeline.predict_proba(X_val)[:, 1]
            auc = roc_auc_score(y_val, y_val_proba)

            # Log parameters, metrics, and model to MLflow
            mlflow.log_param("model", model_name)
            mlflow.log_metric("auc", auc)
            signature = infer_signature(X_train, clf_pipeline.predict(X_train))
            mlflow.sklearn.log_model(
                clf_pipeline,
                artifact_path="model",
                input_example=X_train[:5],
                signature=signature,
            )

            if auc > best_auc:
                best_auc = auc
                best_model = clf_pipeline

    # Save the best model locally
    best_model_path = os.path.join(best_model_dir, "model.pkl")
    os.makedirs(best_model_dir, exist_ok=True)
    with open(best_model_path, "wb") as f:
        joblib.dump(best_model, f)

    print(f"Best Model AUC: {best_auc}")
    print(f"Best model saved to {best_model_path}")


if __name__ == "__main__":
    DATA_DIR = "data/processed"
    BEST_MODEL_DIR = "models/best_model"
    BUCKET_NAME = "models"
    MODEL_NAME = "best_model"

    # Train and save the model locally
    model_path = train_and_save_model(DATA_DIR, BEST_MODEL_DIR)

    # Upload the model to MinIO
    model_uri = upload_model_to_minio(BEST_MODEL_DIR, BUCKET_NAME, MODEL_NAME)
    print(f"Model available at: {model_uri}")

    # Upload reference data to MinIO
    REFERENCE_OBJECT_NAME = "reference_data.csv"
    reference_uri = upload_reference_data_to_minio(DATA_DIR, BUCKET_NAME, REFERENCE_OBJECT_NAME)
    print(f"Reference data available at: {reference_uri}")
    