# data_preparation.py
import pandas as pd
from sklearn.model_selection import train_test_split
import os
import yaml
import boto3


def upload_file_to_minio(local_file_path, bucket_name, object_name):
    """
    Upload a local file to MinIO.
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
        # Ensure bucket exists
        try:
            s3_client.head_bucket(Bucket=bucket_name)
        except s3_client.exceptions.ClientError:
            print(f"Bucket {bucket_name} does not exist. Creating it...")
            s3_client.create_bucket(Bucket=bucket_name)

        # Upload file
        print(f"Uploading {local_file_path} to MinIO bucket {bucket_name} with key {object_name}...")
        s3_client.upload_file(local_file_path, bucket_name, object_name)
        print(f"Upload successful: {object_name}")
    except Exception as e:
        raise RuntimeError(f"Failed to upload file to MinIO: {e}")


def preprocess_data(data_path, config_path, output_dir="data/processed", bucket_name="data"):
    """
    Preprocess the raw data, save preprocessed datasets locally and upload them to MinIO.
    """
    # Load data
    df = pd.read_csv(data_path)

    # Basic cleaning
    df.drop_duplicates(inplace=True)
    df.dropna(inplace=True)

    # Handle invalid entries in TotalCharges
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)

    # Feature engineering
    df['Tenure_Bin'] = pd.cut(
        df['tenure'],
        bins=[0, 12, 24, 36, 48, 60, 72],
        labels=["0-1 yr", "1-2 yrs", "2-3 yrs", "3-4 yrs", "4-5 yrs", "5-6 yrs"]
    )

    # Load preprocessing configuration
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    # Define feature columns and target
    numeric_features = config["preprocessing"]["numeric_features"]
    categorical_features = config["preprocessing"]["categorical_features"]
    target_column = config["preprocessing"]["target_column"]

    # Prepare features (X) and target (y)
    X = df[numeric_features + categorical_features]
    y = df[target_column].apply(lambda x: 1 if x == "Yes" else 0)

    # Log preprocessing steps
    print(f"Number of rows after cleaning: {len(df)}")
    print(f"Columns used for training: {numeric_features + categorical_features}")
    print("Data preparation completed successfully.")

    # Split data into train, validation, and test sets
    train_test_split_params = config["preprocessing"]["train_test_split"]
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y,
        test_size=train_test_split_params["test_size"],
        random_state=train_test_split_params["random_state"],
        stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=0.5,
        random_state=train_test_split_params["random_state"],
        stratify=y_temp
    )

    # Save preprocessed datasets locally and upload to MinIO
    os.makedirs(output_dir, exist_ok=True)
    datasets = {
        "X_train.csv": X_train,
        "y_train.csv": y_train,
        "X_val.csv": X_val,
        "y_val.csv": y_val,
        "X_test.csv": X_test,
        "y_test.csv": y_test,
    }
    for filename, dataset in datasets.items():
        local_path = os.path.join(output_dir, filename)
        dataset.to_csv(local_path, index=False)
        upload_file_to_minio(local_path, bucket_name, f"{output_dir}/{filename}")

    # Save and upload reference data
    reference_data_path = os.path.join(output_dir, "reference_data.csv")
    X_train.to_csv(reference_data_path, index=False)
    upload_file_to_minio(reference_data_path, bucket_name, f"{output_dir}/reference_data.csv")
    print(f"Reference data uploaded successfully: {reference_data_path}")

    return {
        "X_train": X_train, "y_train": y_train,
        "X_val": X_val, "y_val": y_val,
        "X_test": X_test, "y_test": y_test
    }


