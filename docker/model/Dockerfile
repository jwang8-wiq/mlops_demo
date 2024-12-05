# Base image with Conda pre-installed
FROM continuumio/miniconda3

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libssl-dev \
    && apt-get clean

# Set the working directory
WORKDIR /app

# Copy the requirements file into the image
COPY requirements/requirements.retrain.txt /app/requirements.txt

# Create a new Conda environment with a compatible Python version and install dependencies
RUN conda create -n train-env python=3.9 -y && \
    conda run -n train-env pip install --no-cache-dir -r /app/requirements.txt && \
    conda clean -a

# Activate the environment
ENV PATH="/opt/conda/envs/train-env/bin:$PATH"

# Copy project files into the image
COPY scripts/ /app/scripts/
COPY config/ /app/config/
COPY data/raw/ /app/data/raw/

# Add environment variables for MLflow and MinIO
ENV MLFLOW_TRACKING_URI=http://mlflow-server.mlflow.svc.cluster.local:5000
ENV MINIO_URL=http://minio.minio.svc.cluster.local:9000

# Default command
CMD ["bash", "-c", "python /app/scripts/run_data_preparation.py --data-path /app/data/raw/telco_churn.csv --config-path /app/config/process.yaml --output-dir /app/data/processed && python /app/scripts/model_training.py && python /app/scripts/evaluate_model.py"]
