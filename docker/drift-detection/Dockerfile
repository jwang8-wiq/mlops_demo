# Base image with Conda pre-installed
FROM continuumio/miniconda3

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libssl-dev \
    && apt-get clean

# Set the working directory
WORKDIR /app

# Copy the requirements file for drift detection
COPY requirements/requirements.drift.txt /app/requirements.txt

# Create a new Conda environment and install dependencies
RUN conda create -n drift-env python=3.9 -y && \
    conda run -n drift-env pip install --no-cache-dir -r /app/requirements.txt && \
    conda clean -a

# Activate the environment
ENV PATH="/opt/conda/envs/drift-env/bin:$PATH"

# Copy drift detection script into the image
COPY scripts/drift_detection.py /app/scripts/

# Add environment variables for MinIO and Webhook
ENV MINIO_URL=http://minio.minio.svc.cluster.local:9000
ENV AWS_ACCESS_KEY_ID=minioadmin
ENV AWS_SECRET_ACCESS_KEY=minioadmin
ENV REFERENCE_DATA_BUCKET=models
ENV CURRENT_DATA_BUCKET=data
ENV WEBHOOK_URL=http://argo-events-service.argo-events.svc.cluster.local:12000/drift

# Default command to run drift detection
CMD ["python", "/app/scripts/drift_detection.py"]
