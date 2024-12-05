# Use a lightweight Python base image
FROM python:3.11-slim

# Install necessary dependencies
RUN pip install --no-cache-dir \
    mlserver mlserver-sklearn boto3 mlflow

# Create a working directory
WORKDIR /app

# Set environment variables for MLServer
ENV MODEL_NAME=best-model
ENV MODEL_URI=/app/models/best_model
ENV AWS_ACCESS_KEY_ID=EMEF3W3E95RRWuVAsImh
ENV AWS_SECRET_ACCESS_KEY=vByKT6NLdLTCbuQkvxOXmyBhmM8SuhDQXaG0qWoS

# Expose the default MLServer and MLflow ports
EXPOSE 8080 5000

# Copy entrypoint script
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Set the entrypoint
ENTRYPOINT ["/entrypoint.sh"]
