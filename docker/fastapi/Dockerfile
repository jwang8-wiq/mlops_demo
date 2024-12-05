# Use an official Python image as the base
FROM python:3.9-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    libssl-dev \
    && apt-get clean

# Set the working directory
WORKDIR /app

# Install required Python libraries
COPY requirements/requirements.app.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy the application code
COPY application /app/application
# COPY models/best_model /app/models/best_model

# Expose the port that FastAPI will run on
EXPOSE 8000

# Default command to run the FastAPI application
CMD ["uvicorn", "application.src.create_app:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
