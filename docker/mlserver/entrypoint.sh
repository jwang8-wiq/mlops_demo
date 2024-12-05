#!/bin/bash

# Start MLServer in the background
mlserver start . &

# Start the MLflow UI
mlflow ui --host 0.0.0.0 --port 5000
