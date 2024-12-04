# import argparse
# import os
# import sys
# import pandas as pd

# # Add the project root directory to the PYTHONPATH
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

# from scripts.data_preparation import preprocess_data
# from scripts.model_training import train_and_save_model

# # Paths
# RAW_DATA_PATH = "data/raw/telco_churn.csv"
# CONFIG_PATH = "config/process.yaml"
# PROCESSED_DIR = "data/processed"
# BEST_MODEL_DIR = "models/best_model"

# def retrain():
#     # Step 1: Preprocess the data
#     print("Starting data preparation...")
#     datasets = preprocess_data(RAW_DATA_PATH, CONFIG_PATH, PROCESSED_DIR)
#     print("Data preparation completed.")

#     # Step 2: Train and save the best model
#     print("Starting model training...")
#     train_and_save_model(PROCESSED_DIR, BEST_MODEL_DIR)
#     print("Model training completed.")

# if __name__ == "__main__":
#     retrain()

import os
import sys

# Add the project root directory to the PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

from scripts.model_training import train_and_save_model

if __name__ == "__main__":
    DATA_DIR = "data/processed"
    BEST_MODEL_DIR = "models/best_model"
    print("Starting retraining process...")
    train_and_save_model(DATA_DIR, BEST_MODEL_DIR)
    print("Retraining completed successfully.")
