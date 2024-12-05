# retrain_model.py
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
