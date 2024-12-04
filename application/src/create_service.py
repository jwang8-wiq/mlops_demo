import joblib
import os

# Path to the saved model
BEST_MODEL_PATH = "models/best_model/model.pkl"

def load_model():
    """
    Load the best model from the saved path.

    Returns:
        Loaded model object.
    """
    if not os.path.exists(BEST_MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at {BEST_MODEL_PATH}")

    try:
        model = joblib.load(BEST_MODEL_PATH)
        print(f"Model loaded successfully from {BEST_MODEL_PATH}")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        raise RuntimeError(f"Failed to load the model. Reason: {str(e)}")
