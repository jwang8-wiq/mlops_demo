# evaluate_model.py

import os
import pandas as pd
import joblib
from sklearn.metrics import roc_auc_score, classification_report
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

def evaluate_model(model_path, data_dir):
    """
    Load the model and test dataset, evaluate metrics, and save the evaluation report.
    """
    # Load the model
    try:
        model = joblib.load(model_path)
        print(f"Model loaded successfully from {model_path}")
    except Exception as e:
        raise RuntimeError(f"Error loading model: {e}")

    # Load test datasets
    X_test_path = os.path.join(data_dir, "X_test.csv")
    y_test_path = os.path.join(data_dir, "y_test.csv")
    try:
        X_test = pd.read_csv(X_test_path)
        y_test = pd.read_csv(y_test_path)
    except Exception as e:
        raise RuntimeError(f"Error loading test data: {e}")

    # Evaluate the model
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_pred_proba)
    print(f"ROC AUC on test set: {auc}")

    # Generate classification report
    y_pred = (y_pred_proba > 0.5).astype(int)
    report = classification_report(y_test, y_pred)
    print("Classification Report:")
    print(report)

    # Save the classification report
    report_dir = "models/evaluation_reports"
    os.makedirs(report_dir, exist_ok=True)
    report_path = os.path.join(report_dir, "classification_report.txt")
    with open(report_path, "w") as f:
        f.write(f"ROC AUC: {auc}\n\n")
        f.write(report)

    print(f"Evaluation report saved to {report_path}")

    # Data Drift Detection
    reference_data_path = os.path.join(data_dir, "X_train.csv")
    reference_data = pd.read_csv(reference_data_path)
    drift_report = Report(metrics=[DataDriftPreset()])
    drift_report.run(reference_data=reference_data, current_data=X_test)

    # Save the drift report
    drift_report_path = os.path.join(report_dir, "data_drift_report.html")
    drift_report.save_html(drift_report_path)
    print(f"Data Drift Report saved to {drift_report_path}")

if __name__ == "__main__":
    MODEL_PATH = "models/best_model/model.pkl"
    DATA_DIR = "data/processed"
    print("Evaluating the best model...")
    evaluate_model(MODEL_PATH, DATA_DIR)


