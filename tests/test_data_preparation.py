import pandas as pd
import sys
import os

# Add project root to PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

from scripts.data_preparation import preprocess_data

def test_preprocess_data():
    # Mock raw data with enough samples per class
    raw_data = pd.DataFrame({
        "tenure": [1, 24, 60, 12, 36, 48, 5, 18, 72, 6],
        "MonthlyCharges": [29.85, 56.95, 53.85, 70.00, 85.50, 99.99, 45.50, 60.00, 120.00, 39.99],
        "TotalCharges": ["29.85", "1889.5", "3050.5", "840.0", "3000.0", "4500.0", "220.5", "730.5", "8700.5", "300.0"],
        "Churn": ["No", "Yes", "No", "Yes", "No", "Yes", "No", "Yes", "No", "Yes"],
        "gender": ["Male", "Female", "Female", "Male", "Female", "Male", "Female", "Male", "Female", "Male"],
        "Partner": ["Yes", "No", "Yes", "No", "Yes", "No", "No", "Yes", "No", "Yes"],
        "Dependents": ["No", "Yes", "No", "Yes", "No", "Yes", "Yes", "No", "No", "Yes"],
        "PhoneService": ["Yes", "No", "Yes", "Yes", "No", "Yes", "Yes", "No", "Yes", "No"],
        "InternetService": ["DSL", "Fiber optic", "DSL", "No", "Fiber optic", "DSL", "DSL", "No", "Fiber optic", "No"],
        "Contract": ["Month-to-month", "One year", "Two year", "Month-to-month", "One year", "Two year", "Month-to-month", "One year", "Two year", "One year"],
        "PaymentMethod": ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Electronic check", "Mailed check", "Bank transfer (automatic)", "Mailed check", "Electronic check", "Credit card (automatic)", "Bank transfer (automatic)"]
    })

    # Mock config file path
    config_path = "config/process.yaml"

    # Save raw data to a temporary CSV
    raw_data_path = "data/raw/test_raw_data.csv"
    raw_data.to_csv(raw_data_path, index=False)

    # Preprocess the data
    processed = preprocess_data(raw_data_path, config_path, output_dir="data/test_processed")

    # Assertions
    assert "tenure" in processed["X_train"].columns
    assert "Churn" not in processed["X_train"].columns
    assert processed["y_train"].isnull().sum() == 0
    assert processed["X_train"].isnull().sum().sum() == 0

    # Check saved files
    assert os.path.exists("data/test_processed/X_train.csv")
    assert os.path.exists("data/test_processed/y_train.csv")
