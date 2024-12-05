# run_data_preparation.py
import argparse
import os
import sys

# Add the project root directory to the PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

from scripts.data_preparation import preprocess_data

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Run data preparation.")
    parser.add_argument(
        "--data-path", 
        type=str, 
        required=True, 
        help="Path to the raw data CSV file."
    )
    parser.add_argument(
        "--config-path", 
        type=str, 
        default="config/process.yaml", 
        help="Path to the YAML configuration file."
    )
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default="data/processed", 
        help="Directory to save processed data."
    )
    args = parser.parse_args()

    # Run preprocessing
    datasets = preprocess_data(args.data_path, args.config_path, args.output_dir)

    # Log results
    print("Data preparation completed successfully.")
    print(f"Processed data saved in: {args.output_dir}")
    print(f"Training set shape: {datasets['X_train'].shape}")
    print(f"Validation set shape: {datasets['X_val'].shape}")
    print(f"Test set shape: {datasets['X_test'].shape}")

if __name__ == "__main__":
    main()

