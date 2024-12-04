# # data_preparation.py
# import pandas as pd
# from sklearn.model_selection import train_test_split
# import os
# import yaml

# def preprocess_data(data_path, config_path, output_dir="data/processed"):
#     # Load data
#     df = pd.read_csv(data_path)

#     # Basic cleaning
#     df.drop_duplicates(inplace=True)
#     df.dropna(inplace=True)

#     # Handle invalid entries in TotalCharges
#     df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
#     df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)

#     # Feature engineering
#     df['Tenure_Bin'] = pd.cut(df['tenure'], bins=[0, 12, 24, 36, 48, 60, 72], labels=["0-1 yr", "1-2 yrs", "2-3 yrs", "3-4 yrs", "4-5 yrs", "5-6 yrs"])

#     # Load preprocessing configuration
#     with open(config_path, "r") as file:
#         config = yaml.safe_load(file)

#     # Define feature columns and target
#     numeric_features = config["preprocessing"]["numeric_features"]
#     categorical_features = config["preprocessing"]["categorical_features"]
#     target_column = config["preprocessing"]["target_column"]

#     # Prepare features (X) and target (y)
#     X = df[numeric_features + categorical_features]
#     y = df[target_column].apply(lambda x: 1 if x == "Yes" else 0)

#     # Log preprocessing steps
#     print(f"Number of rows after cleaning: {len(df)}")
#     print(f"Columns used for training: {numeric_features + categorical_features}")
#     print("Data preparation completed successfully.")

#     # Split data into train, validation, and test sets
#     train_test_split_params = config["preprocessing"]["train_test_split"]
#     X_train, X_temp, y_train, y_temp = train_test_split(
#         X, y, 
#         test_size=train_test_split_params["test_size"], 
#         random_state=train_test_split_params["random_state"], 
#         stratify=y
#     )
#     X_val, X_test, y_val, y_test = train_test_split(
#         X_temp, y_temp, 
#         test_size=0.5, 
#         random_state=train_test_split_params["random_state"], 
#         stratify=y_temp
#     )

#     # Save preprocessed datasets
#     os.makedirs(output_dir, exist_ok=True)
#     X_train.to_csv(os.path.join(output_dir, "X_train.csv"), index=False)
#     y_train.to_csv(os.path.join(output_dir, "y_train.csv"), index=False)
#     X_val.to_csv(os.path.join(output_dir, "X_val.csv"), index=False)
#     y_val.to_csv(os.path.join(output_dir, "y_val.csv"), index=False)
#     X_test.to_csv(os.path.join(output_dir, "X_test.csv"), index=False)
#     y_test.to_csv(os.path.join(output_dir, "y_test.csv"), index=False)

#     return {
#         "X_train": X_train, "y_train": y_train,
#         "X_val": X_val, "y_val": y_val,
#         "X_test": X_test, "y_test": y_test
#     }


import pandas as pd
from sklearn.model_selection import train_test_split
import os
import yaml

def preprocess_data(data_path, config_path, output_dir="data/processed"):
    # Load data
    df = pd.read_csv(data_path)

    # Basic cleaning
    df.drop_duplicates(inplace=True)
    df.dropna(inplace=True)

    # Handle invalid entries in TotalCharges
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)

    # Feature engineering
    df['Tenure_Bin'] = pd.cut(df['tenure'], bins=[0, 12, 24, 36, 48, 60, 72], labels=["0-1 yr", "1-2 yrs", "2-3 yrs", "3-4 yrs", "4-5 yrs", "5-6 yrs"])

    # Load preprocessing configuration
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    # Define feature columns and target
    numeric_features = config["preprocessing"]["numeric_features"]
    categorical_features = config["preprocessing"]["categorical_features"]
    target_column = config["preprocessing"]["target_column"]

    # Prepare features (X) and target (y)
    X = df[numeric_features + categorical_features]
    y = df[target_column].apply(lambda x: 1 if x == "Yes" else 0)

    # Log preprocessing steps
    print(f"Number of rows after cleaning: {len(df)}")
    print(f"Columns used for training: {numeric_features + categorical_features}")
    print("Data preparation completed successfully.")

    # Split data into train, validation, and test sets
    train_test_split_params = config["preprocessing"]["train_test_split"]
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, 
        test_size=train_test_split_params["test_size"], 
        random_state=train_test_split_params["random_state"], 
        stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, 
        test_size=0.5, 
        random_state=train_test_split_params["random_state"], 
        stratify=y_temp
    )

    # Save preprocessed datasets
    os.makedirs(output_dir, exist_ok=True)
    X_train.to_csv(os.path.join(output_dir, "X_train.csv"), index=False)
    y_train.to_csv(os.path.join(output_dir, "y_train.csv"), index=False)
    X_val.to_csv(os.path.join(output_dir, "X_val.csv"), index=False)
    y_val.to_csv(os.path.join(output_dir, "y_val.csv"), index=False)
    X_test.to_csv(os.path.join(output_dir, "X_test.csv"), index=False)
    y_test.to_csv(os.path.join(output_dir, "y_test.csv"), index=False)

    return {
        "X_train": X_train, "y_train": y_train,
        "X_val": X_val, "y_val": y_val,
        "X_test": X_test, "y_test": y_test
    }
