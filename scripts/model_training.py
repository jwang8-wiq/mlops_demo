# # model_training.py
# import pandas as pd
# import mlflow
# import mlflow.sklearn
# from mlflow.models.signature import infer_signature
# from sklearn.ensemble import RandomForestClassifier
# from xgboost import XGBClassifier
# from sklearn.pipeline import Pipeline
# from sklearn.metrics import roc_auc_score
# from sklearn.preprocessing import StandardScaler, OneHotEncoder
# from sklearn.compose import ColumnTransformer
# import os

# # Paths for processed datasets
# DATA_DIR = "data/processed"
# X_train_path = os.path.join(DATA_DIR, "X_train.csv")
# y_train_path = os.path.join(DATA_DIR, "y_train.csv")
# X_val_path = os.path.join(DATA_DIR, "X_val.csv")
# y_val_path = os.path.join(DATA_DIR, "y_val.csv")
# X_test_path = os.path.join(DATA_DIR, "X_test.csv")
# y_test_path = os.path.join(DATA_DIR, "y_test.csv")


# # Paths for saving the best model
# BEST_MODEL_DIR = "models/best_model"
# os.makedirs(BEST_MODEL_DIR, exist_ok=True)

# # Load datasets
# X_train = pd.read_csv(X_train_path)
# y_train = pd.read_csv(y_train_path)
# X_val = pd.read_csv(X_val_path)
# y_val = pd.read_csv(y_val_path)
# X_test = pd.read_csv(X_test_path)
# y_test = pd.read_csv(y_test_path)

# # Preprocessing configuration
# numeric_features = ["tenure", "MonthlyCharges", "TotalCharges"]
# categorical_features = ["gender", "Partner", "Dependents", "PhoneService", "InternetService", "Contract", "PaymentMethod", "Tenure_Bin"]

# numeric_transformer = StandardScaler()
# categorical_transformer = OneHotEncoder(handle_unknown="ignore")

# preprocessor = ColumnTransformer(
#     transformers=[
#         ("num", numeric_transformer, numeric_features),
#         ("cat", categorical_transformer, categorical_features),
#     ]
# )

# # Configure MLflow
# MLFLOW_TRACKING_URI = "http://127.0.0.1:8080" 
# mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
# mlflow.set_experiment("Churn_Prediction")

# # Define models
# models = {
#     "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
#     "XGBoost": XGBClassifier(n_estimators=100, random_state=42, eval_metric="logloss"),
# }

# best_model = None
# best_auc = 0

# # Train and evaluate models
# for model_name, model in models.items():
#     with mlflow.start_run(run_name=model_name):
#         # Create pipeline
#         clf_pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", model)])
        
#         # Train the model
#         clf_pipeline.fit(X_train, y_train.values.ravel())
        
#         # Validate the model
#         y_val_proba = clf_pipeline.predict_proba(X_val)[:, 1]
#         auc = roc_auc_score(y_val, y_val_proba)
        
#         # Log parameters, metrics, and model to MLflow
#         mlflow.log_param("model", model_name)
#         mlflow.log_metric("auc", auc)

#         # Infer the signature of the model
#         signature = infer_signature(X_train, clf_pipeline.predict(X_train))
        
#         mlflow.sklearn.log_model(
#             clf_pipeline, 
#             artifact_path="model", 
#             input_example=X_train[:5], 
#             signature=signature  
#         )

#         print(f"{model_name} AUC: {auc}")

#         # Update the best model
#         if auc > best_auc:
#             best_auc = auc
#             best_model = clf_pipeline


# # Save the best model locally to `models/best_model/`
# best_model_path = os.path.join(BEST_MODEL_DIR, "model.pkl")
# with open(best_model_path, "wb") as f:
#     import joblib
#     joblib.dump(best_model, f)

# print(f"Best Model AUC: {best_auc}")
# print(f"Best model saved to {best_model_path}")


import pandas as pd
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import os
import joblib


def train_and_save_model(data_dir, best_model_dir):
    # Paths for processed datasets
    X_train_path = os.path.join(data_dir, "X_train.csv")
    y_train_path = os.path.join(data_dir, "y_train.csv")
    X_val_path = os.path.join(data_dir, "X_val.csv")
    y_val_path = os.path.join(data_dir, "y_val.csv")

    # Load datasets
    X_train = pd.read_csv(X_train_path)
    y_train = pd.read_csv(y_train_path)
    X_val = pd.read_csv(X_val_path)
    y_val = pd.read_csv(y_val_path)

    # Preprocessing configuration
    numeric_features = ["tenure", "MonthlyCharges", "TotalCharges"]
    categorical_features = [
        "gender", "Partner", "Dependents", "PhoneService", 
        "InternetService", "Contract", "PaymentMethod", "Tenure_Bin"
    ]
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown="ignore")

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    # Configure MLflow
    mlflow.set_tracking_uri("http://127.0.0.1:8080")
    mlflow.set_experiment("Churn_Prediction")

    # Define models
    models = {
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
        "XGBoost": XGBClassifier(n_estimators=100, random_state=42, eval_metric="logloss"),
    }

    best_model = None
    best_auc = 0

    for model_name, model in models.items():
        with mlflow.start_run(run_name=model_name):
            clf_pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", model)])
            clf_pipeline.fit(X_train, y_train.values.ravel())
            y_val_proba = clf_pipeline.predict_proba(X_val)[:, 1]
            auc = roc_auc_score(y_val, y_val_proba)

            # Log parameters, metrics, and model to MLflow
            mlflow.log_param("model", model_name)
            mlflow.log_metric("auc", auc)
            signature = infer_signature(X_train, clf_pipeline.predict(X_train))
            mlflow.sklearn.log_model(
                clf_pipeline,
                artifact_path="model",
                input_example=X_train[:5],
                signature=signature,
            )

            if auc > best_auc:
                best_auc = auc
                best_model = clf_pipeline

    # Save the best model locally
    best_model_path = os.path.join(best_model_dir, "model.pkl")
    os.makedirs(best_model_dir, exist_ok=True)
    with open(best_model_path, "wb") as f:
        joblib.dump(best_model, f)

    print(f"Best Model AUC: {best_auc}")
    print(f"Best model saved to {best_model_path}")


if __name__ == "__main__":
    DATA_DIR = "data/processed"
    BEST_MODEL_DIR = "models/best_model"
    train_and_save_model(DATA_DIR, BEST_MODEL_DIR)

