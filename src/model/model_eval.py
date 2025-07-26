import numpy as np
import pandas as pd
import pickle
import json
import mlflow
import yaml
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from mlflow.models import infer_signature
from dagshub import dagshub_logger
import os
import dagshub

def load_data(filepath: str) -> pd.DataFrame:
    """Load data from CSV file"""
    try:
        return pd.read_csv(filepath)
    except Exception as e:
        raise Exception(f"Error loading data from {filepath}: {e}")

def prepare_data(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Prepare features and target"""
    try:
        X = data.drop(columns=['Potability'], axis=1)
        y = data['Potability']
        return X, y
    except Exception as e:
        raise Exception(f"Error preparing data: {e}")

def load_model(filepath: str):
    """Load trained model"""
    try:
        with open(filepath, "rb") as file:
            return pickle.load(file)
    except Exception as e:
        raise Exception(f"Error loading model from {filepath}: {e}")

def log_metrics_and_artifacts(model, X_test: pd.DataFrame, y_test: pd.Series, model_name: str):
    """Calculate and log metrics and artifacts"""
    try:
        params = yaml.safe_load(open("params.yaml", "r"))
        
        # Make predictions
        y_pred = model.predict(X_test)

        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred)
        }

        # Log parameters
        mlflow.log_params({
            "test_size": params["data_collection"]["test_size"],
            "n_estimators": params["model_building"]["n_estimators"]
        })

        # Log metrics
        mlflow.log_metrics(metrics)

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(5, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title(f"Confusion Matrix for {model_name}")
        cm_path = f"confusion_matrix_{model_name.replace(' ', '_')}.png"
        plt.savefig(cm_path)
        plt.close()
        mlflow.log_artifact(cm_path)

        return metrics
    except Exception as e:
        raise Exception(f"Error during evaluation: {e}")

def save_metrics(metrics: dict, metrics_path: str):
    """Save metrics to JSON file"""
    try:
        with open(metrics_path, 'w') as file:
            json.dump(metrics, file, indent=4)
    except Exception as e:
        raise Exception(f"Error saving metrics: {e}")

def main():
    try:
        # Initialize MLflow and DagsHub
        mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
        dagshub.init(
            repo_owner='Sudip-8345',
            repo_name='End-to-End-Machine-Learning-Portfolio-Project-MLOps-DVC-Pipeline',
            mlflow=True
        )

        # Set experiment (only needed once)
        mlflow.set_experiment("water-potability-prediction")

        # File paths
        test_data_path = "./data/processed/test_processed.csv"
        model_path = "models/model.pkl"
        metrics_path = "reports/metrics.json"
        model_name = "Water Potability Classifier"

        # Load and prepare data
        test_data = load_data(test_data_path)
        X_test, y_test = prepare_data(test_data)
        model = load_model(model_path)

        # Start MLflow run
        with mlflow.start_run() as run:
            # Evaluate model and log results
            metrics = log_metrics_and_artifacts(model, X_test, y_test, model_name)
            save_metrics(metrics, metrics_path)

            # Log artifacts
            mlflow.log_artifact(model_path)
            mlflow.log_artifact(metrics_path)
            mlflow.log_artifact(__file__)

            # Log model with signature
            signature = infer_signature(X_test, model.predict(X_test))
            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path="model",
                signature=signature,
                registered_model_name=model_name
            )

            # Save run info
            run_info = {
                'run_id': run.info.run_id,
                'model_name': model_name,
                'model_uri': f"runs:/{run.info.run_id}/model"
            }
            with open("reports/run_info.json", 'w') as f:
                json.dump(run_info, f, indent=4)

    except Exception as e:
        print(f"Error in model evaluation: {str(e)}")
        raise

if __name__ == "__main__":
    main()