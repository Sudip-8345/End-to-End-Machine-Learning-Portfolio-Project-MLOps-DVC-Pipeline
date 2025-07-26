import json
import os
import mlflow
import dagshub
from mlflow.tracking import MlflowClient
from dagshub import dagshub_logger

def load_run_info(run_info_path: str) -> dict:
    """Load run information from JSON file"""
    try:
        with open(run_info_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        raise Exception(f"Error loading run info: {e}")

def log_model_info(model_uri: str, model_name: str):
    """Log model information to console"""
    print(f"Model URI: {model_uri}")
    print(f"Model Name: {model_name}")
    print("Model metrics and artifacts can be viewed at:")
    print(f"{os.getenv('MLFLOW_TRACKING_URI')}")

def main():
    try:
        # Initialize MLflow tracking from environment variables
        mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
        
        # Initialize DagsHub with MLflow integration
        dagshub.init(
            repo_owner='Sudip-8345',
            repo_name='End-to-End-Machine-Learning-Portfolio-Project-MLOps-DVC-Pipeline',
            mlflow=True
        )

        # Load run information
        run_info = load_run_info("reports/run_info.json")
        run_id = run_info['run_id']
        model_name = run_info['model_name']

        # Get model URI
        model_uri = f"runs:/{run_id}/model_artifact/model.pkl"

        # Log model information
        log_model_info(model_uri, model_name)

        print("Model registration complete. Note: DagsHub uses automatic model tracking")
        print("The model is already available in your MLflow experiments")

    except Exception as e:
        print(f"An error occurred during model registration: {e}")
        raise

if __name__ == "__main__":
    main()