import os
import sys
import pickle
import json
import yaml
import mlflow
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from mlflow.models import infer_signature
from dagshub.auth import add_app_token
import dagshub
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def initialize_environment():
    """Initialize MLflow and DagsHub with proper authentication"""
    try:
        # Verify required environment variables
        required_vars = ['MLFLOW_TRACKING_URI', 'DAGSHUB_TOKEN']
        missing_vars = [var for var in required_vars if not os.getenv(var)]
        if missing_vars:
            raise ValueError(f"Missing environment variables: {', '.join(missing_vars)}")
        
        # Initialize connections
        add_app_token(os.getenv("DAGSHUB_TOKEN"))
        dagshub.init(
            repo_owner='Sudip-8345',
            repo_name='End-to-End-Machine-Learning-Portfolio-Project-MLOps-DVC-Pipeline',
            mlflow=True
        )
        mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
        mlflow.set_experiment("water-potability-final")
        
    except Exception as e:
        print(f"❌ Initialization failed: {str(e)}", file=sys.stderr)
        sys.exit(1)

def load_and_validate_data() -> tuple[pd.DataFrame, pd.Series]:
    """Load and validate test data"""
    data_path = Path("data/processed/test_processed.csv")
    if not data_path.exists():
        raise FileNotFoundError(f"Test data not found at {data_path.absolute()}")
    
    test_df = pd.read_csv(data_path)
    return test_df.drop(columns=['Potability']), test_df['Potability']

def load_and_validate_model():
    """Load and validate trained model"""
    model_path = Path("models/model.pkl")
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path.absolute()}")
    
    with open(model_path, "rb") as f:
        return pickle.load(f)

def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """Calculate evaluation metrics and generate artifacts"""
    y_pred = model.predict(X_test)
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred)
    }
    
    # Generate confusion matrix
    cm_path = Path("reports/confusion_matrix.png")
    cm_path.parent.mkdir(exist_ok=True)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d')
    plt.title("Confusion Matrix")
    plt.savefig(cm_path)
    plt.close()
    
    return metrics, cm_path

def main():
    initialize_environment()
    
    try:
        # Load data and model
        X_test, y_test = load_and_validate_data()
        model = load_and_validate_model()
        
        # Start MLflow run
        with mlflow.start_run():
            # Evaluate model
            metrics, cm_path = evaluate_model(model, X_test, y_test)
            
            # Log parameters
            params = yaml.safe_load(open("params.yaml"))
            mlflow.log_params({
                "test_size": params["data_collection"]["test_size"],
                "n_estimators": params["model_building"]["n_estimators"]
            })
            
            # Log metrics
            mlflow.log_metrics(metrics)
            
            # Log model with signature
            signature = infer_signature(X_test, model.predict(X_test))
            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path="model",
                signature=signature,
                registered_model_name="water_potability_model"
            )
            
            # Log artifacts
            mlflow.log_artifact(cm_path)
            
            # Save run info
            run_info = {
                "metrics": metrics,
                "model_uri": mlflow.get_artifact_uri("model")
            }
            with open("reports/run_info.json", "w") as f:
                json.dump(run_info, f, indent=2)
            
            print("✅ Evaluation completed successfully!")
            print(f"Model URI: {run_info['model_uri']}")
            print(f"Accuracy: {metrics['accuracy']:.4f}")
            
    except Exception as e:
        print(f"❌ Evaluation failed: {str(e)}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()