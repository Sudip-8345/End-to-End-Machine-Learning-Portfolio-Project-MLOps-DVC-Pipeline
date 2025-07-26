import os
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

# Initialize authentication
add_app_token(os.getenv("DAGSHUB_TOKEN"))
dagshub.init(repo_owner='Sudip-8345', 
             repo_name='End-to-End-Machine-Learning-Portfolio-Project-MLOps-DVC-Pipeline',
             mlflow=True)

def load_data(filepath: str) -> pd.DataFrame:
    return pd.read_csv(filepath)

def prepare_data(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    return data.drop(columns=['Potability'], axis=1), data['Potability']

def load_model(filepath: str):
    with open(filepath, "rb") as f:
        return pickle.load(f)

def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    y_pred = model.predict(X_test)
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred)
    }
    
    # Log params from params.yaml
    params = yaml.safe_load(open("params.yaml"))
    mlflow.log_params({
        "test_size": params["data_collection"]["test_size"],
        "n_estimators": params["model_building"]["n_estimators"]
    })
    
    # Log metrics
    mlflow.log_metrics(metrics)
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d')
    plt.title("Confusion Matrix")
    plt.savefig("confusion_matrix.png")
    mlflow.log_artifact("confusion_matrix.png")
    
    return metrics

def main():
    mlflow.set_experiment("water-potability-final")
    
    with mlflow.start_run() as run:
        # Load data and model
        test_df = load_data("data/processed/test_processed.csv")
        X_test, y_test = prepare_data(test_df)
        model = load_model("models/model.pkl")
        
        # Evaluate
        metrics = evaluate_model(model, X_test, y_test)
        
        # Log model
        signature = infer_signature(X_test, model.predict(X_test))
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            signature=signature,
            registered_model_name="water_potability_model"
        )
        
        # Save run info
        run_info = {
            "run_id": run.info.run_id,
            "model_uri": f"runs:/{run.info.run_id}/model",
            "metrics": metrics
        }
        with open("reports/run_info.json", "w") as f:
            json.dump(run_info, f, indent=2)

if __name__ == "__main__":
    main()