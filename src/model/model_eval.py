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

import os
import mlflow
import pickle
import pandas as pd
from sklearn.metrics import accuracy_score
from dagshub import dagshub_logger

# Initialize with DagsHub's recommended approach
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))

def log_to_dagshub():
    with dagshub_logger() as logger:
        # Load data
        test_df = pd.read_csv("data/processed/test_processed.csv")
        X_test = test_df.drop(columns=['Potability'])
        y_test = test_df['Potability']
        
        # Load model
        with open("models\model.pkl", "rb") as f:
            model = pickle.load(f)
        
        # Calculate metrics
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Log to DagsHub
        logger.log_metrics({"accuracy": accuracy})
        logger.log_hyperparams({"model_type": "RandomForest"})
        
        print(f"Successfully logged accuracy: {accuracy}")

if __name__ == "__main__":
    log_to_dagshub()
# Enhanced error handling
def safe_mlflow_setup():
    try:
        add_app_token(os.getenv("DAGSHUB_TOKEN"))
        dagshub.init(
            repo_owner='Sudip-8345',
            repo_name='End-to-End-Machine-Learning-Portfolio-Project-MLOps-DVC-Pipeline',
            mlflow=True
        )
        mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
    except Exception as e:
        print(f"Setup failed: {str(e)}", file=sys.stderr)
        sys.exit(1)

def main():
    safe_mlflow_setup()
    
    try:
        # 1. Load Data
        test_df = pd.read_csv("data/processed/test_processed.csv")
        X_test = test_df.drop(columns=['Potability'])
        y_test = test_df['Potability']
        
        # 2. Load Model
        with open("models/model.pkl", "rb") as f:
            model = pickle.load(f)
        
        # 3. Start MLflow Run
        with mlflow.start_run() as run:
            # 4. Evaluate
            y_pred = model.predict(X_test)
            
            metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred),
                'recall': recall_score(y_test, y_pred),
                'f1': f1_score(y_test, y_pred)
            }
            
            # 5. Log Parameters
            params = yaml.safe_load(open("params.yaml"))
            mlflow.log_params({
                "test_size": params["data_collection"]["test_size"],
                "n_estimators": params["model_building"]["n_estimators"]
            })
            
            # 6. Log Metrics
            mlflow.log_metrics(metrics)
            
            # 7. Save Confusion Matrix
            cm = confusion_matrix(y_test, y_pred)
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d')
            plt.title("Confusion Matrix")
            cm_path = "confusion_matrix.png"
            plt.savefig(cm_path)
            mlflow.log_artifact(cm_path)
            
            # 8. Log Model
            signature = infer_signature(X_test, y_pred)
            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path="model",
                signature=signature,
                registered_model_name="water_potability_model"
            )
            
            # 9. Save Run Info
            run_info = {
                "run_id": run.info.run_id,
                "metrics": metrics,
                "model_uri": f"runs:/{run.info.run_id}/model"
            }
            with open("reports/run_info.json", "w") as f:
                json.dump(run_info, f, indent=2)
                
        print("✅ Evaluation completed successfully")
        
    except Exception as e:
        print(f" Evaluation failed: {str(e)}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    log_to_dagshub()
    main()