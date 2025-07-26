import json
import mlflow
from mlflow.tracking import MlflowClient
import os
os.environ["MLFLOW_TRACKING_USERNAME"] = "Sudip-8345"
os.environ["MLFLOW_TRACKING_PASSWORD"] = "12972f00f3e9e497b33238b5832f455b866ef5d0"

# Set the experiment name in MLflow
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))

mlflow.set_experiment("final_model3")
import dagshub
dagshub.init(repo_owner='Sudip-8345', repo_name='End-to-End-Machine-Learning-Portfolio-Project-MLOps-DVC-Pipeline', mlflow=True)

mlflow.set_tracking_uri("https://dagshub.com/Sudip-8345/End-to-End-Machine-Learning-Portfolio-Project-MLOps-DVC-Pipeline.mlflow")
mlflow.set_experiment("final_model")

# Load run info
with open("reports/run_info.json", 'r') as f:
    run_info = json.load(f)

run_id = run_info['run_id']
model_name = run_info['model_name']

# You can now access or use the model artifact like this
model_uri = f"runs:/{run_id}/model_artifact/model.pkl"

print(f"Model is available at: {model_uri}")
print("Skipping model registry as DagsHub does not support it.")
