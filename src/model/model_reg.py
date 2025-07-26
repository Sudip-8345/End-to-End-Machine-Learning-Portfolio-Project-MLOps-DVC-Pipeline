import json
import mlflow
from mlflow.tracking import MlflowClient

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
print("✅ Skipping model registry as DagsHub does not support it.")
