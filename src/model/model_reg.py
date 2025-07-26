import os
import json
import mlflow
from dagshub.auth import add_app_token
import dagshub

# Initialize authentication
add_app_token(os.getenv("DAGSHUB_TOKEN"))
dagshub.init(repo_owner='Sudip-8345', 
             repo_name='End-to-End-Machine-Learning-Portfolio-Project-MLOps-DVC-Pipeline',
             mlflow=True)

def main():
    # Load run info
    with open("reports/run_info.json") as f:
        run_info = json.load(f)
    
    print(f"✅ Model successfully registered at:")
    print(f"URI: {run_info['model_uri']}")
    print(f"Metrics: {json.dumps(run_info['metrics'], indent=2)}")
    print(f"\nView at: {os.getenv('MLFLOW_TRACKING_URI')}")

if __name__ == "__main__":
    main()