import os
import pickle
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from dagshub import dagshub_logger
import json

def validate_environment():
    """Check required environment variables"""
    required_vars = ['MLFLOW_TRACKING_URI', 'DAGSHUB_TOKEN']
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    if missing_vars:
        raise ValueError(f"Missing environment variables: {', '.join(missing_vars)}")

def load_data() -> tuple[pd.DataFrame, pd.Series]:
    """Load and validate test data"""
    data_path = Path("data/processed/test_processed.csv")
    if not data_path.exists():
        raise FileNotFoundError(f"Test data not found at {data_path.absolute()}")
    
    test_df = pd.read_csv(data_path)
    return test_df.drop(columns=['Potability']), test_df['Potability']

def load_model():
    """Load and validate trained model"""
    model_path = Path("models/model.pkl")
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path.absolute()}")
    
    with open(model_path, "rb") as f:
        return pickle.load(f)

def main():
    try:
        # 1. Environment validation
        validate_environment()
        
        # 2. Load data and model
        X_test, y_test = load_data()
        model = load_model()
        
        # 3. Calculate metrics
        y_pred = model.predict(X_test)
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred)
        }
        
        # 4. Generate confusion matrix
        cm_path = Path("reports/confusion_matrix.png")
        cm_path.parent.mkdir(exist_ok=True)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d')
        plt.title("Confusion Matrix")
        plt.savefig(cm_path)
        plt.close()
        
        # 5. DagsHub logging (simplified for compatibility)
        with dagshub_logger() as logger:
            logger.log_metrics(metrics)
            logger.log_hyperparams({
                'model_type': 'RandomForest',
                'data_version': '1.0'
            })
            logger.log_artifact(cm_path)
        
        # 6. Save local reports
        run_info = {
            'metrics': metrics,
            'confusion_matrix': str(cm_path.absolute())
        }
        with open("reports/run_info.json", "w") as f:
            json.dump(run_info, f, indent=2)
        
        print("✅ Evaluation completed successfully!")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"View at: {os.getenv('MLFLOW_TRACKING_URI')}")
        
    except Exception as e:
        print(f"❌ Evaluation failed: {str(e)}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()