import pandas as pd
import pickle
import yaml
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import mlflow
import os

# DAGsHub MLflow Setup
os.environ["MLFLOW_TRACKING_URI"] = "https://dagshub.com/YoussefMagdy111/endtoendml.mlflow"
os.environ["MLFLOW_TRACKING_USERNAME"] = "YoussefMagdy111"
os.environ["MLFLOW_TRACKING_PASSWORD"] = "e3df0cde4d704f60a5621e396fb95fb6454b4ef9"

mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
mlflow.set_experiment("sales-revenue-prediction-evaluation")

def evaluate_model(model_path, test_data_path):
    
    data = pd.read_csv(test_data_path)
    
    
    params = yaml.safe_load(open("params.yaml"))
    target = params["train"]["target"]
    
    
    X_test = data.drop([target], axis=1)
    y_test = data[target]

    
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    
    y_pred = model.predict(X_test)

    
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    
    
    with mlflow.start_run():
        mlflow.log_metric("R2_Evaluation", r2)
        mlflow.log_metric("MAE_Evaluation", mae)
        mlflow.log_metric("MSE_Evaluation", mse)

    
    print(f"Evaluation Metrics:")
    print(f"R² Score: {r2:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"MSE: {mse:.4f}")

if __name__ == "__main__":
    evaluate_model(
        model_path="artifacts/model.pkl",
        test_data_path="data/processed/realistic_linear_regression_dataset.csv")
