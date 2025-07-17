import pandas as pd
import os
import pickle
import yaml
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from mlflow.models import infer_signature
import mlflow

# DAGsHub MLflow Setup
os.environ["MLFLOW_TRACKING_URI"] = "https://dagshub.com/YoussefMagdy111/endtoendml.mlflow"
os.environ["MLFLOW_TRACKING_USERNAME"] = "YoussefMagdy111"
os.environ["MLFLOW_TRACKING_PASSWORD"] = "e3df0cde4d704f60a5621e396fb95fb6454b4ef9"

mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
mlflow.set_experiment("sales-revenue-prediction")

def hyperparameter_tuning(x_train, y_train, pipeline, param_grid):
    """Perform hyperparameter tuning using GridSearchCV."""
    grid_search = GridSearchCV(estimator=pipeline, param_grid=param_grid, cv=3, verbose=1)
    grid_search.fit(x_train, y_train)
    return grid_search

def model_train(data_path, model_path):
    # Load data
    data = pd.read_csv(data_path)

    # Load params
    params = yaml.safe_load(open("params.yaml"))
    target = params["train"]["target"]

    # Features and target
    x = data.drop([target], axis=1)
    y = data[target]

    # Define preprocessing
    categorical_cols = ['ProductCategory', 'Region', 'CustomerSegment', 'IsPromotionApplied']
    numeric_cols = ['ProductionCost', 'MarketingSpend', 'SeasonalDemandIndex',
                    'CompetitorPrice', 'CustomerRating', 'EconomicIndex', 'StoreCount']

    numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])
    categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore'))])

    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_transformer, numeric_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

    # Define pipeline
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', LinearRegression())
    ])

    # Split data
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=42)
    signature = infer_signature(x_train, y_train)

    # Define param grid for pipeline
    param_grid = {
        'regressor__fit_intercept': [True, False],
        'regressor__copy_X': [True, False],
        'regressor__positive': [True, False]
    }

    # Start MLflow run
    with mlflow.start_run() as run:
        mlflow.sklearn.autolog()

        # Hyperparameter tuning
        grid_search = hyperparameter_tuning(x_train, y_train, pipeline, param_grid)
        best_model = grid_search.best_estimator_

        # Predictions
        y_predict = best_model.predict(x_test)

        # Metrics
        r2 = r2_score(y_test, y_predict)
        mae = mean_absolute_error(y_test, y_predict)

        # Log metrics
        mlflow.log_metric("R2", r2)
        mlflow.log_metric("MAE", mae)

        
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        with open(model_path, "wb") as f:
            pickle.dump(best_model, f)

        
        mlflow.log_artifact(model_path, artifact_path="model")

        print(f" R2 Score: {r2}")
        print(f" MAE: {mae}")
        

if __name__ == "__main__":
    model_train(data_path="data/processed/realistic_linear_regression_dataset.csv", model_path="artifacts/model.pkl")
