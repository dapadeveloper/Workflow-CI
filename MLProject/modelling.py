import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import mlflow
import mlflow.sklearn
import os

def load_data():
    try:
        df = pd.read_csv('wine_quality_processed.csv')
    except FileNotFoundError:
        df = pd.read_csv('MLProject/wine_quality_processed.csv')
    print(f"Dataset shape: {df.shape}")
    return df

def preprocess_data(df):
    X = df.drop('quality', axis=1)
    y = df['quality']
    return train_test_split(X, y, test_size=0.2, random_state=42)

def train_model(X_train, y_train, n_estimators=100, max_depth=None):
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42
    )
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    return (
        mean_squared_error(y_test, y_pred),
        mean_absolute_error(y_test, y_pred),
        r2_score(y_test, y_pred),
        y_pred
    )

def main():
    mlflow.set_tracking_uri("./mlruns")

    with mlflow.start_run():
        df = load_data()
        X_train, X_test, y_train, y_test = preprocess_data(df)

        n_estimators = 100
        max_depth = 10
        
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)

        model = train_model(X_train, y_train, n_estimators, max_depth)
        mse, mae, r2, y_pred = evaluate_model(model, X_test, y_test)

        mlflow.log_metric("mse", mse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2", r2)

        mlflow.sklearn.log_model(model, "random_forest_model")

        importance_df = pd.DataFrame({
            'feature': X_train.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        importance_df.to_csv("feature_importance.csv", index=False)
        mlflow.log_artifact("feature_importance.csv")

        print("=== Training Completed ===")
        print("MSE:", mse)
        print("MAE:", mae)
        print("R2:", r2)

if __name__ == "__main__":
    main()