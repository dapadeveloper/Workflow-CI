import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder
import mlflow
import mlflow.sklearn
import os
import warnings
warnings.filterwarnings('ignore')

def load_data():
    """Load and inspect dataset"""
    try:
        df = pd.read_csv('wine_quality_processed.csv')
    except FileNotFoundError:
        df = pd.read_csv('../MLProject/wine_quality_processed.csv')
    
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print(f"Data types:\n{df.dtypes}")
    
    return df

def preprocess_data(df):
    """Preprocess the data"""
    # Convert all columns to numeric if possible
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Fill NaN values
    df = df.fillna(df.mean())
    
    # Separate features and target
    X = df.drop('quality', axis=1)
    y = df['quality']
    
    # Convert to appropriate types
    X = X.astype(np.float32)
    y = y.astype(int)
    
    print(f"Features shape: {X.shape}")
    print(f"Target unique values: {y.unique()}")
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    return X_train, X_test, y_train, y_test

def train_model(X_train, y_train, n_estimators=100, max_depth=10):
    """Train Random Forest model"""
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42
    )
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """Evaluate model performance"""
    y_pred = model.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    return mse, mae, r2, y_pred

def main():
    # Set MLflow tracking
    mlflow.set_tracking_uri("./mlruns")
    
    # Start MLflow run
    with mlflow.start_run():
        print("=== WINE QUALITY MODEL TRAINING ===")
        print("Loading data...")
        df = load_data()
        
        print("\nPreprocessing data...")
        X_train, X_test, y_train, y_test = preprocess_data(df)
        
        # Log parameters
        n_estimators = 100
        max_depth = 10
        
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("dataset", "wine_quality_processed.csv")
        
        print(f"\nTraining model...")
        model = train_model(X_train, y_train, n_estimators, max_depth)
        
        print("Evaluating model...")
        mse, mae, r2, y_pred = evaluate_model(model, X_test, y_test)
        
        # Log metrics
        mlflow.log_metric("mse", mse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2", r2)
        
        # Log model
        mlflow.sklearn.log_model(model, "model")
        
        # Log feature importance
        importance_df = pd.DataFrame({
            'feature': X_train.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        importance_df.to_csv("feature_importance.csv", index=False)
        mlflow.log_artifact("feature_importance.csv")
        
        print("\n=== Model Training Completed Successfully ===")
        print(f"MSE: {mse:.4f}")
        print(f"MAE: {mae:.4f}")
        print(f"R2 Score: {r2:.4f}")
        print("Model and artifacts logged to MLflow")

if __name__ == "__main__":
    main()