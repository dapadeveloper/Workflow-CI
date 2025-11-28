import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
import mlflow
import mlflow.sklearn
import os
import warnings
warnings.filterwarnings('ignore')

def load_data():
    """Load and inspect dataset"""
    try:
        # Try to read from current directory first
        df = pd.read_csv('wine_quality_processed.csv')
    except FileNotFoundError:
        # If not found, try from MLProject directory
        df = pd.read_csv('MLProject/wine_quality_processed.csv')
    
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print(f"Data types:\n{df.dtypes}")
    print(f"First 5 rows:\n{df.head()}")
    
    # Check for non-numeric values in target column
    if 'quality' in df.columns:
        print(f"Unique values in 'quality': {df['quality'].unique()}")
    
    return df

def preprocess_data(df):
    """Preprocess the data - handle both numeric and categorical target"""
    # Check if target is categorical
    target_col = 'quality'
    
    if df[target_col].dtype == 'object' or set(df[target_col].unique()) <= {'bad', 'good', 'excellent'}:
        print("Target is categorical, converting to numeric...")
        # Map categorical values to numeric
        quality_mapping = {'bad': 0, 'good': 1, 'excellent': 2}
        df[target_col] = df[target_col].map(quality_mapping)
        
        # Fill any NaN values (if mapping didn't cover all cases)
        if df[target_col].isna().any():
            # If mapping failed, use label encoding
            le = LabelEncoder()
            df[target_col] = le.fit_transform(df[target_col].astype(str))
    
    # Separate features and target
    X = df.drop(target_col, axis=1)
    y = df[target_col]
    
    # Convert all feature columns to numeric, coercing errors
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors='coerce')
    
    # Fill any NaN values created during conversion
    X = X.fillna(X.mean())
    
    # Convert target to integer
    y = y.astype(int)
    
    print(f"Features shape: {X.shape}")
    print(f"Target unique values: {y.unique()}")
    print(f"Target value counts:\n{y.value_counts()}")
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    return X_train, X_test, y_train, y_test

def train_model(X_train, y_train, problem_type='classification', n_estimators=100, max_depth=None):
    """Train model based on problem type"""
    if problem_type == 'classification':
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42
        )
    else:
        model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42
        )
    
    model.fit(X_train, y_train)
    return model, problem_type

def evaluate_model(model, X_test, y_test, problem_type='classification'):
    """Evaluate model performance"""
    y_pred = model.predict(X_test)
    
    if problem_type == 'classification':
        accuracy = accuracy_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        
        print("Classification Report:")
        print(classification_report(y_test, y_pred))
        
        return mse, mae, accuracy, y_pred, 'classification'
    else:
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        return mse, mae, r2, y_pred, 'regression'

def determine_problem_type(y):
    """Determine if it's classification or regression based on target"""
    unique_values = len(y.unique())
    if unique_values <= 10:  # If 10 or fewer unique values, treat as classification
        return 'classification'
    else:
        return 'regression'

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
        
        # Determine problem type
        problem_type = determine_problem_type(y_train)
        print(f"Problem type: {problem_type}")
        
        # Log parameters
        n_estimators = 100
        max_depth = 10
        
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("problem_type", problem_type)
        mlflow.log_param("dataset", "wine_quality_processed.csv")
        
        print(f"\nTraining {problem_type} model...")
        model, problem_type = train_model(X_train, y_train, problem_type, n_estimators, max_depth)
        
        print("Evaluating model...")
        if problem_type == 'classification':
            mse, mae, accuracy, y_pred, eval_type = evaluate_model(model, X_test, y_test, problem_type)
            
            # Log metrics
            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("mse", mse)
            mlflow.log_metric("mae", mae)
            
            print(f"\n=== Classification Results ===")
            print(f"Accuracy: {accuracy:.4f}")
            print(f"MSE: {mse:.4f}")
            print(f"MAE: {mae:.4f}")
            
        else:
            mse, mae, r2, y_pred, eval_type = evaluate_model(model, X_test, y_test, problem_type)
            
            # Log metrics
            mlflow.log_metric("mse", mse)
            mlflow.log_metric("mae", mae)
            mlflow.log_metric("r2", r2)
            
            print(f"\n=== Regression Results ===")
            print(f"MSE: {mse:.4f}")
            print(f"MAE: {mae:.4f}")
            print(f"R2 Score: {r2:.4f}")
        
        # Log model
        if problem_type == 'classification':
            mlflow.sklearn.log_model(model, "random_forest_classifier")
        else:
            mlflow.sklearn.log_model(model, "random_forest_regressor")
        
        # Log feature importance
        importance_df = pd.DataFrame({
            'feature': X_train.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        importance_df.to_csv("feature_importance.csv", index=False)
        mlflow.log_artifact("feature_importance.csv")
        
        print("\n=== Model Training Completed Successfully ===")
        print("Model and artifacts logged to MLflow")

if __name__ == "__main__":
    main()