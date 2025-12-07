import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import mlflow
import mlflow.sklearn
import warnings
warnings.filterwarnings('ignore')

def load_data():
    """Load dataset"""
    try:
        df = pd.read_csv('banknote_preprocessed.csv')
    except FileNotFoundError:
        df = pd.read_csv('../MLProject/banknote_preprocessed.csv')

    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    return df

def preprocess_data(df):
    """Split data into X and y"""
    X = df.drop("class", axis=1)
    y = df["class"]

    print(f"Features shape: {X.shape}")
    print(f"Target unique values: {y.unique()}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    return X_train, X_test, y_train, y_test

def train_model(X_train, y_train, n_estimators=100, max_depth=10):
    """Train Random Forest model"""
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42
    )
    model.fit(X_train, y_train)
    return model

def evaluate(model, X_test, y_test):
    """Evaluate model"""
    preds = model.predict(X_test)

    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds)
    rec = recall_score(y_test, preds)
    f1 = f1_score(y_test, preds)

    return acc, prec, rec, f1, preds

def main():
    mlflow.set_tracking_uri("./mlruns")

    with mlflow.start_run():
        print("=== BANKNOTE AUTH MODEL TRAINING ===")

        print("Loading dataset...")
        df = load_data()

        print("Preprocessing...")
        X_train, X_test, y_train, y_test = preprocess_data(df)

        # log parameters
        n_estimators = 100
        max_depth = 10

        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("dataset", "banknote_preprocessed.csv")

        print("Training model...")
        model = train_model(X_train, y_train, n_estimators, max_depth)

        print("Evaluating...")
        acc, prec, rec, f1, preds = evaluate(model, X_test, y_test)

        # log metrics
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", rec)
        mlflow.log_metric("f1_score", f1)

        # log model
        mlflow.sklearn.log_model(model, "model")

        print("\n=== Training Completed Successfully ===")
        print(f"Accuracy: {acc:.4f}")
        print(f"Precision: {prec:.4f}")
        print(f"Recall: {rec:.4f}")
        print(f"F1 Score: {f1:.4f}")

if __name__ == "__main__":
    main()