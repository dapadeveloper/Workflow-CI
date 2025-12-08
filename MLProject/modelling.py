import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# FIX: MLflow tracking directory (wajib untuk GitHub Actions)
mlflow.set_tracking_uri("file:./mlruns")

# Set experiment
mlflow.set_experiment("Banknote_Authentication")

# Load dataset hasil preprocessing
data = pd.read_csv("namadataset_preprocessing/banknote_preprocessed.csv")

X = data.drop("class", axis=1)
y = data["class"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

with mlflow.start_run():
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42
    )

    model.fit(X_train, y_train)
    acc = model.score(X_test, y_test)

    # Log param & metrics
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("max_depth", 10)
    mlflow.log_metric("accuracy", acc)

    # Log model
    mlflow.sklearn.log_model(model, "model")

print("Model training selesai. Akurasi:", acc)