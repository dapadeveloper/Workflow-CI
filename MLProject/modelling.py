import pandas as pd
import mlflow
import mlflow.sklearn
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

mlflow.set_tracking_uri("file:///./mlruns")
mlflow.set_experiment("Banknote_Authentication")

df = pd.read_csv("namadataset_preprocessing/banknote_preprocessed.csv")

X = df.drop("class", axis=1)
y = df["class"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

with mlflow.start_run():
    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)
    
    acc = model.score(X_test, y_test)
    mlflow.log_metric("accuracy", acc)

    mlflow.sklearn.log_model(model, "model")

    joblib.dump(model, "banknote_rf_model.pkl")
    mlflow.log_artifact("banknote_rf_model.pkl")

print(" Training selesai.")