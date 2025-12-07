FROM python:3.10-slim

WORKDIR /app

COPY MLProject/mlruns /app/mlruns
COPY MLProject /app

RUN pip install --no-cache-dir mlflow pandas numpy scikit-learn

EXPOSE 5000

CMD ["mlflow", "models", "serve", "-m", "mlruns/0/1/artifacts/model", "-h", "0.0.0.0", "-p", "5000"]
