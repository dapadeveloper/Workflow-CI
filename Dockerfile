FROM continuumio/miniconda3:latest

# Set working directory
WORKDIR /app

# Copy project files
COPY MLProject/ /app/

# Create conda environment
RUN conda env create -f conda.yaml

# Make RUN commands use the new environment
SHELL ["conda", "run", "-n", "wine-quality-model", "/bin/bash", "-c"]

# Install additional dependencies
RUN pip install --no-cache-dir mlflow>=2.0

# Create dataset
RUN python -c "\
import pandas as pd; \
import numpy as np; \
np.random.seed(42); \
df = pd.DataFrame({ \
    'fixed_acidity': np.random.uniform(4.0, 16.0, 100), \
    'volatile_acidity': np.random.uniform(0.1, 1.6, 100), \
    'citric_acid': np.random.uniform(0.0, 1.0, 100), \
    'residual_sugar': np.random.uniform(0.5, 15.5, 100), \
    'chlorides': np.random.uniform(0.01, 0.2, 100), \
    'free_sulfur_dioxide': np.random.uniform(1, 72, 100), \
    'total_sulfur_dioxide': np.random.uniform(6, 289, 100), \
    'density': np.random.uniform(0.98, 1.04, 100), \
    'pH': np.random.uniform(2.8, 4.0, 100), \
    'sulphates': np.random.uniform(0.3, 2.0, 100), \
    'alcohol': np.random.uniform(8.0, 15.0, 100), \
    'quality': np.random.randint(3, 9, 100) \
}); \
df.to_csv('wine_quality_processed.csv', index=False); \
print('Dataset created') \
"

# Expose MLflow port
EXPOSE 5000

# Set entry point
ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "wine-quality-model", "python", "modelling.py"]