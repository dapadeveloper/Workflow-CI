FROM continuumio/miniconda3:latest

# Set working directory
WORKDIR /app

# Copy project files
COPY MLProject/ /app/

# Create conda environment
RUN conda env create -f conda.yaml

# Make RUN commands use the new environment
SHELL ["conda", "run", "-n", "wine-quality-model", "/bin/bash", "-c"]

# Ensure conda environment is activated
RUN echo "conda activate wine-quality-model" >> ~/.bashrc

# Install additional dependencies for MLflow
RUN pip install mlflow>=2.0

# Expose MLflow port
EXPOSE 5000

# Set entry point
ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "wine-quality-model", "python", "modelling.py"]