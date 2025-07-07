# Loan Default Predictor

This project demonstrates a simple machine learning pipeline for predicting loan defaults. It uses **MLflow** for model management, a **FastAPI** application for serving predictions, and **Docker Compose** to orchestrate the services.

## Components

- **train.py** – Generates a synthetic dataset, trains a `RandomForestClassifier`, and logs the model to MLflow.
- **app/main.py** – FastAPI service that loads the latest registered model from MLflow and exposes a `/predict` endpoint.
- **docker-compose.yaml** – Runs PostgreSQL, MLflow, the training job, and the FastAPI server in separate containers.

## Prerequisites

- [Docker](https://docs.docker.com/get-docker/)
- [Docker Compose](https://docs.docker.com/compose/)

Create an `.env` file in the project root with the following variables:

```bash
POSTGRES_USER=mlflow
POSTGRES_PASSWORD=mlflow
POSTGRES_DB=mlflow
```

These credentials are used by the PostgreSQL service for the MLflow tracking backend.

## Running the project

1. Build and start the containers:
   ```bash
   docker-compose up --build
   ```
2. The MLflow tracking UI will be available at [http://localhost:5500](http://localhost:5500).
3. The FastAPI application will be available at [http://localhost:8000](http://localhost:8000).

The training container runs once on startup and registers the model under the name `LoanDefaultModel`. The FastAPI service automatically loads this model.

## Making predictions

Send a POST request to `/predict` with the required fields:

```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
           "loan_amount": 15000,
           "term": 36,
           "employment_length": 5,
           "annual_income": 45000,
           "credit_score": 700,
           "dti": 20,
           "purpose": 0
         }'
```

The response will contain the predicted class (`0` or `1`).

## Training again

To retrain the model, run the training container:

```bash
docker-compose run --rm train
```

A new model version will be logged to MLflow.

## License

This project is provided for demonstration purposes and has no specific license.
