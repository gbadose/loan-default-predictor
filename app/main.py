import mlflow.pyfunc
import pandas as pd
import time
from fastapi import FastAPI, HTTPException
from mlflow.tracking import MlflowClient
from fastapi import FastAPI
from pydantic import BaseModel
from mlflow.exceptions import RestException


app = FastAPI()

# mlflow.set_tracking_uri("http://127.0.0.1:5000") # when running locally
mlflow.set_tracking_uri("http://mlflow:5000")

model = None

for i in range(10):
    try:
        model = mlflow.pyfunc.load_model("models:/LoanDefaultModel/1")
        break
    except RestException:
        print(f"[INFO] Model not ready yet, retrying ({i + 1}/10)...")
        time.sleep(3)

if not model:
    raise RuntimeError(
        "Model LoanDefaultModel version 1 not found after retries")


class LoanInput(BaseModel):
    loan_amount: float
    term: int
    employment_length: int
    annual_income: float
    credit_score: int
    dti: float
    purpose: int  # encoded category


@app.post("/predict")
def predict(input: LoanInput):
    try:

        income_to_loan_ratio = input.annual_income / input.loan_amount

        features = pd.DataFrame([{
            "loan_amount": int(input.loan_amount),
            "term": int(input.term),
            "employment_length": int(input.employment_length),
            "annual_income": int(input.annual_income),
            "credit_score": int(input.credit_score),
            "dti": float(input.dti),
            "purpose": int(input.purpose),
            "income_to_loan_ratio": float(income_to_loan_ratio)
        }])

        prediction = model.predict(features)
        return {"prediction": int(prediction)}
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Prediction failed: {str(e)}")
