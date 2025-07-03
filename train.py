from mlflow.models.signature import infer_signature
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import mlflow
import mlflow.sklearn

# Simulating loan dataset
np.random.seed(42)
n = 10000
data = pd.DataFrame({
    'loan_amount': np.random.randint(1000, 80000, size=n),
    'term': np.random.choice([12, 24, 36, 60, 72], size=n),
    'employment_length': np.random.randint(0, 10, size=n),
    'annual_income': np.random.randint(20000, 450000, size=n),
    'credit_score': np.random.randint(600, 850, size=n),
    'dti': np.random.uniform(0, 40, size=n),
    'purpose': np.random.choice([0, 1, 2], size=n)

})

# Creating outcome variable with real signal
data['loan_status'] = (
    (data['credit_score'] > 700) &
    (data['annual_income'] > 100000) &
    (data['dti'] < 25)
).astype(int)

# Add derived feature
data['income_to_loan_ratio'] = data['annual_income'] / data['loan_amount']

# Flip label for 5% of the rows to simulate noise
noise = np.random.rand(len(data)) < 0.05
data.loc[noise, 'loan_status'] = 1 - data.loc[noise, 'loan_status']


data.to_csv("data.csv")
X = data.drop("loan_status", axis=1)
y = data["loan_status"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=300, max_depth=15, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print(f"Accuracy: {acc:.2f}")

# MLflow logging
signature = infer_signature(X_train, model.predict(X_train[:5]))
input_example = X_train.iloc[:5]

# mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_tracking_uri("http://mlflow:5000")
mlflow.set_experiment("Loan Default Experiment")

with mlflow.start_run():
    # mlflow.log_params("random_state", 42)
    mlflow.log_params(model.get_params())
    mlflow.log_metric("accuracy", acc)
    mlflow.sklearn.log_model(
        model,
        artifact_path="model",
        registered_model_name="LoanDefaultModel",
        input_example=input_example,
        signature=signature
    )

with open("metrics.txt", "w") as f:
    f.write(classification_report(y_test, y_pred))
mlflow.log_artifact("metrics.txt")
