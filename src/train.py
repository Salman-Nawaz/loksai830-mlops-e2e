```python
import os
import pandas as pd
import joblib
import boto3
import mlflow

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# ============================================================
# Configuration
# ============================================================

S3_BUCKET = "salman-nawaz-mlproject"
S3_KEY = "latest/model.pkl"

LOCAL_MODEL_PATH = "models/model.pkl"

MLFLOW_DB = (
    "sqlite:////home/devopsadmin/"
    "MLops/loksai830-mlops-e2e/mlflow.db"
)


# ============================================================
# MLflow configuration
# ============================================================

mlflow.set_tracking_uri(MLFLOW_DB)

mlflow.set_experiment("Sentiment Analysis")


# ============================================================
# Load processed data
# ============================================================

df = pd.read_csv("data/processed/clean.csv")

X = df["review_text"]
y = df["sentiment"]

print("Dataset size:", len(df))


# ============================================================
# Start MLflow run
# ============================================================

with mlflow.start_run() as run:

    # --------------------------------------------------------
    # Log training parameters
    # --------------------------------------------------------

    mlflow.log_param("dataset_size", len(X))
    mlflow.log_param("test_size", 0.2)
    mlflow.log_param("random_state", 42)
    mlflow.log_param("max_iter", 300)

    # --------------------------------------------------------
    # TF-IDF vectorization
    # --------------------------------------------------------

    vectorizer = TfidfVectorizer()

    X_vec = vectorizer.fit_transform(X)

    print("TF-IDF matrix created:", X_vec.shape)

    # --------------------------------------------------------
    # Train/test split
    # --------------------------------------------------------

    Xtr, Xte, ytr, yte = train_test_split(
        X_vec,
        y,
        test_size=0.2,
        random_state=42
    )

    # --------------------------------------------------------
    # Train Logistic Regression model
    # --------------------------------------------------------

    model = LogisticRegression(max_iter=300)

    model.fit(Xtr, ytr)

    # --------------------------------------------------------
    # Evaluate model
    # --------------------------------------------------------

    preds = model.predict(Xte)

    acc = accuracy_score(yte, preds)

    print("Accuracy:", acc)

    # --------------------------------------------------------
    # Log metric to MLflow
    # --------------------------------------------------------

    mlflow.log_metric("accuracy", acc)

    # --------------------------------------------------------
    # Save model locally
    # --------------------------------------------------------

    os.makedirs("models", exist_ok=True)

    joblib.dump(
        (model, vectorizer),
        LOCAL_MODEL_PATH
    )

    print("Model saved:", LOCAL_MODEL_PATH)

    # --------------------------------------------------------
    # Log model artifact to MLflow
    # --------------------------------------------------------

    mlflow.log_artifact(LOCAL_MODEL_PATH)

    # --------------------------------------------------------
    # Display MLflow Run ID
    # --------------------------------------------------------

    print("MLflow Run ID:", run.info.run_id)


# ============================================================
# Upload model to S3
# ============================================================

s3 = boto3.client("s3")

s3.upload_file(
    LOCAL_MODEL_PATH,
    S3_BUCKET,
    S3_KEY
)

print(
    f"Model uploaded to "
    f"s3://{S3_BUCKET}/{S3_KEY}"
)

print("✅ Model trained, tracked with MLflow, and uploaded to S3")
```
