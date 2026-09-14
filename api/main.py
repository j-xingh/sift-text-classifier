import joblib
from pathlib import Path

from fastapi import FastAPI
from pydantic import BaseModel

from src.data.preprocessing import clean_text


# --------------------------------------------------
# Load trained model artifacts
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"

vectorizer = joblib.load(
    MODELS_DIR / "tfidf_vectorizer.joblib"
)

model = joblib.load(
    MODELS_DIR / "logistic_regression.joblib"
)


# --------------------------------------------------
# FastAPI application
# --------------------------------------------------

app = FastAPI(
    title="SIFT",
    description="Statistical Intelligence for Text Filtering",
    version="1.0.0"
)


# --------------------------------------------------
# Request schema
# --------------------------------------------------

class MessageRequest(BaseModel):
    message: str


# --------------------------------------------------
# Health check
# --------------------------------------------------

@app.get("/")
def root():
    return {
        "project": "SIFT",
        "status": "running"
    }


# --------------------------------------------------
# Prediction endpoint
# --------------------------------------------------

@app.post("/predict")
def predict(request: MessageRequest):

    cleaned_message = clean_text(request.message)

    features = vectorizer.transform([cleaned_message])

    probability = model.predict_proba(features)[0][1]

    prediction = "spam" if probability >= 0.5 else "ham"

    return {
        "message": request.message,
        "prediction": prediction,
        "spam_probability": round(float(probability), 4)
    }