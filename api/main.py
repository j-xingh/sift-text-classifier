from pathlib import Path

import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from src.data.preprocessing import clean_text


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "raw" / "SMSSpamCollection"


# ============================================================
# APP
# ============================================================

app = FastAPI(
    title="SIFT API",
    description="Statistical Intelligence for Text Filtering",
    version="1.0.0",
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "https://frontend-six-lilac-69.vercel.app",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# LOAD DATASET
# ============================================================

df = pd.read_csv(
    DATA_PATH,
    sep="\t",
    header=None,
    names=["label", "message"],
)


# ============================================================
# CLEAN TEXT
# ============================================================

df["clean_message"] = df["message"].apply(clean_text)


# ============================================================
# PREPARE FEATURES AND LABELS
# ============================================================

X_text = df["clean_message"]

y = df["label"].map({
    "ham": 0,
    "spam": 1,
})


# ============================================================
# TF-IDF
# ============================================================

vectorizer = TfidfVectorizer(
    lowercase=False,
    token_pattern=r"(?u)\b\w+\b|[^\w\s]",
)


X = vectorizer.fit_transform(X_text)


# ============================================================
# LOGISTIC REGRESSION
# ============================================================

model = LogisticRegression(
    max_iter=1000,
    random_state=42,
)

model.fit(X, y)


# ============================================================
# REQUEST SCHEMA
# ============================================================

class MessageRequest(BaseModel):
    message: str


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/")
def root():
    return {
        "status": "online",
        "service": "SIFT API",
        "version": "1.0.0",
    }


# ============================================================
# PREDICTION
# ============================================================

@app.post("/predict")
def predict(request: MessageRequest):

    # --------------------------------------------------------
    # 1. Validate input
    # --------------------------------------------------------

    message = request.message.strip()

    if not message:
        return {
            "prediction": "unknown",
            "spam_probability": 0.0,
            "message": "Please provide a message.",
        }


    # --------------------------------------------------------
    # 2. Clean message
    # --------------------------------------------------------

    cleaned_message = clean_text(message)


    # --------------------------------------------------------
    # 3. Convert message to TF-IDF
    # --------------------------------------------------------

    features = vectorizer.transform(
        [cleaned_message]
    )


    # --------------------------------------------------------
    # 4. Predict probability
    # --------------------------------------------------------

    spam_probability = float(
        model.predict_proba(features)[0][1]
    )


    # --------------------------------------------------------
    # 5. Classify
    # --------------------------------------------------------

    prediction = (
        "spam"
        if spam_probability >= 0.5
        else "ham"
    )


    # --------------------------------------------------------
    # 6. Return result
    # --------------------------------------------------------

    return {
        "prediction": prediction,
        "spam_probability": spam_probability,
    }