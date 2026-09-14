# SIFT — Statistical Intelligence for Text Filtering

SIFT is an end-to-end SMS spam classification system built to understand and demonstrate the complete machine learning development workflow — from raw text data and exploratory analysis to feature engineering, model training, evaluation, API development, and deployment.

The project classifies an incoming SMS message as either **Spam** or **Ham (legitimate)** using TF-IDF text features and Logistic Regression.

> **Live Demo:** https://frontend-six-lilac-69.vercel.app/

---

## Overview

Spam messages are often characterized by recurring linguistic patterns such as promotional language, unusual punctuation, financial terms, urgency, and frequently occurring keywords.

SIFT converts these textual patterns into numerical features using **TF-IDF (Term Frequency–Inverse Document Frequency)** and uses **Logistic Regression** to estimate the probability that a message is spam.

The project was intentionally developed in two stages:

1. **Machine learning implementation from scratch** using NumPy to understand the underlying mathematics.
2. **Production-style web application** using scikit-learn, FastAPI, React, and Vite.

This separation helped distinguish between understanding how an algorithm works internally and building a practical deployable ML application.

---

## Project Goals

The main goals of SIFT were:

- Understand the complete ML project lifecycle.
- Perform exploratory data analysis before modeling.
- Make deliberate text preprocessing decisions.
- Implement TF-IDF from scratch.
- Implement Logistic Regression from scratch using NumPy.
- Understand gradient descent, loss, sigmoid activation, and parameter updates.
- Evaluate a classification model using appropriate metrics.
- Expose the trained pipeline through a REST API.
- Build a user-facing interface for real-time predictions.
- Deploy both the frontend and backend.
- Structure the project like a real software/ML project rather than a single notebook.

---

## Machine Learning Pipeline

```text
Raw SMS Dataset
      ↓
Data Cleaning
      ↓
Exploratory Data Analysis
      ↓
Train / Test Split
      ↓
TF-IDF Feature Engineering
      ↓
Logistic Regression
      ↓
Probability Prediction
      ↓
Classification
      ↓
Evaluation
      ↓
FastAPI Backend
      ↓
React Frontend
```

---

## Dataset

The project uses the `SMSSpamCollection` dataset.

The version used in this project contains:

- **5,169 SMS messages**
- Binary classification:
  - `ham` → legitimate message
  - `spam` → spam message
- Approximately **87% ham**
- Approximately **13% spam**

The class imbalance is important because an accuracy-only evaluation can be misleading.

For example, a model that predicted every message as ham would already achieve roughly 87% accuracy while being completely useless at detecting spam.

Therefore, SIFT considers additional classification metrics rather than relying only on accuracy.

---

## Exploratory Data Analysis

Before building the model, the dataset was inspected to understand its structure and characteristics.

The analysis included:

- Dataset dimensions
- Class distribution
- Missing values
- Duplicate messages
- Conflicting labels
- Message length
- Spam vs ham message examples
- Punctuation usage

The dataset contained:

- **5,169 rows**
- **No duplicate rows**
- **No conflicting labels**

One interesting observation was the difference in punctuation usage:

| Class | Average punctuation |
|-------|---------------------|
| Ham | ~3.83 |
| Spam | ~5.64 |

This supported the decision to preserve punctuation during preprocessing rather than removing it blindly.

---

## Text Preprocessing

The preprocessing pipeline intentionally remains simple.

The current cleaning process performs:

1. Conversion to string
2. Lowercasing
3. Whitespace normalization

Example:

```text
"FREE     MONEY\nNOW!!!"
```

becomes:

```text
"free money now!!!"
```

### What was intentionally preserved?

| Element | Decision |
|---------|----------|
| Case | Normalized |
| Whitespace | Normalized |
| Punctuation | Preserved |
| Numbers | Preserved |
| Currency symbols | Preserved |
| URLs | Preserved |
| Stopwords | Preserved |
| Duplicate messages | None found |
| Conflicting labels | None found |

The goal was to avoid aggressive preprocessing that could accidentally remove useful spam-related signals.

---

## TF-IDF Feature Engineering

Text cannot be directly provided to Logistic Regression, so messages must first be converted into numerical vectors.

SIFT uses **TF-IDF**.

### Term Frequency

Term Frequency measures how often a word occurs within a document.

```text
TF(term) = occurrences of term / total terms in document
```

### Inverse Document Frequency

IDF reduces the importance of words that appear across many documents.

```text
IDF(term) = log(N / document_frequency)
```

The final feature value is:

```text
TF-IDF = TF × IDF
```

For example, words such as:

```text
free
prize
winner
claim
urgent
```

may become useful features if their occurrence patterns differ significantly between spam and legitimate messages.

### Implementation

TF-IDF was implemented from scratch using Python and NumPy.

The implementation includes:

- Tokenization
- Vocabulary construction
- Term Frequency calculation
- Document Frequency calculation
- IDF calculation
- TF-IDF vector generation
- Document matrix transformation

The training vocabulary is built **only from the training data** to avoid information leakage from the test set.

The resulting feature space contains:

```text
7,758 features
```

with:

```text
X_train: (4134, 7758)
X_test : (1035, 7758)
```

---

## Logistic Regression From Scratch

To understand the algorithm rather than treating it as a black box, Logistic Regression was implemented using NumPy.

The model uses the sigmoid function:

```text
σ(z) = 1 / (1 + e^-z)
```

The sigmoid converts the model's linear output into a probability between 0 and 1.

The model then uses Binary Cross-Entropy as its loss function:

```text
Loss =
- [ y log(p) + (1-y) log(1-p) ]
```

The parameters are updated using gradient descent.

Conceptually:

```text
Input features
      ↓
Linear combination
      ↓
Sigmoid
      ↓
Probability
      ↓
Binary Cross-Entropy
      ↓
Gradients
      ↓
Parameter update
      ↓
Repeat
```

The implementation was tested independently before training on the complete dataset.

Controlled tests were used to verify:

- Sigmoid output
- Loss calculation
- Gradient calculation
- Weight updates
- Loss reduction during training

During training, the loss decreased from approximately:

```text
0.6931
```

to:

```text
0.3454
```

after 1,000 iterations.

This confirmed that the optimization process was learning rather than remaining at the initial parameter state.

---

## Model Evaluation

Because this is a spam detection problem with imbalanced classes, evaluation is not based solely on accuracy.

The project considers:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

The frontend currently displays an F1-score of approximately:

```text
84.12%
```

F1-score is particularly useful here because it balances precision and recall.

### Why not accuracy alone?

Consider a dataset containing 87% legitimate messages.

A model predicting:

```text
Everything → Ham
```

could achieve around 87% accuracy while detecting zero spam messages.

That is why SIFT evaluates how well the classifier actually distinguishes spam from legitimate messages.

---

## Production Prediction Pipeline

The deployed application uses a practical scikit-learn implementation of the same modeling approach.

The FastAPI application:

1. Loads the SMS dataset at startup.
2. Applies the project's text cleaning function.
3. Creates TF-IDF features using `TfidfVectorizer`.
4. Trains scikit-learn's `LogisticRegression`.
5. Accepts a message through the API.
6. Cleans the incoming message.
7. Converts it into TF-IDF features.
8. Generates a spam probability.
9. Classifies the message using a 0.5 probability threshold.
10. Returns the prediction and probability.

The deployed API therefore uses:

```text
TfidfVectorizer
        +
sklearn LogisticRegression
```

while the educational ML implementation in `src/` contains the from-scratch NumPy versions.

This distinction is intentional: the project demonstrates both **algorithmic understanding** and **practical application development**.

---

## API

The backend is built with **FastAPI**.

### Health Check

```http
GET /
```

Response:

```json
{
  "status": "online",
  "service": "SIFT API",
  "version": "1.0.0"
}
```

### Prediction

```http
POST /predict
```

Request:

```json
{
  "message": "Congratulations! You won a free prize!"
}
```

Response:

```json
{
  "prediction": "spam",
  "spam_probability": 0.98
}
```

The actual probability depends on the trained model and input message.

---

## Frontend

The user interface is built using:

- React
- Vite
- CSS

The interface is designed around a simple message-analysis workflow.

Users can:

1. Enter or paste an SMS message.
2. Submit it for analysis.
3. View the predicted classification.
4. View the spam probability.
5. See an explanation-oriented result interface.

The design follows an editorial/technical visual language using a warm paper background, dark typography, cobalt blue accents, and orange status elements.

---

## Architecture

```text
                    ┌──────────────────────┐
                    │       Browser        │
                    │   React + Vite UI    │
                    └──────────┬───────────┘
                               │
                               │ HTTP POST
                               ▼
                    ┌──────────────────────┐
                    │     FastAPI API      │
                    │      /predict        │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │     Text Cleaning    │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │       TF-IDF         │
                    │   Feature Extraction │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │  Logistic Regression │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Prediction +         │
                    │ Spam Probability     │
                    └──────────────────────┘
```

---

## Project Structure

```text
SIFT/
│
├── api/
│   └── main.py
│
├── data/
│   ├── raw/
│   │   └── SMSSpamCollection
│   └── processed/
│
├── frontend/
│   ├── src/
│   │   ├── App.jsx
│   │   ├── App.css
│   │   ├── index.css
│   │   └── main.jsx
│   ├── package.json
│   └── vite.config.js
│
├── models/
│   ├── logistic_regression.joblib
│   └── tfidf_vectorizer.joblib
│
├── notebooks/
│   └── 01_data_exploration.ipynb
│
├── src/
│   ├── data/
│   │   └── preprocessing.py
│   ├── features/
│   │   └── tfidf.py
│   ├── models/
│   │   └── logistic_regression.py
│   ├── evaluation/
│   └── utils/
│
├── tests/
│   ├── test_api.py
│   ├── test_preprocessing.py
│   └── ...
│
├── .gitignore
├── README.md
└── requirements.txt
```

---

## Testing

The project includes automated tests for the core components.

Tests cover areas such as:

- Text preprocessing
- TF-IDF calculations
- Tokenization
- Vocabulary construction
- Logistic Regression components
- API endpoints

The API tests verify both:

```text
GET /
```

and:

```text
POST /predict
```

The test suite also verifies that the returned spam probability remains within the valid range:

```text
0 ≤ probability ≤ 1
```

---

## Running Locally

### 1. Clone the repository

```bash
git clone <your-repository-url>
cd SIFT
```

### 2. Create a Python virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 3. Install Python dependencies

```powershell
pip install -r requirements.txt
```

### 4. Start the FastAPI backend

From the project root:

```powershell
uvicorn api.main:app --reload
```

The API will be available at:

```text
http://127.0.0.1:8000
```

FastAPI documentation is available at:

```text
http://127.0.0.1:8000/docs
```

### 5. Start the frontend

Open another terminal:

```powershell
cd frontend
npm install
npm run dev
```

The Vite development server will provide the local frontend URL.

---

## Production Build

To verify the frontend production build:

```powershell
cd frontend
npm run build
```

The production files are generated inside:

```text
frontend/dist/
```

---

## Deployment

The backend is deployed using Render.

**Live API:**  
https://sift-text-classifier.onrender.com

The React frontend is deployed using Vercel.

**Live Demo:**  
https://frontend-six-lilac-69.vercel.app/

Deployment architecture:

```text
Vercel
  │
  │ React frontend
  ▼
User Browser
  │
  │ API request
  ▼
Render
  │
  │ FastAPI
  ▼
SIFT ML Pipeline
```

---

## Technologies Used

### Machine Learning

- Python
- NumPy
- Pandas
- scikit-learn

### Backend

- FastAPI
- Pydantic
- Uvicorn

### Frontend

- React
- Vite
- CSS

### Development & Testing

- Jupyter Notebook
- VS Code
- Pytest
- Git
- GitHub

### Deployment

- Vercel
- Render

---

## Key Learning Outcomes

SIFT was built primarily as a learning project to understand what happens between a raw dataset and a deployed machine learning application.

The project helped demonstrate:

- How to inspect and understand a dataset before modeling.
- Why preprocessing decisions should be based on the problem rather than blindly following standard recipes.
- How TF-IDF converts text into numerical features.
- How Logistic Regression performs binary classification.
- How gradient descent updates model parameters.
- How to validate mathematical implementations using controlled tests.
- Why train/test separation matters.
- How data leakage can occur during feature engineering.
- Why class imbalance affects model evaluation.
- Why precision, recall, and F1-score can be more informative than accuracy.
- How to expose an ML model through an API.
- How a frontend communicates with a backend ML service.
- How to structure and deploy an end-to-end ML application.

---

## Limitations

SIFT is a learning-focused project and has several limitations.

- The dataset is relatively small.
- The classifier is trained on SMS messages and may not generalize to other forms of spam.
- The current production API trains the scikit-learn pipeline when the application starts.
- The model does not use more advanced NLP techniques such as word embeddings or transformer-based representations.
- The probability output should be interpreted as a model estimate rather than a guaranteed measure of whether a message is malicious.
- The current classifier is not designed for adversarial or continuously evolving spam patterns.

---

## Future Improvements

Potential improvements include:

- Persisting and loading the production model instead of retraining at API startup.
- Adding proper model versioning.
- Adding a dedicated evaluation module.
- Experimenting with n-grams and alternative feature representations.
- Hyperparameter tuning.
- Threshold optimization based on the precision/recall trade-off.
- Comparing Logistic Regression with models such as Naive Bayes and Linear SVM.
- Adding explainability for individual predictions.
- Monitoring model performance after deployment.
- Adding a larger and more diverse spam dataset.
- Introducing automated CI/CD testing.

---

## License

This project is intended primarily for educational and portfolio purposes.
