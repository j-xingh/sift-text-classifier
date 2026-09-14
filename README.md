# SIFT — Statistical Intelligence for Text Filtering

> An end-to-end SMS spam classification system built from data exploration and preprocessing to machine learning, evaluation, API development, and a production web interface.

SIFT is a machine learning project that classifies SMS messages as **spam** or **ham (legitimate)**.

The project was built to understand the complete ML development lifecycle rather than focusing only on model accuracy:

**Data → EDA → Cleaning → Feature Engineering → Model → Evaluation → API → Web Interface → Deployment**

---

## Live Demo

### Web Application
[https://YOUR-VERCEL-URL.vercel.app](https://YOUR-VERCEL-URL.vercel.app)

### Backend API
https://sift-text-classifier.onrender.com

---

## Project Overview

Spam messages are often characterized by recurring linguistic patterns such as promotional language, unusual punctuation, monetary terms, urgency, and suspicious offers.

SIFT converts an SMS message into numerical features using **TF-IDF** and uses **Logistic Regression** to estimate the probability that the message is spam.

The project also includes a custom implementation of important machine learning components to understand what happens internally rather than treating them as black boxes.

---

## Features

- SMS spam/ham classification
- Text preprocessing and normalization
- Exploratory Data Analysis
- TF-IDF feature engineering
- Logistic Regression classification
- Custom TF-IDF implementation using NumPy
- Logistic Regression implementation from scratch using NumPy
- Model evaluation using classification metrics
- FastAPI prediction endpoint
- React + Vite frontend
- Responsive editorial-style interface
- REST API integration
- Production deployment

---

## Machine Learning Pipeline

```text
SMS Message
     │
     ▼
Text Cleaning
     │
     ▼
TF-IDF Feature Extraction
     │
     ▼
Logistic Regression
     │
     ▼
Spam Probability
     │
     ▼
Spam / Ham Classification
The classification threshold is set to 0.5:

P(spam) ≥ 0.5  →  Spam
P(spam) < 0.5  →  Ham
Dataset

The project uses the SMSSpamCollection SMS spam dataset.

The dataset used in this project contains:

5,169 SMS messages
Two classes:
ham
spam
Approximately 87% ham
Approximately 13% spam

The class imbalance makes accuracy alone insufficient for evaluating the classifier.

This is why the project also considers precision, recall, and F1-score.

Exploratory Data Analysis

Before building the model, the dataset was analyzed to understand:

Class distribution
Message length
Duplicate messages
Missing values
Label consistency
Punctuation usage
Differences between spam and ham messages

The analysis showed that spam messages tend to contain stronger signals such as promotional language, monetary references, urgency, and higher punctuation usage.

Based on this analysis, punctuation and numerical information were preserved during preprocessing instead of being removed blindly.

Data Preprocessing

SIFT uses conservative text normalization.

Current preprocessing steps
Convert input to string
Convert text to lowercase
Normalize whitespace
Preserve punctuation
Preserve numbers
Preserve URLs

Example:

"FREE     MONEY!!!"

becomes:

"free money!!!"

The preprocessing strategy was intentionally kept simple to avoid destroying potentially useful spam signals.

Feature Engineering — TF-IDF

SIFT implements TF-IDF feature extraction from scratch.

TF-IDF gives higher importance to terms that are:

frequent within a particular message
relatively uncommon across the complete training corpus

The implementation includes:

Tokenization
Vocabulary construction
Term Frequency
Document Frequency
Inverse Document Frequency
TF-IDF vector generation
Document transformation into a numerical matrix

The vocabulary is built using the training data to avoid data leakage.

Feature matrix

The current training pipeline produces:

Training samples: 4,134
Test samples:     1,035
Features:         7,758
Logistic Regression

The project contains a Logistic Regression implementation built from scratch using NumPy.

The implementation covers the fundamental training cycle:

Initialize parameters
        ↓
Forward pass
        ↓
Sigmoid
        ↓
Binary cross-entropy loss
        ↓
Gradient calculation
        ↓
Parameter update
        ↓
Repeat

The implementation was created to understand the mathematics and optimization process behind logistic regression.

The deployed FastAPI application currently uses scikit-learn's LogisticRegression implementation for prediction.

Model Evaluation

Because the dataset is imbalanced, SIFT does not rely on accuracy alone.

The evaluation considers:

Accuracy
Precision
Recall
F1-score
Confusion Matrix

The current interface reports a test F1-score of:

84.12%

The F1-score is particularly useful here because it balances precision and recall.

API

SIFT exposes a FastAPI backend.

Health Check
GET /

Example response:

{
  "status": "online",
  "service": "SIFT API",
  "version": "1.0.0"
}
Prediction
POST /predict

Request:

{
  "message": "Congratulations! You won a free prize!"
}

Response:

{
  "prediction": "spam",
  "spam_probability": 0.95
}

The API:

Validates the message
Cleans the text
Converts it into TF-IDF features
Calculates spam probability
Applies the classification threshold
Returns the prediction
Frontend

The user interface was built using:

React
Vite
JavaScript
CSS
Lucide icons

The interface follows an editorial/technical visual language rather than a conventional dashboard aesthetic.

The main interaction is intentionally simple:

Paste message
      ↓
SIFT analyzes message
      ↓
View classification
      ↓
View spam probability
Tech Stack
Machine Learning
Python
NumPy
Pandas
Scikit-learn
Backend
FastAPI
Uvicorn
Pydantic
Frontend
React
Vite
JavaScript
CSS
Lucide React
Testing
Pytest
Deployment
Render — FastAPI backend
Vercel — React frontend
Project Structure
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
│   ├── public/
│   ├── src/
│   │   ├── App.jsx
│   │   ├── App.css
│   │   ├── index.css
│   │   └── main.jsx
│   ├── package.json
│   └── vite.config.js
│
├── models/
│
├── notebooks/
│   └── 01_data_exploration.ipynb
│
├── src/
│   ├── data/
│   │   └── preprocessing.py
│   ├── features/
│   │   └── tfidf.py
│   └── models/
│       └── logistic_regression.py
│
├── tests/
│   ├── test_api.py
│   ├── test_preprocessing.py
│   └── test_tfidf.py
│
├── .gitignore
├── README.md
└── requirements.txt
Running SIFT Locally
1. Clone the repository
git clone <YOUR-GITHUB-REPOSITORY-URL>
cd SIFT
2. Create a virtual environment
python -m venv .venv
3. Activate the environment

Windows PowerShell:

.venv\Scripts\Activate.ps1
4. Install Python dependencies
pip install -r requirements.txt
5. Start the FastAPI backend

From the project root:

uvicorn api.main:app --reload

The API will be available at:

http://127.0.0.1:8000
6. Start the frontend

Open another terminal:

cd frontend
npm install
npm run dev

The frontend will be available at:

http://localhost:5173
Running Tests

From the project root:

python -m pytest

The project currently contains tests covering:

API endpoints
Text preprocessing
TF-IDF implementation
Production Build

To create a production build of the React frontend:

cd frontend
npm run build

The production files are generated inside:

frontend/dist/
Deployment Architecture

SIFT is deployed using separate frontend and backend services.

                    User
                      │
                      ▼
              ┌──────────────┐
              │    Vercel    │
              │ React + Vite │
              └──────┬───────┘
                     │
                     │ POST /predict
                     ▼
              ┌──────────────┐
              │    Render    │
              │   FastAPI    │
              └──────┬───────┘
                     │
                     ▼
             Machine Learning
                Pipeline
                     │
             ┌───────┴───────┐
             ▼               ▼
           TF-IDF       Logistic Regression
             │               │
             └───────┬───────┘
                     ▼
               Prediction
Limitations

SIFT is an educational machine learning project and has several limitations.

The model is trained on a relatively small SMS dataset.
SMS language evolves over time.
New spam patterns may not be represented in the training data.
TF-IDF does not understand deeper semantic relationships between words.
The classifier should not be treated as a complete security or fraud detection system.
Probability scores should be interpreted as model confidence rather than guaranteed truth.
Future Improvements

Potential improvements include:

Hyperparameter tuning
Cross-validation
Improved handling of URLs and phone numbers
Character-level TF-IDF
N-gram features
Model comparison
Calibration of probability estimates
Better error analysis
Explainable predictions
Highlighting suspicious words or patterns
Larger and more diverse datasets
Monitoring model performance after deployment
What I Learned

This project was built as an exercise in understanding the complete machine learning development lifecycle.

The main learning objectives were:

How to explore a real dataset before modeling
Why data cleaning decisions should be driven by evidence
How TF-IDF converts text into numerical features
How Logistic Regression learns a decision boundary
How gradient descent updates model parameters
Why class imbalance affects evaluation
Why F1-score can be more informative than accuracy
How to structure an ML project
How to expose a model through an API
How a frontend communicates with a backend
How to take an ML experiment and turn it into a deployable application
Author

Jay Singh

Engineering — Electronics & Computer Science

Built as an end-to-end machine learning project to strengthen practical ML engineering skills.

License

This project is intended for educational and portfolio purposes.


## One thing before you save it

Replace:

```text
https://https://frontend-six-lilac-69.vercel.app/