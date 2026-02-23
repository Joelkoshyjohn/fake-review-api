from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
from scipy.sparse import hstack

from features import extract_features
from similarity_detector import check_similarity
from ai_detector import ai_likeness_score
from rating_anomaly import rating_anomaly_score

# ===============================
# Load models once at startup
# ===============================

model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")
scaler = joblib.load("scaler.pkl")
tfidf_matrix = joblib.load("tfidf_matrix.pkl")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# ===============================
# Request Schema
# ===============================

class ReviewRequest(BaseModel):
    review_text: str
    ratings_history: list[int] = []

# ===============================
# API Endpoint
# ===============================

@app.post("/predict")
def predict_review(data: ReviewRequest):

    text = data.review_text
    ratings_history = data.ratings_history

    # TF-IDF
    tfidf = vectorizer.transform([text])

    # Custom features
    custom = np.array([extract_features(text)])
    custom = scaler.transform(custom)

    combined = hstack([tfidf, custom])

    # ML probability
    prob = model.predict_proba(combined)[0][1]

    # Similarity detection
    similarity_score, similarity_flag = check_similarity(tfidf, tfidf_matrix)

    # AI-likeness
    ai_score = ai_likeness_score(text)

    # Rating anomaly (if ratings provided)
    if len(ratings_history) > 0:
        rating_score, rating_flag = rating_anomaly_score(ratings_history)
    else:
        rating_score = 0
        rating_flag = False

    # Final risk score
    final_score = (
        prob * 0.35
        + similarity_score * 0.15
        + ai_score * 0.15
        + rating_score * 0.35
    )

    return {
        "fake_probability": round(float(prob), 3),
        "similarity_score": round(float(similarity_score), 3),
        "ai_likeness_score": round(float(ai_score), 3),
        "rating_anomaly_score": round(float(rating_score), 3),
        "final_risk_score": round(float(final_score), 3),
        "similarity_alert": similarity_flag,
        "rating_alert": rating_flag
    }