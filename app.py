from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import numpy as np
from scipy.sparse import hstack
from features import extract_features

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")
scaler = joblib.load("scaler.pkl")

class ReviewRequest(BaseModel):
    review_text: str

@app.get("/")
def home():
    return {"message": "Fake Review API Running"}

@app.post("/predict")
def predict_review(data: ReviewRequest):
    text = data.review_text

    tfidf = vectorizer.transform([text])
    custom = np.array([extract_features(text)])
    custom = scaler.transform(custom)
    combined = hstack([tfidf, custom])

    prob = model.predict_proba(combined)[0][1]

    if prob > 0.6:
        prediction = "FAKE"
    else:
        prediction = "GENUINE"

    return {
        "prediction": prediction,
        "fake_probability": round(float(prob), 3)
    }
