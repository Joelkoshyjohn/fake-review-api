import joblib
import numpy as np
from scipy.sparse import hstack
from features import extract_features

model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")
scaler = joblib.load("scaler.pkl")

def predict_text(text):
    tfidf = vectorizer.transform([text])
    custom = np.array([extract_features(text)])
    custom = scaler.transform(custom)
    combined = hstack([tfidf, custom])

    prob = model.predict_proba(combined)[0][1]

    print("\nSuspicious Probability:", round(float(prob), 3))

    if prob > 0.6:
        print("⚠️ SUSPICIOUS (Fake or AI)")
    else:
        print("✅ GENUINE")

text = input("Enter review:\n")
predict_text(text)