import joblib
import numpy as np
from scipy.sparse import hstack

from features import extract_features
from similarity_detector import check_similarity
from ai_detector import ai_likeness_score
from rating_anomaly import rating_anomaly_score

# ==============================
# Load saved models
# ==============================

model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")
scaler = joblib.load("scaler.pkl")
tfidf_matrix = joblib.load("tfidf_matrix.pkl")

# ==============================
# Prediction Function
# ==============================

def predict_review(text):

    # 1️⃣ TF-IDF transform
    tfidf = vectorizer.transform([text])

    # 2️⃣ Custom feature extraction
    custom = np.array([extract_features(text)])
    custom = scaler.transform(custom)

    # 3️⃣ Combine features
    combined = hstack([tfidf, custom])

    # 4️⃣ ML Fake Probability
    prob = model.predict_proba(combined)[0][1]

    # 5️⃣ Similarity Detection
    similarity_score, similarity_flag = check_similarity(tfidf, tfidf_matrix)

    # 6️⃣ AI-Likeness Detection
    ai_score = ai_likeness_score(text)

    # 7️⃣ Rating Anomaly Detection (Demo Data for Now)
    # Replace this later with real rating history from extension/API
    ratings_history = [3,4,3,3,4,3,3,4,3,3,5,5,5,5,5]
    rating_score, rating_flag = rating_anomaly_score(ratings_history)

    # 8️⃣ Final Combined Risk Score
    final_score = (
        prob * 0.35
        + similarity_score * 0.15
        + ai_score * 0.15
        + rating_score * 0.35
    )

    # ==============================
    # OUTPUT SECTION
    # ==============================

    print("\n==============================")
    print("Review Analysis Result")
    print("==============================\n")

    print("Review:")
    print(text)

    print("\nFake Probability:", round(prob, 3))
    print("Similarity Score:", round(similarity_score, 3))
    print("AI-Likeness Score:", round(ai_score, 3))
    print("Rating Anomaly Score:", round(rating_score, 3))
    print("Final Risk Score:", round(final_score, 3))

    if final_score > 0.75:
        print("\n🔴 HIGH RISK REVIEW")
    elif final_score > 0.45:
        print("\n🟡 SUSPICIOUS REVIEW")
    else:
        print("\n🟢 LIKELY GENUINE")

    if similarity_flag:
        print("⚠ Similarity Alert: Possible coordinated review detected")

    if rating_flag:
        print("⚠ Statistical Rating Spike Detected")


# ==============================
# User Input
# ==============================

review_input = input("Enter review text:\n")
predict_review(review_input)