import pandas as pd
import numpy as np
import joblib
from scipy.sparse import hstack

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from xgboost import XGBClassifier

from features import extract_features

# ===============================
# 1️⃣ LOAD DATASET 1 (Fake Review)
# ===============================

df1 = pd.read_csv("data/reviews.csv").dropna()

# Adjust column names if needed
df1["label"] = df1["label"].map({"CG": 0, "OR": 1})  # Example
df1 = df1.rename(columns={"text_": "text"})

# ===============================
# 2️⃣ LOAD DATASET 2 (AI vs Human)
# ===============================

df2 = pd.read_csv("data/new_dataset.csv").dropna()

# Rename columns properly
df2 = df2.rename(columns={
    "article": "text",
    "class": "label"
})

# Keep only required columns
df2 = df2[["text", "label"]]

# ===============================
# 3️⃣ MERGE BOTH
# ===============================

df = pd.concat([df1[["text", "label"]], df2[["text", "label"]]])

print("Total samples:", len(df))

# ===============================
# 4️⃣ TF-IDF
# ===============================

vectorizer = TfidfVectorizer(
    max_features=12000,
    ngram_range=(1,3),
    min_df=2,
    max_df=0.9,
    stop_words="english"
)

X_tfidf = vectorizer.fit_transform(df["text"])

# ===============================
# 5️⃣ Custom Features
# ===============================

custom_features = np.array([extract_features(t) for t in df["text"]])

scaler = StandardScaler()
custom_features = scaler.fit_transform(custom_features)

X = hstack([X_tfidf, custom_features])
y = df["label"]

# ===============================
# 6️⃣ Train/Test Split
# ===============================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ===============================
# 7️⃣ Train Model
# ===============================

model = XGBClassifier(
    n_estimators=350,
    max_depth=6,
    learning_rate=0.07,
    subsample=0.9,
    colsample_bytree=0.8,
    eval_metric="logloss"
)

model.fit(X_train, y_train)

# ===============================
# 8️⃣ Evaluation
# ===============================

y_pred = model.predict(X_test)

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# ===============================
# 9️⃣ Save Model
# ===============================

joblib.dump(model, "model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")
joblib.dump(scaler, "scaler.pkl")

print("\nModel trained successfully with both datasets.")