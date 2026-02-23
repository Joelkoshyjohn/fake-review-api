
import pandas as pd
import numpy as np
import joblib
from scipy.sparse import hstack

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from xgboost import XGBClassifier

from features import extract_features

# Load dataset
df = pd.read_csv("data/reviews.csv")
df = df.dropna()

# Convert labels
df["label"] = df["label"].map({"CG": 0, "OR": 1})

X_text = df["text_"]
y = df["label"]

# Improved TF-IDF
vectorizer = TfidfVectorizer(
    max_features=8000,
    ngram_range=(1,3),
    min_df=5,
    max_df=0.9,
    stop_words="english"
)

X_tfidf = vectorizer.fit_transform(X_text)

# Save TF-IDF matrix for similarity detection
joblib.dump(X_tfidf, "tfidf_matrix.pkl")

# Custom features
custom_features = np.array([extract_features(text) for text in X_text])
scaler = StandardScaler()
custom_features = scaler.fit_transform(custom_features)

X = hstack([X_tfidf, custom_features])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# GridSearch for XGBoost
param_grid = {
    'n_estimators': [200, 300],
    'max_depth': [4, 6],
    'learning_rate': [0.05, 0.1],
    'subsample': [0.8, 1.0]
}

xgb = XGBClassifier(eval_metric="logloss")

grid = GridSearchCV(
    xgb,
    param_grid,
    cv=3,
    scoring='f1',
    verbose=1,
    n_jobs=-1
)

grid.fit(X_train, y_train)

model = grid.best_estimator_

# Evaluation
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# Cross-validation score
cv_scores = cross_val_score(model, X, y, cv=5, scoring='f1')
print("Cross-validated F1 Score:", cv_scores.mean())

# Save model
joblib.dump(model, "model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")
joblib.dump(scaler, "scaler.pkl")

print("Improved model trained and saved successfully.")
