import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
import pickle #It is a Python tool used to save Python objects to a file and load them later.


# ============================================
# LOAD DATASET
# ============================================

print("Loading dataset...")

data = pd.read_csv("hallucination_dataset.csv")

print("Dataset loaded!")
print()

print("Number of examples:", len(data))
print()


# ============================================
# SELECT FEATURES
# ============================================

X = data[
    [
        "answer_evidence_similarity",
        "question_evidence_similarity",
        "answer_length"
    ]
]


# ============================================
# SELECT LABEL
# ============================================

y = data["label"]


# ============================================
# SPLIT DATA
# ============================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Training examples:", len(X_train))
print("Testing examples:", len(X_test))
print()


# ============================================
# CREATE XGBOOST MODEL
# ============================================

print("Creating XGBoost model...")

model = XGBClassifier(
    n_estimators=100,
    max_depth=3,
    learning_rate=0.1,
    random_state=42
)

print("XGBoost model created!")
print()


# ============================================
# TRAIN MODEL
# ============================================

print("Training XGBoost model...")

model.fit(X_train, y_train)

print("Training completed!")
print()


# ============================================
# TEST MODEL
# ============================================

print("Making predictions...")

predictions = model.predict(X_test)

print("Predictions completed!")
print()


# ============================================
# CALCULATE ACCURACY
# ============================================

accuracy = accuracy_score(
    y_test,
    predictions
)


print("============================================")
print("MODEL RESULTS")
print("============================================")

print("Accuracy:", accuracy)

print("============================================")
print()


# ============================================
# SAVE TRAINED MODEL
# ============================================

print("Saving trained XGBoost model...")

with open("xgboost_model.pkl", "wb") as file:
    pickle.dump(model, file)

print("Model saved!")
print()

print("Saved as:")
print("xgboost_model.pkl")