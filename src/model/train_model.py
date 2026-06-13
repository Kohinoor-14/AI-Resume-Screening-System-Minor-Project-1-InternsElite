import os
import pandas as pd
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

# =====================================
# Project Root
# =====================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

# =====================================
# Load Dataset
# =====================================

dataset_path = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "cleaned_resume_dataset.csv"
)

df = pd.read_csv(dataset_path)

print("\nDataset Shape:")
print(df.shape)

# =====================================
# Features & Labels
# =====================================

X_text = df["Cleaned_Resume"]
y_text = df["Category"]

# =====================================
# Label Encoding
# =====================================

encoder = LabelEncoder()

y = encoder.fit_transform(
    y_text
)

# Save Encoder
encoder_path = os.path.join(
    BASE_DIR,
    "models",
    "label_encoder.pkl"
)

joblib.dump(
    encoder,
    encoder_path
)

# =====================================
# TF-IDF Vectorizer
# =====================================

vectorizer = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2),
    stop_words="english"
)

X = vectorizer.fit_transform(
    X_text
)

print("\nTF-IDF Matrix Shape:")
print(X.shape)

# Save Vectorizer
vectorizer_path = os.path.join(
    BASE_DIR,
    "models",
    "tfidf_vectorizer.pkl"
)

joblib.dump(
    vectorizer,
    vectorizer_path
)

# =====================================
# Train-Test Split
# =====================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# =====================================
# Model
# =====================================

model = LogisticRegression(
    max_iter=5000,
    C=2.0,
    solver="lbfgs"
)

# =====================================
# Train
# =====================================

model.fit(
    X_train,
    y_train
)

# =====================================
# Predictions
# =====================================

y_pred = model.predict(
    X_test
)

# =====================================
# Accuracy
# =====================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

print("\n==============================")
print("MODEL ACCURACY")
print("==============================")
print(f"{accuracy * 100:.2f}%")

# =====================================
# Classification Report
# =====================================

print("\n==============================")
print("CLASSIFICATION REPORT")
print("==============================")

print(
    classification_report(
        y_test,
        y_pred
    )
)

# =====================================
# Save Model
# =====================================

model_path = os.path.join(
    BASE_DIR,
    "models",
    "resume_classifier.pkl"
)

joblib.dump(
    model,
    model_path
)

print("\n==============================")
print("FILES SAVED")
print("==============================")

print("Model:")
print(model_path)

print("\nVectorizer:")
print(vectorizer_path)

print("\nEncoder:")
print(encoder_path)

print("\nTraining Completed Successfully!")