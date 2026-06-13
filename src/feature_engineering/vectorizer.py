import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

# Get project root
BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )
)

dataset_path = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "cleaned_resume_dataset.csv"
)

df = pd.read_csv(dataset_path)

# Improved TF-IDF
vectorizer = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2)
)

X = vectorizer.fit_transform(
    df["Cleaned_Resume"]
)

print("TF-IDF Matrix Shape:")
print(X.shape)

model_path = os.path.join(
    BASE_DIR,
    "models",
    "tfidf_vectorizer.pkl"
)

joblib.dump(
    vectorizer,
    model_path
)

print("\nVectorizer saved successfully!")
print("Saved at:", model_path)