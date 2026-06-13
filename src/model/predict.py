import os
import joblib

# -----------------------------
# Get Project Root
# -----------------------------
BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

# -----------------------------
# Load Model
# -----------------------------
model = joblib.load(
    os.path.join(
        BASE_DIR,
        "models",
        "resume_classifier.pkl"
    )
)

# -----------------------------
# Load TF-IDF Vectorizer
# -----------------------------
vectorizer = joblib.load(
    os.path.join(
        BASE_DIR,
        "models",
        "tfidf_vectorizer.pkl"
    )
)

# -----------------------------
# Load Label Encoder
# -----------------------------
encoder = joblib.load(
    os.path.join(
        BASE_DIR,
        "models",
        "label_encoder.pkl"
    )
)

# -----------------------------
# Resume Input
# -----------------------------
print("\nResume Screening System")
print("-" * 40)

print("\nPaste Resume Text")
print("Type END on a new line when finished:\n")

lines = []

while True:

    line = input()

    if line.strip().upper() == "END":
        break

    lines.append(line)

resume_text = "\n".join(lines)

# -----------------------------
# Vectorize Resume
# -----------------------------
resume_vector = vectorizer.transform(
    [resume_text]
)

# -----------------------------
# Predict Category
# -----------------------------
prediction = model.predict(
    resume_vector
)

category = encoder.inverse_transform(
    prediction
)

# -----------------------------
# Predict Probabilities
# -----------------------------
probabilities = model.predict_proba(
    resume_vector
)[0]

confidence = max(probabilities) * 100

# -----------------------------
# Output
# -----------------------------
print("\nPredicted Category:")
print(category[0])

print("\nConfidence:")
print(f"{confidence:.2f}%")

# -----------------------------
# Top 3 Predictions
# -----------------------------
top_indices = probabilities.argsort()[-3:][::-1]

print("\nTop 3 Predictions:")
print("-" * 40)

for idx in top_indices:

    category_name = encoder.inverse_transform(
        [idx]
    )[0]

    score = probabilities[idx] * 100

    print(
        f"{category_name} -> {score:.2f}%"
    )