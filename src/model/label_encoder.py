import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import joblib

# Get project root
BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )
)

# Load cleaned dataset
dataset_path = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "cleaned_resume_dataset.csv"
)

df = pd.read_csv(dataset_path)

# Encode labels
encoder = LabelEncoder()

df["Category_Encoded"] = encoder.fit_transform(df["Category"])

print("Encoded Categories:\n")
print(df[["Category", "Category_Encoded"]].drop_duplicates())

# Save encoder
encoder_path = os.path.join(
    BASE_DIR,
    "models",
    "label_encoder.pkl"
)

joblib.dump(encoder, encoder_path)

print("\nLabel Encoder saved successfully!")
print("Saved at:", encoder_path)