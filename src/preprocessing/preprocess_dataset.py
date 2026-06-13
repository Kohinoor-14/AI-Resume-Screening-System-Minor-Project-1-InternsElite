import os
import pandas as pd
from text_cleaner import clean_resume

# Get project root
BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )
)

# Dataset path
input_path = os.path.join(
    BASE_DIR,
    "data",
    "raw",
    "UpdatedResumeDataSet.csv"
)

# Output path
output_path = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "cleaned_resume_dataset.csv"
)

# Load dataset
df = pd.read_csv(input_path)

print("Original Dataset Shape:", df.shape)

# Clean resumes
df["Cleaned_Resume"] = df["Resume"].apply(clean_resume)

# Save cleaned dataset
df.to_csv(output_path, index=False)

print("\nDataset cleaned successfully!")
print("Saved to:", output_path)

print("\nFirst 5 Rows:")
print(df.head())