import streamlit as st
import joblib
import os
import sys
import numpy as np
import re

# ---------------------------
# Project Root
# ---------------------------

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

sys.path.append(BASE_DIR)

# ---------------------------
# Imports
# ---------------------------

from src.utils.pdf_reader import extract_text_from_pdf
from src.utils.skill_extractor import extract_skills
from src.utils.resume_score import calculate_resume_score
from src.utils.ats_matcher import calculate_ats_score

# ---------------------------
# Text Cleaning
# ---------------------------

def clean_resume_text(text):

    text = text.lower()

    text = re.sub(
        r"http\S+",
        " ",
        text
    )

    text = re.sub(
        r"www\S+",
        " ",
        text
    )

    text = re.sub(
        r"[^a-zA-Z ]",
        " ",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()

# ---------------------------
# Load Models
# ---------------------------

model = joblib.load(
    os.path.join(
        BASE_DIR,
        "models",
        "resume_classifier.pkl"
    )
)

vectorizer = joblib.load(
    os.path.join(
        BASE_DIR,
        "models",
        "tfidf_vectorizer.pkl"
    )
)

encoder = joblib.load(
    os.path.join(
        BASE_DIR,
        "models",
        "label_encoder.pkl"
    )
)

# ---------------------------
# Streamlit UI
# ---------------------------

st.set_page_config(
    page_title="AI Resume Screening System",
    page_icon="📄",
    layout="centered"
)

st.title(
    "📄 AI Resume Screening System"
)

st.write(
    "Upload a PDF Resume or Paste Resume Text"
)

# ---------------------------
# PDF Upload
# ---------------------------

uploaded_file = st.file_uploader(
    "Upload Resume PDF",
    type=["pdf"]
)

resume_text = ""

if uploaded_file is not None:

    try:

        resume_text = extract_text_from_pdf(
            uploaded_file
        )

        st.success(
            "PDF uploaded successfully!"
        )

        with st.expander(
            "View Extracted Text"
        ):
            st.write(
                resume_text
            )

    except Exception as e:

        st.error(
            f"Error reading PDF: {e}"
        )

# ---------------------------
# Manual Resume Input
# ---------------------------

resume_text_input = st.text_area(
    "Resume Text",
    height=200
)

if resume_text_input.strip():

    resume_text = resume_text_input

# ---------------------------
# Job Description
# ---------------------------

job_description = st.text_area(
    "Job Description (For ATS Matching)",
    height=200
)

# ---------------------------
# Predict Button
# ---------------------------

if st.button(
    "Predict Category"
):

    if resume_text.strip() == "":

        st.warning(
            "Please upload a PDF or enter resume text."
        )

    else:

        # ---------------------------
        # Clean Resume
        # ---------------------------

        cleaned_text = clean_resume_text(
            resume_text
        )

        # ---------------------------
        # Vectorize
        # ---------------------------

        resume_vector = vectorizer.transform(
            [cleaned_text]
        )

        # ---------------------------
        # Prediction
        # ---------------------------

        prediction = model.predict(
            resume_vector
        )

        category = encoder.inverse_transform(
            prediction
        )

        probabilities = model.predict_proba(
            resume_vector
        )[0]

        confidence = (
            np.max(probabilities)
            * 100
        )

        st.success(
            f"Predicted Category: {category[0]}"
        )

        st.info(
            f"Confidence Score: {confidence:.2f}%"
        )

        # ---------------------------
        # Resume Score
        # ---------------------------

        score = calculate_resume_score(
            resume_text
        )

        st.subheader(
            "Resume Score"
        )

        st.success(
            f"Resume Score: {score:.2f}/100"
        )

        # ---------------------------
        # Skills
        # ---------------------------

        skills = extract_skills(
            resume_text
        )

        st.subheader(
            "Detected Skills"
        )

        if skills:

            for skill in sorted(skills):

                st.write(
                    f"✅ {skill}"
                )

        else:

            st.warning(
                "No skills detected."
            )

        # ---------------------------
        # ATS Score
        # ---------------------------

        if job_description.strip():

            ats_score = calculate_ats_score(
                resume_text,
                job_description
            )

            st.subheader(
                "ATS Match Score"
            )

            st.success(
                f"ATS Match Score: {ats_score}%"
            )

        # ---------------------------
        # Top 3 Predictions
        # ---------------------------

        st.subheader(
            "Top 3 Predictions"
        )

        top_indices = np.argsort(
            probabilities
        )[-3:][::-1]

        for idx in top_indices:

            category_name = encoder.inverse_transform(
                [idx]
            )[0]

            prob = (
                probabilities[idx]
                * 100
            )

            st.write(
                f"✅ {category_name} — {prob:.2f}%"
            )