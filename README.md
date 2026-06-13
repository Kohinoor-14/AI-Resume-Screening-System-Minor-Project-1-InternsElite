# AI Resume Screening System

An AI-powered Resume Screening System built using Machine Learning, NLP, and Streamlit.

## Features

- Resume Category Prediction
- PDF Resume Upload
- Resume Text Input
- Skill Extraction
- Resume Score Calculation
- ATS Match Score
- Top 3 Job Predictions
- Streamlit Web Application

## Technologies Used

- Python
- Scikit-Learn
- Pandas
- NumPy
- NLP
- TF-IDF
- Logistic Regression
- Streamlit
- PyPDF2

## Dataset

- UpdatedResumeDataSet.csv
- 962 resumes
- 25 job categories

## Project Structure

```
Resume_Screening_System
│
├── app
│   └── app.py
│
├── data
│   ├── raw
│   └── processed
│
├── models
│   ├── resume_classifier.pkl
│   ├── tfidf_vectorizer.pkl
│   └── label_encoder.pkl
│
├── src
│   ├── preprocessing
│   ├── feature_engineering
│   ├── model
│   └── utils
│
├── requirements.txt
└── README.md
```

## Installation

### Clone Repository

```bash
git clone https://github.com/Kohinoor-14/AI-Resume-Screening-System-Minor-Project-1-InternsElite.git
```

### Move Into Project

```bash
cd AI-Resume-Screening-System-Minor-Project-1-InternsElite
```

### Create Virtual Environment

```bash
python -m venv venv
```

### Activate Environment

Windows:

```bash
venv\Scripts\activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

## Train Model

```bash
python src/model/train_model.py
```

## Run Application

```bash
streamlit run app/app.py
```

## Application URL

```text
http://localhost:8501
```

## Model Performance

- Accuracy: 99.48%
- TF-IDF Features: 5000
- Algorithm: Logistic Regression

## Author

Kohinoor Soni

InternsElite Minor Project 1