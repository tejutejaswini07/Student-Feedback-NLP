# AI-Based Student Feedback & Complaint Analysis System using NLP and Machine Learning

## 📌 Project Overview

The **AI-Based Student Feedback & Complaint Analysis System** is a Machine Learning and Natural Language Processing (NLP) based application developed to automatically analyze student feedback and complaints.

The system helps institutions understand student feedback by identifying:

- Sentiment of the feedback
- Feedback category
- Priority of the feedback
- Total number of feedback records
- Category-wise feedback analysis

The application is developed using **Python, NLP, Machine Learning, and Streamlit**.

---

## 🎯 Project Objective

The main objective of this project is to reduce the manual effort required to analyze a large amount of student feedback.

The system automatically processes student feedback and provides useful information that can help institutions understand student concerns.

---

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Natural Language Processing (NLP)
- TF-IDF Vectorization
- Logistic Regression
- Joblib
- Streamlit
- Excel / CSV

---

## 🤖 Machine Learning Approach

The project uses **TF-IDF Vectorization** to convert text feedback into numerical features.

The converted text data is then given to Machine Learning classification models.

### Main Classification Tasks

### 1. Sentiment Analysis

Identifies the sentiment of student feedback.

Examples:

- Positive
- Negative
- Neutral

### 2. Category Classification

Classifies feedback into different categories such as:

- Teaching
- Infrastructure
- Academics
- Facilities
- Extracurricular Activities

### 3. Priority Classification

Identifies the priority level of student feedback.

Examples:

- High
- Medium
- Low

---

## 📂 Project Structure

```text
Student_Feedback_NLP/
│
├── app/
│   └── app.py
│
├── dataset/
│   ├── original_dataset.xlsx
│   └── cleaned_feedback.csv
│
├── models/
│   ├── sentiment_model.pkl
│   ├── category_model.pkl
│   └── priority_model.pkl
│
├── preprocess.py
├── train_sentiment.py
├── README.md
└── requirements.txt