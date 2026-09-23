import pandas as pd
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report

# Load training and testing datasets
train_df = pd.read_csv("dataset/university_query_train.csv")
test_df = pd.read_csv("dataset/university_query_test.csv")

# Remove missing values
train_df = train_df.dropna(subset=["Student_Query", "Priority_Label"])
test_df = test_df.dropna(subset=["Student_Query", "Priority_Label"])

# Input and output
X_train = train_df["Student_Query"]
y_train = train_df["Priority_Label"]

X_test = test_df["Student_Query"]
y_test = test_df["Priority_Label"]

# TF-IDF + Logistic Regression
model = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("classifier", LogisticRegression(max_iter=1000))
])

# Train model
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("Priority ML Model Training Completed!")
print("Accuracy:", accuracy)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Save model
joblib.dump(model, "models/priority_model.pkl")

print("\nPriority ML model saved successfully!")