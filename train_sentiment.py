import pandas as pd
import joblib

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report


# --------------------------------
# 1. Load Dataset
# --------------------------------
df = pd.read_csv("dataset/cleaned_feedback.csv")

# Remove missing values
df = df.dropna(subset=["Feedback", "Sentiment"])

# Original dataset
X = df["Feedback"].astype(str)
y = df["Sentiment"]


# --------------------------------
# 2. Add Extra Training Examples
# --------------------------------
extra_feedback = [
    "Teaching is very bad",
    "Teaching quality is very poor",
    "The teaching is bad",
    "The teacher explains very poorly",
    "The teacher does not explain properly",
    "Teaching is not good",
    "The teaching quality is poor",
    "Classes are very bad",
    "The faculty teaching is poor",
    "I am unhappy with the teaching",

    "Teaching is very good",
    "Teaching quality is excellent",
    "The teaching is excellent",
    "The teacher explains very clearly",
    "The teacher explains concepts well",
    "Teaching is good",
    "The teaching quality is good",
    "Classes are very good",
    "The faculty teaching is excellent",
    "I am happy with the teaching"
]

extra_sentiment = [
    "Negative",
    "Negative",
    "Negative",
    "Negative",
    "Negative",
    "Negative",
    "Negative",
    "Negative",
    "Negative",
    "Negative",

    "Positive",
    "Positive",
    "Positive",
    "Positive",
    "Positive",
    "Positive",
    "Positive",
    "Positive",
    "Positive",
    "Positive"
]


# Add extra examples to original dataset
extra_df = pd.DataFrame({
    "Feedback": extra_feedback,
    "Sentiment": extra_sentiment
})

X = pd.concat(
    [X, extra_df["Feedback"]],
    ignore_index=True
)

y = pd.concat(
    [y, extra_df["Sentiment"]],
    ignore_index=True
)


# --------------------------------
# 3. Split Dataset
# --------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# --------------------------------
# 4. ML Model
# --------------------------------
model = Pipeline([
    (
        "tfidf",
        TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 2),
            lowercase=True,
            sublinear_tf=True
        )
    ),

    (
        "classifier",
        LogisticRegression(
            max_iter=3000,
            class_weight="balanced"
        )
    )
])


# --------------------------------
# 5. Train Model
# --------------------------------
model.fit(X_train, y_train)


# --------------------------------
# 6. Evaluate Model
# --------------------------------
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("Sentiment Model Accuracy:", accuracy)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# --------------------------------
# 7. Test Feedback
# --------------------------------
test_feedback = [
    "teaching is very bad"
]

prediction = model.predict(test_feedback)

print("\nTest Feedback:", test_feedback[0])
print("Predicted Sentiment:", prediction[0])


# --------------------------------
# 8. Test More Examples
# --------------------------------
test_examples = [
    "The teaching quality is excellent",
    "The teacher does not explain properly",
    "The classes are very good",
    "The teaching is poor"
]

print("\nAdditional Predictions:")

for text in test_examples:
    result = model.predict([text])[0]
    print(text, "->", result)


# --------------------------------
# 9. Save Model
# --------------------------------
joblib.dump(
    model,
    "models/sentiment_model.pkl"
)

print("\nSentiment model saved successfully!")