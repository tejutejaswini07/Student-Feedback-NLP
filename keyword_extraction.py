import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# Load cleaned dataset
df = pd.read_csv("dataset/cleaned_feedback.csv")

# Remove missing feedback
df = df.dropna(subset=["Feedback"])

# TF-IDF
vectorizer = TfidfVectorizer(
    stop_words="english",
    max_features=1000
)

X = vectorizer.fit_transform(df["Feedback"])

# Get words
words = vectorizer.get_feature_names_out()

print("Keyword Extraction Results\n")

# Display keywords for first 10 feedback
for i in range(10):

    scores = X[i].toarray()[0]

    top_indices = scores.argsort()[-5:][::-1]

    keywords = []

    for index in top_indices:
        if scores[index] > 0:
            keywords.append(words[index])

    print("Feedback:", df["Feedback"].iloc[i])
    print("Keywords:", ", ".join(keywords))
    print()