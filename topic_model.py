import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# Load cleaned feedback
df = pd.read_csv("dataset/cleaned_feedback.csv")

# Remove missing feedback
df = df.dropna(subset=["Feedback"])

# Convert text into word counts
vectorizer = CountVectorizer(
    stop_words="english",
    max_features=1000
)

X = vectorizer.fit_transform(df["Feedback"])

# Create LDA model
lda = LatentDirichletAllocation(
    n_components=6,
    random_state=42
)

# Train LDA
lda.fit(X)

# Get words
words = vectorizer.get_feature_names_out()

# Display topics
print("LDA Topic Modeling Results\n")

for topic_index, topic in enumerate(lda.components_):

    print("Topic", topic_index + 1)

    top_words = topic.argsort()[-10:][::-1]

    for word_index in top_words:
        print(words[word_index], end=" ")

    print("\n")