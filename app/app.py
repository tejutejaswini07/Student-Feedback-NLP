import streamlit as st
import pandas as pd
import joblib
import smtplib
from email.message import EmailMessage
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation


# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Student Feedback Analysis",
    page_icon="🎓",
    layout="wide"
)


# -----------------------------
# Title
# -----------------------------
st.title("🎓 AI-Based Student Feedback & Complaint Analysis System")

st.write(
    "This system uses Machine Learning and NLP to analyze "
    "student feedback."
)


# -----------------------------
# Load Models
# -----------------------------
sentiment_model = joblib.load("models/sentiment_model.pkl")
category_model = joblib.load("models/category_model.pkl")
priority_model = joblib.load("models/priority_model.pkl")


# -----------------------------
# Load Dataset
# -----------------------------
df = pd.read_csv("dataset/cleaned_feedback.csv")


# -----------------------------
# Dashboard Summary
# -----------------------------
st.header("📊 Dashboard")

total_feedback = len(df)

positive_count = (df["Sentiment"] == "Positive").sum()

negative_count = (df["Sentiment"] == "Negative").sum()

priority_predictions = priority_model.predict(
    df["Feedback"].astype(str).tolist()
)

high_priority_count = (
    priority_predictions == "High"
).sum()


col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("📊 Total Feedback", total_feedback)

with col2:
    st.metric("😊 Positive", positive_count)

with col3:
    st.metric("😞 Negative", negative_count)

with col4:
    st.metric("🚨 High Priority", high_priority_count)


# -----------------------------
# Additional Summary
# -----------------------------
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Total Feedback", len(df))

with col2:
    st.metric("Categories", df["Category"].nunique())

with col3:
    st.metric("Sentiments", df["Sentiment"].nunique())


# -----------------------------
# Charts
# -----------------------------
st.subheader("📈 Feedback Analysis")

col1, col2 = st.columns(2)

with col1:
    st.write("### Sentiment Distribution")

    sentiment_count = df["Sentiment"].value_counts()

    st.bar_chart(sentiment_count)


with col2:
    st.write("### Category Distribution")

    category_count = df["Category"].value_counts()

    st.bar_chart(category_count)


# -----------------------------
# Priority Distribution
# -----------------------------
st.subheader("🚨 Priority Distribution")

priority_predictions = priority_model.predict(
    df["Feedback"].astype(str).tolist()
)

priority_count = pd.Series(
    priority_predictions
).value_counts()

priority_count = priority_count.reindex(
    ["High", "Medium", "Low"],
    fill_value=0
)

st.bar_chart(priority_count)


# -----------------------------
# Keyword Analysis
# -----------------------------
st.header("🔑 Keyword Analysis")

keyword_vectorizer = TfidfVectorizer(
    stop_words="english",
    max_features=1000
)

keyword_matrix = keyword_vectorizer.fit_transform(
    df["Feedback"].astype(str)
)

keyword_names = keyword_vectorizer.get_feature_names_out()

keyword_scores = keyword_matrix.sum(axis=0).A1

keyword_df = pd.DataFrame({
    "Keyword": keyword_names,
    "Score": keyword_scores
})

keyword_df = keyword_df.sort_values(
    by="Score",
    ascending=False
).head(15)

st.subheader("Top 15 Keywords")

st.dataframe(
    keyword_df,
    use_container_width=True
)

st.bar_chart(
    keyword_df.set_index("Keyword")["Score"]
)


# -----------------------------
# CSV Upload
# -----------------------------
st.header("📁 Upload Student Feedback CSV")

uploaded_file = st.file_uploader(
    "Upload CSV file",
    type=["csv"]
)

if uploaded_file is not None:

    uploaded_df = pd.read_csv(uploaded_file)

    st.success("CSV file uploaded successfully!")

    st.subheader("Uploaded Data")

    st.dataframe(uploaded_df)

    if "Feedback" in uploaded_df.columns:

        if st.button("Analyze Uploaded Feedback"):

            results = []

            for text in uploaded_df["Feedback"]:

                if pd.notna(text):

                    sentiment = sentiment_model.predict(
                        [str(text)]
                    )[0]

                    category = category_model.predict(
                        [str(text)]
                    )[0]

                    priority = priority_model.predict(
                        [str(text)]
                    )[0]

                    results.append([
                        str(text),
                        sentiment,
                        category,
                        priority
                    ])

            result_df = pd.DataFrame(
                results,
                columns=[
                    "Feedback",
                    "Sentiment",
                    "Category",
                    "Priority"
                ]
            )

            st.subheader("📊 Analysis Results")

            st.dataframe(
                result_df,
                use_container_width=True
            )

    else:

        st.error(
            "CSV file must contain a column named 'Feedback'."
        )


# -----------------------------
# LDA Topic Analysis
# -----------------------------
st.header("📌 Topic Analysis")

st.write(
    "LDA is used to identify common topics present "
    "in student feedback."
)

topic_df = df.dropna(
    subset=["Feedback"]
)

if len(topic_df) >= 10:

    topic_vectorizer = CountVectorizer(
        stop_words="english",
        max_features=1000
    )

    X = topic_vectorizer.fit_transform(
        topic_df["Feedback"]
    )

    lda = LatentDirichletAllocation(
        n_components=6,
        random_state=42
    )

    lda.fit(X)

    words = topic_vectorizer.get_feature_names_out()

    st.subheader("🔍 Discovered Topics")

    for topic_index, topic in enumerate(
        lda.components_
    ):

        top_indices = topic.argsort()[-10:][::-1]

        top_words = []

        for word_index in top_indices:

            top_words.append(
                words[word_index]
            )

        st.write(
            f"**Topic {topic_index + 1}:** "
            + ", ".join(top_words)
        )

else:

    st.warning(
        "At least 10 feedback records are required "
        "for topic analysis."
    )


# -----------------------------
# Analyze New Feedback
# -----------------------------
st.header("🔍 Analyze New Feedback")

student_email = st.text_input(
    "📧 Student Email",
    placeholder="student@gmail.com"
)
# Check email format
email_pattern = r"^[^@\s]+@[^@\s]+\.[^@\s]+$"

if student_email:
    if not re.match(email_pattern, student_email):
        st.error("⚠️ Please enter a valid email address.")
        st.stop()
feedback = st.text_area(
    "Enter Student Feedback",
    placeholder="Example: The laboratory equipment is not working properly."
)


# -----------------------------
# Analyze Button
# -----------------------------
if st.button("Analyze Feedback"):

    if feedback.strip():

        # ML Predictions
        sentiment = sentiment_model.predict(
            [feedback]
        )[0]

        category = category_model.predict(
            [feedback]
        )[0]

        priority = priority_model.predict(
            [feedback]
        )[0]


        # -----------------------------
        # Prepare Email Response
        # -----------------------------
        if sentiment == "Positive":

            email_response = f"""
Dear Student,

Thank you for your positive feedback.

We are glad that you are satisfied with our {category}.

Your feedback is valuable to us.

Regards,
Student Feedback & Complaint Analysis System
"""

        elif priority == "High":

            email_response = f"""
Dear Student,

Thank you for bringing this issue to our attention.

Your concern regarding {category} has been identified as a high-priority issue.

Your feedback has been recorded for necessary attention.

Regards,
Student Feedback & Complaint Analysis System
"""

        else:

            email_response = f"""
Dear Student,

Thank you for your feedback.

Your concern regarding {category} has been recorded.

Your feedback will be considered for further improvement.

Regards,
Student Feedback & Complaint Analysis System
"""


        # -----------------------------
        # Keyword Extraction
        # -----------------------------
        feedback_vectorizer = TfidfVectorizer(
            stop_words="english"
        )

        X = feedback_vectorizer.fit_transform(
            [feedback]
        )

        words = feedback_vectorizer.get_feature_names_out()

        scores = X.toarray()[0]

        top_indices = scores.argsort()[-5:][::-1]

        keywords = []

        for index in top_indices:

            if scores[index] > 0:

                keywords.append(
                    words[index]
                )


        # -----------------------------
        # Save Analysis in Session
        # -----------------------------
        st.session_state["sentiment"] = sentiment
        st.session_state["category"] = category
        st.session_state["priority"] = priority
        st.session_state["keywords"] = keywords
        st.session_state["email_response"] = email_response
        st.session_state["student_email"] = student_email


        st.success(
            "Feedback analyzed successfully!"
        )


# -----------------------------
# Display Analysis Result
# -----------------------------
if "sentiment" in st.session_state:

    st.subheader("📋 Analysis Result")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Sentiment",
            st.session_state["sentiment"]
        )

    with col2:
        st.metric(
            "Category",
            st.session_state["category"]
        )

    with col3:
        st.metric(
            "Priority",
            st.session_state["priority"]
        )


    # -----------------------------
    # Keywords
    # -----------------------------
    st.subheader("🔑 Keywords")

    keywords = st.session_state["keywords"]

    if keywords:

        st.write(
            ", ".join(keywords)
        )

    else:

        st.write(
            "No keywords found."
        )


    # -----------------------------
    # Email Response Preview
    # -----------------------------
    st.subheader("📧 Email Response")

    st.text(
        st.session_state["email_response"]
    )


    # -----------------------------
    # Automatic Email Sending
    # -----------------------------
    email_to = st.session_state["student_email"]

    if email_to.strip():

        try:

            msg = EmailMessage()

            msg["Subject"] = "Student Feedback Analysis Response"

            msg["From"] = st.secrets["EMAIL_ADDRESS"]

            msg["To"] = email_to

            msg.set_content(
                st.session_state["email_response"]
            )

            with smtplib.SMTP_SSL(
                "smtp.gmail.com",
                465
            ) as server:

                server.login(
                    st.secrets["EMAIL_ADDRESS"],
                    st.secrets["EMAIL_PASSWORD"]
                )

                server.send_message(msg)

            st.success(
                "📧 Response email sent automatically!"
            )

        except Exception as e:

            st.error(
                f"Email could not be sent: {e}"
            )

    else:

        st.warning(
            "Please enter student email address."
        )


# -----------------------------
# Dataset Preview
# -----------------------------
st.header("📁 Dataset Preview")

st.dataframe(
    df.head(10)
)

st.success(
    "Dashboard loaded successfully!"
)