import pandas as pd

# Load cleaned dataset
df = pd.read_csv("dataset/cleaned_feedback.csv")

# Function to detect priority
def get_priority(feedback):

    feedback = feedback.lower()

    high_words = [
        "not working",
        "broken",
        "unsafe",
        "emergency",
        "serious",
        "danger",
        "no water",
        "no electricity",
        "problem",
        "worst"
    ]

    medium_words = [
        "poor",
        "bad",
        "issue",
        "delay",
        "lack",
        "need improvement",
        "insufficient"
    ]

    # Check high priority
    for word in high_words:
        if word in feedback:
            return "High"

    # Check medium priority
    for word in medium_words:
        if word in feedback:
            return "Medium"

    # Otherwise low priority
    return "Low"


# Apply priority detection
df["Priority"] = df["Feedback"].apply(get_priority)

# Save updated dataset
df.to_csv("dataset/feedback_with_priority.csv", index=False)

# Display results
print("Priority Detection Completed!")

print("\nSample Results:")
print(df[["Feedback", "Sentiment", "Category", "Priority"]].head(10))

print("\nPriority Counts:")
print(df["Priority"].value_counts())

print("\nDataset saved successfully!")