import pandas as pd

# Read Excel file
file_path = "dataset/finalDataset0.2.xlsx"

df = pd.read_excel(file_path)

print("Original Dataset:")
print(df.head())

# Create empty list
data = []

# Each category has rating column and feedback column
categories = [
    ("Teaching", 0, 1),
    ("Course Content", 2, 3),
    ("Examination", 4, 5),
    ("Lab Work", 6, 7),
    ("Library Facilities", 8, 9),
    ("Extra Curricular", 10, 11)
]

# Convert data into common format
for category, rating_col, text_col in categories:

    for i in range(len(df)):

        rating = df.iloc[i, rating_col]
        feedback = df.iloc[i, text_col]

        if pd.notna(feedback):

            if rating == 1:
                sentiment = "Positive"
            elif rating == -1:
                sentiment = "Negative"
            else:
                sentiment = "Neutral"

            data.append([
                str(feedback),
                sentiment,
                category
            ])

# Create cleaned dataset
cleaned_df = pd.DataFrame(
    data,
    columns=["Feedback", "Sentiment", "Category"]
)

# Remove empty and duplicate feedback
cleaned_df = cleaned_df.dropna()
cleaned_df = cleaned_df.drop_duplicates()

# Save cleaned dataset
cleaned_df.to_csv(
    "dataset/cleaned_feedback.csv",
    index=False
)

print("\nCleaned Dataset:")
print(cleaned_df.head())

print("\nTotal records:", len(cleaned_df))

print("\nDataset saved successfully!")