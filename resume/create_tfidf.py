import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the dataset
df = pd.read_csv("C:/Users/dell/OneDrive/Desktop/mainproject/resume/UpdatedResumeDataSet.csv")

# Check column names
print(df.columns)

# Clean text (if needed)
df["cleaned_resume"] = df["Resume"].str.lower()  # Simple cleaning

# Train TF-IDF vectorizer
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df["cleaned_resume"])

# Save the vectorizer
joblib.dump(vectorizer, "C:/Users/dell/OneDrive/Desktop/mainproject/resume/tfidf_vectorizer.pkl")

print("TF-IDF Vectorizer saved successfully!")
