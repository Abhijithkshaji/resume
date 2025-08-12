import streamlit as st
import PyPDF2
import joblib
import re
import nltk
from nltk.corpus import stopwords
import pdfplumber

# Load necessary NLTK data
nltk.download("stopwords")

# Load the trained model and TF-IDF vectorizer
model = joblib.load("C:/Users/dell/OneDrive/Desktop/mainproject/resume/model (1).pkl")
vectorizer = joblib.load("C:/Users/dell/OneDrive/Desktop/mainproject/resume/tfidf.pkl")

# Define category mapping
category_map = {
    15: "Java Developer", 23: "Testing", 8: "DevOps Engineer", 20: "Python Developer",
    24: "Web Designing", 12: "HR", 13: "Hadoop", 3: "Blockchain", 10: "ETL Developer",
    18: "Operations Manager", 6: "Data Science", 22: "Sales", 16: "Mechanical Engineer",
    1: "Arts", 7: "Database", 11: "Electrical Engineering", 14: "Health and fitness",
    19: "PMO", 4: "Business Analyst", 9: "DotNet Developer", 2: "Automation Testing",
    17: "Network Security Engineer", 21: "SAP Developer", 5: "Civil Engineer", 0: "Advocate",
}

# Function to clean text
def clean_text(text):
    text = re.sub(r"https?://\S+|www\.\S+", "", text)  # Remove URLs
    text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b', '', text)  # Remove emails
    text = re.sub(r"[^\w\s]", "", text)  # Remove special characters
    stop_words = set(stopwords.words("english"))
    text = " ".join(word.lower() for word in text.split() if word.lower() not in stop_words)
    return text

# Function to extract text from PDF
def extract_text_from_pdf(pdf_file):
    text = ""
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            extracted_text = page.extract_text()
            if extracted_text:
                text += extracted_text + "\n"
    return text.strip()

# Streamlit App UI
st.title("📄 Resume Classification ")
st.write("Upload a resume in PDF format to classify it into a job category.")

uploaded_file = st.file_uploader("Choose a resume (PDF only)", type="pdf")

if uploaded_file is not None:
    st.success("✅ File uploaded successfully!")

    # Extract text from PDF
    resume_text = extract_text_from_pdf(uploaded_file)
    
    if resume_text:
        # Clean text
        cleaned_resume = clean_text(resume_text)

        # Transform text with TF-IDF
        input_features = vectorizer.transform([cleaned_resume])

        # Predict category
        prediction_id = model.predict(input_features)[0]
        category_name = category_map.get(prediction_id, "Unknown")

        # Display result
        st.subheader("🔍 Predicted Job Category:")
        st.success(f"**{category_name}**")

        # Display extracted resume text
        with st.expander("📝 View Extracted Resume Text"):
            st.text(resume_text)
    
    else:
        st.error("⚠️ Could not extract text from the uploaded PDF. Try another file.")
