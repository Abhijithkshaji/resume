from rest_framework.decorators import api_view
from rest_framework.response import Response
import os
import joblib
import PyPDF2
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile

# Load your trained model and TF-IDF vectorizer
model = joblib.load("resume_classifier.pkl")  # Update with correct path
vectorizer = joblib.load("tfidf_vectorizer.pkl")  # TF-IDF vectorizer

# Function to extract text from PDF
def extract_text_from_pdf(pdf_path):
    text = ""
    with open(pdf_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            text += page.extract_text() + "\n"
    return text

@api_view(["POST"])
def predict_resume(request):
    if "resume" not in request.FILES:
        return Response({"error": "No file uploaded"}, status=400)

    file = request.FILES["resume"]
    file_path = default_storage.save("uploads/" + file.name, ContentFile(file.read()))

    # Extract text from PDF
    resume_text = extract_text_from_pdf(file_path)

    # Convert text using TF-IDF
    input_vector = vectorizer.transform([resume_text])

    # Get prediction
    prediction = model.predict(input_vector)[0]

    return Response({"prediction": prediction})
