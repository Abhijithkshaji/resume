INSTALLED_APPS = [
    'rest_framework',
    'corsheaders',  # Allow frontend to make requests
    'your_app_name',  # Replace with your actual app name
]

MIDDLEWARE = [
    'corsheaders.middleware.CorsMiddleware',  # Add this middleware
    'django.middleware.common.CommonMiddleware',
]

# Allow all frontend requests (you can restrict this later)
CORS_ALLOW_ALL_ORIGINS = True  
