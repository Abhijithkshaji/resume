from django.urls import path
from .views import predict_resume

urlpatterns = [
    path("predict/", predict_resume, name="predict_resume"),
]
