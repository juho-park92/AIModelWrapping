from django.urls import path
from .views import home, upload_and_recognition

urlpatterns = [
    path('', home, name='home'),
    path('upload_and_recognition/', upload_and_recognition, name='upload_and_recognition'),
]