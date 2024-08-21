from django.urls import path
from .views import home, upload_image, recognition

urlpatterns = [
    path('', home, name='home'),
    path('upload/', upload_image, name='upload'),
    path('recognition/', recognition, name='recognition'),
]