from django.urls import path
from .views import ImageToVideoView

urlpatterns = [
    path('generate-video/', ImageToVideoView.as_view(), name='generate-video'),
]
