from rest_framework import serializers
from .models import GeneratedVideo

class GeneratedVideoSerializer(serializers.ModelSerializer):
    class Meta:
        model = GeneratedVideo
        fields = ['id', 'image', 'text', 'generated_video']
