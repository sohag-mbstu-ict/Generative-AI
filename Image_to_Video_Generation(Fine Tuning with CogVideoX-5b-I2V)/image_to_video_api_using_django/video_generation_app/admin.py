from django.contrib import admin
from .models import GeneratedVideo

@admin.register(GeneratedVideo)
class GeneratedVideoAdmin(admin.ModelAdmin):
    list_display = ['id', 'image', 'text', 'generated_video']
