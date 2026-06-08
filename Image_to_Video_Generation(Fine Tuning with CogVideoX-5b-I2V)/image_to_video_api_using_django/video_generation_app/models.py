from django.db import models

class GeneratedVideo(models.Model):
    image = models.ImageField(upload_to='images/')  # Uploaded image
    text = models.TextField()  # Prompt text
    generated_video = models.FileField(upload_to='videos/', blank=True, null=True)  # Generated video

    def __str__(self):
        return f"Video {self.id} - {self.text[:50]}"
