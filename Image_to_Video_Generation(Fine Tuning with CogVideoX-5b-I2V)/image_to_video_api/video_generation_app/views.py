from rest_framework.response import Response
from rest_framework.views import APIView
from rest_framework.parsers import MultiPartParser, FormParser
from django.conf import settings
from .models import GeneratedVideo
from .serializers import GeneratedVideoSerializer
import os
import torch
from diffusers.utils import export_to_video, load_image
from diffusers import (
    CogVideoXDPMScheduler,
    CogVideoXImageToVideoPipeline
)

class ImageToVideoView(APIView):
    parser_classes = (MultiPartParser, FormParser)

    def post(self, request, *args, **kwargs):
        """Handles image upload, generates video, and updates database."""
        serializer = GeneratedVideoSerializer(data=request.data)
        if serializer.is_valid():
            saved_instance = serializer.save()

            # Load the model
            pipe = CogVideoXImageToVideoPipeline.from_pretrained(
                "/workspace/pretrained_model",
                torch_dtype=torch.float16
            ).to("cuda")

            pipe.scheduler = CogVideoXDPMScheduler.from_config(pipe.scheduler.config)

            weight_path = "/workspace/fine_tuned_weight_500"
            # Load LoRA fine-tuned weights
            pipe.load_lora_weights(
                weight_path,
                weight_name="hug_500_2_gpu.safetensors",
                adapter_name="cogvideox-lora"
            )
            pipe.set_adapters(["cogvideox-lora"], [1])

            # Load uploaded image
            image_path = os.path.join(settings.MEDIA_ROOT, saved_instance.image.name)
            image = load_image(image_path)

            # Generate video
            video = pipe(
                image=image,
                prompt=saved_instance.text, # this is prompt
                guidance_scale=6,  # Adjust as needed
                use_dynamic_cfg=True
            ).frames[0]

            # Save the video
            video_filename = f'video_{saved_instance.id}.mp4'
            video_path = os.path.join(settings.MEDIA_ROOT, 'videos', video_filename)
            export_to_video(video, video_path, fps=8)

            # Update model with video path
            saved_instance.generated_video = f'videos/{video_filename}'
            saved_instance.save()

            # Prepare response
            response_data = GeneratedVideoSerializer(saved_instance).data
            # response_data['video_url'] = request.build_absolute_uri(settings.MEDIA_URL + saved_instance.generated_video)
            response_data['video_url'] = request.build_absolute_uri(saved_instance.generated_video.url)

            return Response(response_data, status=201)

        return Response(serializer.errors, status=400)
