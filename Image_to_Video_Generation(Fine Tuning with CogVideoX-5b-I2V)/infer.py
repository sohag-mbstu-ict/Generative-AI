import os
import torch
from diffusers import CogVideoXImageToVideoPipeline
from diffusers.schedulers import CogVideoXDPMScheduler
from diffusers.utils import export_to_video, load_image
from accelerate import Accelerator
from typing import Dict, Any
import wandb
import logging

# Setup logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Validation Function
def log_validation(
    accelerator: Accelerator,
    pipe: CogVideoXImageToVideoPipeline,
    args: Dict[str, Any],
    pipeline_args: Dict[str, Any],
    index_number,
    is_final_validation: bool = False,
):
    logger.info(
        f"Running validation... \n Generating {args['num_validation_videos']} videos with prompt: {pipeline_args['prompt']}."
    )
    pipe = pipe.to(accelerator.device)
    # Run inference
    generator = torch.Generator(device=accelerator.device).manual_seed(args["seed"]) if args.get("seed") else None
    video = pipe(**pipeline_args, generator=generator, output_type="np").frames[0]
    filename = os.path.join(args["output_dir"], f"{index_number}.mp4")
    export_to_video(video, filename, fps=8)
    print("11111111111111111111111111111111111111111111111111111111")


# Main Inference Script
def main():
    # Configuration
    args = {
        "pretrained_model_name_or_path": "/workspace/pretrained_model",
        "output_dir": "/workspace/output_17_dataset",
        "validation_prompts": ["Make a gentle hug video","Make a gentle hug video","Make a gentle hug video",
                               "Make a gentle hug video","Make a gentle hug video","Make a gentle hug video",
                               "Make a gentle hug video","Make a gentle hug video"],
        "validation_images": ["/workspace/cogvideox/Image_to_Video_Generation/images/hugg__.jpg",
                              "/workspace/cogvideox/Image_to_Video_Generation/images/hug_5.png",
                              "/workspace/cogvideox/Image_to_Video_Generation/images/hug_24.jpg",
                              "/workspace/cogvideox/Image_to_Video_Generation/images/hug_18.jpg",
                              "/workspace/cogvideox/Image_to_Video_Generation/images/hug_20.jpg",
                              "/workspace/cogvideox/Image_to_Video_Generation/images/hug_23.jpg",
                              "/workspace/cogvideox/Image_to_Video_Generation/images/hug.png",
                              "/workspace/cogvideox/Image_to_Video_Generation/images/Test_image.png"
                                ],
        "num_validation_videos": 1,
        "guidance_scale": 5.5,
        "height": 480,
        "width": 720,
        # "lora_weights_path": "/workspace/output/cogvideox-lora__optimizer_adam__steps_200__lr-schedule_cosine_with_restarts__learning-rate_1e-4/",
        # "lora_weight_name": "pytorch_lora_weights.safetensors",

        "lora_weights_path": "/workspace/output/cogvideox-lora__optimizer_adam__steps_500__lr-schedule_cosine_with_restarts__learning-rate_1e-4",
        "lora_weight_name": "pytorch_lora_weights.safetensors",

        "lora_scaling": 1,
        "enable_slicing": False,
        "enable_tiling": False,
        "enable_model_cpu_offload": False,
        "seed": 42,
    }

    # Initialize Accelerator
    accelerator = Accelerator()
    os.makedirs(args["output_dir"], exist_ok=True)

    # Load Pretrained Model
    logger.info("Loading the pre-trained model...")
    pipe = CogVideoXImageToVideoPipeline.from_pretrained(
        args["pretrained_model_name_or_path"], torch_dtype=torch.float16
    )
    pipe.scheduler = CogVideoXDPMScheduler.from_config(pipe.scheduler.config)

    # Enable memory optimizations
    if args["enable_slicing"]:
        pipe.vae.enable_slicing()
    if args["enable_tiling"]:
        pipe.vae.enable_tiling()
    if args["enable_model_cpu_offload"]:
        pipe.enable_model_cpu_offload()

    # Load LoRA Weights
    logger.info("Loading LoRA weights...")
    pipe.load_lora_weights(args["lora_weights_path"],
                             weight_name=args["lora_weight_name"],
                             adapter_name="cogvideox-lora")
    pipe.set_adapters(["cogvideox-lora"], [args["lora_scaling"]])

    # Run validation
    for i, (prompt, image_path) in enumerate(zip(args["validation_prompts"], args["validation_images"])):
        logger.info(f"Generating video for prompt: '{prompt}' and image: '{image_path}'")
        pipeline_args = {
            "image": load_image(image_path),
            "prompt": prompt,
            "guidance_scale": args["guidance_scale"],
            "height": args["height"],
            "width": args["width"],
        }

        videos = log_validation(
            accelerator=accelerator,
            pipe=pipe,
            args=args,
            pipeline_args=pipeline_args,
            index_number=i,
            is_final_validation=(i == len(args["validation_prompts"]) - 1),
        )
    logger.info("Inference and validation complete!")


if __name__ == "__main__":
    main()
