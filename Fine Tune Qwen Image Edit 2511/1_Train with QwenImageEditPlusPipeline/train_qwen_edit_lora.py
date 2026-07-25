import argparse
import copy
from copy import deepcopy
import logging
import os
import shutil
import psutil

import torch
from tqdm.auto import tqdm

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration
import datasets
import diffusers
from diffusers import FlowMatchEulerDiscreteScheduler
from diffusers import (
    AutoencoderKLQwenImage,
    QwenImagePipeline,
    QwenImageTransformer2DModel,
)
from diffusers.optimization import get_scheduler
from diffusers.training_utils import (
    compute_density_for_timestep_sampling,
    compute_loss_weighting_for_sd3,
)
from diffusers.utils import convert_state_dict_to_diffusers
from diffusers.utils.torch_utils import is_compiled_module
from image_datasets.control_dataset import loader, image_resize
from omegaconf import OmegaConf
from peft import LoraConfig
from peft.utils import get_peft_model_state_dict
import transformers
from PIL import Image
import numpy as np
from optimum.quanto import quantize, qfloat8, freeze
import bitsandbytes as bnb
logger = get_logger(__name__, log_level="INFO")
from diffusers.loaders import AttnProcsLayers
from diffusers import QwenImageEditPlusPipeline
import gc
import math
from track_gpu_cpu_vram import profile_model_loading

def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        required=True,
        help="path to config",
    )
    args = parser.parse_args()


    return args.config

import torch
from torch.utils.data import Dataset, DataLoader

class ToyDataset(Dataset):
    def __init__(self, num_samples=100, input_dim=10):
        self.data = torch.randn(num_samples, input_dim)    # random features
        self.labels = torch.randint(0, 2, (num_samples,))  # random labels: 0 or 1

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

    def __len__(self):
        return len(self.data)

def lora_processors(model):
    processors = {}

    def fn_recursive_add_processors(name: str, module: torch.nn.Module, processors):
        if 'lora' in name:
            processors[name] = module
            print(name)
        for sub_name, child in module.named_children():
            fn_recursive_add_processors(f"{name}.{sub_name}", child, processors)
        return processors

    for name, module in model.named_children():
        fn_recursive_add_processors(name, module, processors)
    return processors

def calculate_dimensions(target_area, ratio):
    width = math.sqrt(target_area * ratio)
    height = width / ratio

    width = round(width / 32) * 32
    height = round(height / 32) * 32

    return width, height, None

def main():
    args = OmegaConf.load(parse_args())
    args.save_cache_on_disk = False
    args.precompute_text_embeddings = True
    args.precompute_image_embeddings = True

    logging_dir = os.path.join(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )
    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()


    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
        args.mixed_precision = accelerator.mixed_precision
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
        args.mixed_precision = accelerator.mixed_precision
    
    text_encoding_pipeline, vae, flux_transformer, noise_scheduler = profile_model_loading(args, weight_dtype)
    # text_encoding_pipeline = QwenImageEditPlusPipeline.from_pretrained(
    #     args.pretrained_model_name_or_path, transformer=None, vae=None, torch_dtype=weight_dtype)

    text_encoding_pipeline.to(accelerator.device)
    cached_text_embeddings = None
    txt_cache_dir = None
    if args.precompute_text_embeddings or args.precompute_image_embeddings:
        if accelerator.is_main_process:
            cache_dir = os.path.join(args.output_dir, "cache")
            os.makedirs(cache_dir, exist_ok=True)
        accelerator.wait_for_everyone()
        cache_dir = os.path.join(args.output_dir, "cache")
        
    if args.precompute_text_embeddings:
        with torch.no_grad():
            if args.save_cache_on_disk:
                txt_cache_dir = os.path.join(cache_dir, "text_embs")
                os.makedirs(txt_cache_dir, exist_ok=True)
            else:
                cached_text_embeddings = {}

            image_files = [
                i for i in os.listdir(args.data_config.control_dir)
                if i.lower().endswith(('.png', '.jpg', '.jpeg'))
            ]

            for img_name in tqdm(image_files):
                img_path = os.path.join(args.data_config.control_dir, img_name)
                txt_name = img_name.rsplit('.', 1)[0] + '.txt'
                txt_path = os.path.join(args.data_config.img_dir, txt_name)

                # 1. Load raw PIL Image directly
                raw_img = Image.open(img_path).convert('RGB')

                with open(txt_path, 'r', encoding='utf-8') as f:
                    prompt = f.read().strip()

                # 2. Encode main prompt
                prompt_embeds, prompt_embeds_mask = text_encoding_pipeline.encode_prompt(
                    image=[raw_img],  # Pass raw PIL Image inside a list
                    prompt=[prompt],
                    device=text_encoding_pipeline.device,
                    num_images_per_prompt=1,
                    max_sequence_length=1024,
                )

                # Safely slice tensors handling potential None for mask
                p_embed = prompt_embeds[0].to('cpu') if prompt_embeds is not None else None
                p_mask = prompt_embeds_mask[0].to('cpu') if prompt_embeds_mask is not None else None

                if args.save_cache_on_disk:
                    save_path = os.path.join(txt_cache_dir, f"{txt_name}.pt")
                    torch.save({'prompt_embeds': p_embed, 'prompt_embeds_mask': p_mask}, save_path)
                else:
                    cached_text_embeddings[txt_name] = {
                        'prompt_embeds': p_embed,
                        'prompt_embeds_mask': p_mask
                    }

                # 3. Encode empty prompt
                prompt_embeds_empty, prompt_embeds_mask_empty = text_encoding_pipeline.encode_prompt(
                    image=[raw_img],
                    prompt=[' '],
                    device=text_encoding_pipeline.device,
                    num_images_per_prompt=1,
                    max_sequence_length=1024,
                )

                p_embed_empty = prompt_embeds_empty[0].to('cpu') if prompt_embeds_empty is not None else None
                p_mask_empty = prompt_embeds_mask_empty[0].to('cpu') if prompt_embeds_mask_empty is not None else None

                if not args.save_cache_on_disk:
                    cached_text_embeddings[f"{txt_name}_empty_embedding"] = {
                        'prompt_embeds': p_embed_empty,
                        'prompt_embeds_mask': p_mask_empty
                    }


    
    # vae = AutoencoderKLQwenImage.from_pretrained(
    #     args.pretrained_model_name_or_path,
    #     subfolder="vae",)

    vae.to(accelerator.device, dtype=weight_dtype)
    cached_image_embeddings = None
    img_cache_dir = None
    cached_image_embeddings_control = None
    if args.precompute_image_embeddings:
        if args.save_cache_on_disk:
            img_cache_dir = os.path.join(cache_dir, "img_embs")
            os.makedirs(img_cache_dir, exist_ok=True)
        else:
            cached_image_embeddings = {}
        with torch.no_grad():
            for img_name in tqdm([i for i in os.listdir(args.data_config.img_dir) if ".png" in i or ".jpg" in i or ".jpeg" in i]):
                img = Image.open(os.path.join(args.data_config.img_dir, img_name)).convert('RGB')
                calculated_width, calculated_height, _ = calculate_dimensions(1024 * 1024, img.size[0] / img.size[1])
                img = text_encoding_pipeline.image_processor.resize(img, calculated_height, calculated_width)

                img = torch.from_numpy((np.array(img) / 127.5) - 1)
                img = img.permute(2, 0, 1).unsqueeze(0)
                pixel_values = img.unsqueeze(2)
                pixel_values = pixel_values.to(dtype=weight_dtype).to(accelerator.device)
        
                pixel_latents = vae.encode(pixel_values).latent_dist.sample().to('cpu')[0]
                if args.save_cache_on_disk:
                    torch.save(pixel_latents, os.path.join(img_cache_dir, img_name + '.pt'))
                    del pixel_latents
                else:
                    cached_image_embeddings[img_name] = pixel_latents
        if args.save_cache_on_disk:
            img_cache_dir = os.path.join(cache_dir, "img_embs_control")
            os.makedirs(img_cache_dir, exist_ok=True)
        else:
            cached_image_embeddings_control = {}
        with torch.no_grad():
            for img_name in tqdm([i for i in os.listdir(args.data_config.control_dir) if ".png" in i or ".jpg" in i or ".jpeg" in i]):
                img = Image.open(os.path.join(args.data_config.control_dir, img_name)).convert('RGB')
                calculated_width, calculated_height, _ = calculate_dimensions(1024 * 1024, img.size[0] / img.size[1])
                img = text_encoding_pipeline.image_processor.resize(img, calculated_height, calculated_width)

                img = torch.from_numpy((np.array(img) / 127.5) - 1)
                img = img.permute(2, 0, 1).unsqueeze(0)
                pixel_values = img.unsqueeze(2)
                pixel_values = pixel_values.to(dtype=weight_dtype).to(accelerator.device)
        
                pixel_latents = vae.encode(pixel_values).latent_dist.sample().to('cpu')[0]
                if args.save_cache_on_disk:
                    torch.save(pixel_latents, os.path.join(img_cache_dir, img_name + '.pt'))
                    del pixel_latents
                else:
                    cached_image_embeddings_control[img_name] = pixel_latents
        vae.to('cpu')
        torch.cuda.empty_cache()
        text_encoding_pipeline.to("cpu")
        torch.cuda.empty_cache()
    del text_encoding_pipeline
    gc.collect()
    # del vae
    gc.collect()

    process = psutil.Process(os.getpid())

    print("=" * 60)
    print("Before transformer loading")
    print(f"CPU RSS: {process.memory_info().rss/1024**3:.2f} GB")

    if torch.cuda.is_available():
        print(
            f"GPU allocated: {torch.cuda.memory_allocated()/1024**3:.2f} GB"
        )
        print(
            f"GPU reserved : {torch.cuda.memory_reserved()/1024**3:.2f} GB"
        )
    print("=" * 60)
    # flux_transformer = QwenImageTransformer2DModel.from_pretrained(
    #                     args.pretrained_model_name_or_path,
    #                     subfolder="transformer",
    #                     torch_dtype=weight_dtype,
    #                     low_cpu_mem_usage=True, )
    if args.quantize:
        torch_dtype = weight_dtype
        device = accelerator.device
        all_blocks = list(flux_transformer.transformer_blocks)
        for block in tqdm(all_blocks):
            block.to(device, dtype=torch_dtype)
            quantize(block, weights=qfloat8)
            freeze(block)
            block.to('cpu')
        flux_transformer.to(device, dtype=torch_dtype)
        quantize(flux_transformer, weights=qfloat8)
        freeze(flux_transformer)
        #quantize(flux_transformer, weights=qint8, activations=qint8)
        #freeze(flux_transformer)
        
    lora_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.rank,
        init_lora_weights="gaussian",
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
    )
    flux_transformer.to(accelerator.device)
    #flux_transformer.add_adapter(lora_config)
    # noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
    #     args.pretrained_model_name_or_path,
    #     subfolder="scheduler",)

    if args.quantize:
        flux_transformer.to(accelerator.device)
    else:
        flux_transformer.to(accelerator.device, dtype=weight_dtype)
    flux_transformer.add_adapter(lora_config)
    noise_scheduler_copy = copy.deepcopy(noise_scheduler)
    def get_sigmas(timesteps, n_dim=4, dtype=torch.float32):
        sigmas = noise_scheduler_copy.sigmas.to(device=accelerator.device, dtype=dtype)
        schedule_timesteps = noise_scheduler_copy.timesteps.to(accelerator.device)
        timesteps = timesteps.to(accelerator.device)
        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]
    
        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma
        
    flux_transformer.requires_grad_(False)


    flux_transformer.train()
    optimizer_cls = torch.optim.AdamW
    for n, param in flux_transformer.named_parameters():
        if 'lora' not in n:
            param.requires_grad = False
            pass
        else:
            param.requires_grad = True
            print(n)
    print(sum([p.numel() for p in flux_transformer.parameters() if p.requires_grad]) / 1000000, 'parameters')
    lora_layers = filter(lambda p: p.requires_grad, flux_transformer.parameters())
    lora_layers_model = AttnProcsLayers(lora_processors(flux_transformer))
    flux_transformer.enable_gradient_checkpointing()
    if args.adam8bit:
        optimizer = bnb.optim.Adam8bit(lora_layers,
            lr=args.learning_rate,
            betas=(args.adam_beta1, args.adam_beta2),)
    else:
        optimizer = optimizer_cls(
            lora_layers,
            lr=args.learning_rate,
            betas=(args.adam_beta1, args.adam_beta2),
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
        )
    train_dataloader = loader(cached_text_embeddings=cached_text_embeddings, cached_image_embeddings=cached_image_embeddings, 
                              cached_image_embeddings_control=cached_image_embeddings_control,
                              **args.data_config)

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
    )
    global_step = 0
    dataset1 = ToyDataset(num_samples=100, input_dim=10)
    dataloader1 = DataLoader(dataset1, batch_size=8, shuffle=True)

    lora_layers_model, optimizer, _, lr_scheduler = accelerator.prepare(
        lora_layers_model, optimizer, dataloader1, lr_scheduler
    )

    initial_global_step = 0

    if accelerator.is_main_process:
        accelerator.init_trackers(
        project_name=args.tracker_project_name,  # "qwen_lora_editing"
        config=vars(args),)  # Logs your YAML settings to the W&B dashboard
        # accelerator.init_trackers(args.tracker_project_name, {"test": None}) # -------------------------

    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        disable=not accelerator.is_local_main_process,
    )
    vae_scale_factor = 2 ** len(vae.temperal_downsample)
    for epoch in range(1):
        train_loss = 0.0
        for step, batch in enumerate(train_dataloader):
            # # ... forward & backward passes ...
            # if step % args.logging_steps == 0:
            #     accelerator.log(
            #         {"train_loss": loss.item(), "lr": lr_scheduler.get_last_lr()[0]},
            #         step=step,)

            with accelerator.accumulate(flux_transformer):
                if args.precompute_text_embeddings:
                    img, prompt_embeds, prompt_embeds_mask, control_img = batch
                    # Move main embeddings
                    prompt_embeds = prompt_embeds.to(dtype=weight_dtype, device=accelerator.device)
                    control_img = control_img.to(dtype=weight_dtype, device=accelerator.device)  
                    # Check if mask is empty/invalid; construct a full attention mask if so
                    if prompt_embeds_mask is None or prompt_embeds_mask.numel() == 0 or prompt_embeds_mask.shape[-1] == 0:
                        bsz, seq_len = prompt_embeds.shape[0], prompt_embeds.shape[1]
                        prompt_embeds_mask = torch.ones((bsz, seq_len), dtype=torch.int32, device=accelerator.device)
                    else:
                        prompt_embeds_mask = prompt_embeds_mask.to(dtype=torch.int32, device=accelerator.device)     
                else:
                    img, prompts = batch
                with torch.no_grad():
                    if not args.precompute_image_embeddings:
                        pixel_values = img.to(dtype=weight_dtype).to(accelerator.device)
                        pixel_values = pixel_values.unsqueeze(2)
                        pixel_latents = vae.encode(pixel_values).latent_dist.sample()
                    else:
                        pixel_latents = img.to(dtype=weight_dtype).to(accelerator.device)

                    pixel_latents = pixel_latents.permute(0, 2, 1, 3, 4)
                    control_img = control_img.permute(0, 2, 1, 3, 4)
                    latents_mean = (
                        torch.tensor(vae.config.latents_mean)
                        .view(1, 1, vae.config.z_dim, 1, 1)
                        .to(pixel_latents.device, pixel_latents.dtype) )
                    latents_std = 1.0 / torch.tensor(vae.config.latents_std).view(1, 1, vae.config.z_dim, 1, 1).to(
                        pixel_latents.device, pixel_latents.dtype )
                    pixel_latents = (pixel_latents - latents_mean) * latents_std
                    control_img = (control_img - latents_mean) * latents_std

                    bsz = pixel_latents.shape[0]
                    noise = torch.randn_like(pixel_latents, device=accelerator.device, dtype=weight_dtype)
                    u = compute_density_for_timestep_sampling(
                        weighting_scheme="none",
                        batch_size=bsz,
                        logit_mean=0.0,
                        logit_std=1.0,
                        mode_scale=1.29, )
                    indices = (u * noise_scheduler_copy.config.num_train_timesteps).long()
                    timesteps = noise_scheduler_copy.timesteps[indices].to(device=pixel_latents.device)

                sigmas = get_sigmas(timesteps, n_dim=pixel_latents.ndim, dtype=pixel_latents.dtype)
                noisy_model_input = (1.0 - sigmas) * pixel_latents + sigmas * noise

                # Pack latents
                packed_noisy_model_input = QwenImageEditPlusPipeline._pack_latents(
                    noisy_model_input,
                    bsz, 
                    noisy_model_input.shape[2],
                    noisy_model_input.shape[3],
                    noisy_model_input.shape[4],)
                packed_control_img = QwenImageEditPlusPipeline._pack_latents(
                    control_img,
                    bsz, 
                    control_img.shape[2],
                    control_img.shape[3],
                    control_img.shape[4], )

                # latent image ids for RoPE
                img_shapes = [[(1, noisy_model_input.shape[3] // 2, noisy_model_input.shape[4] // 2),
                              (1, control_img.shape[3] // 2, control_img.shape[4] // 2)]] * bsz
                packed_noisy_model_input_concated = torch.cat([packed_noisy_model_input, packed_control_img], dim=1)

                with torch.no_grad():
                    if not args.precompute_text_embeddings:
                        prompt_embeds, prompt_embeds_mask = text_encoding_pipeline.encode_prompt(
                            prompt=prompts,
                            device=packed_noisy_model_input.device,
                            num_images_per_prompt=1,
                            max_sequence_length=1024,)

                # Forward pass
                model_pred = flux_transformer(
                    hidden_states=packed_noisy_model_input_concated,
                    timestep=timesteps / 1000,
                    guidance=None,
                    encoder_hidden_states_mask=prompt_embeds_mask,
                    encoder_hidden_states=prompt_embeds,
                    img_shapes=img_shapes,
                    return_dict=False,)[0]
                model_pred = model_pred[:, : packed_noisy_model_input.size(1)]

                model_pred = QwenImageEditPlusPipeline._unpack_latents(
                    model_pred,
                    height=noisy_model_input.shape[3] * vae_scale_factor,
                    width=noisy_model_input.shape[4] * vae_scale_factor,
                    vae_scale_factor=vae_scale_factor,)
                weighting = compute_loss_weighting_for_sd3(weighting_scheme="none", sigmas=sigmas)
                
                # Flow-matching loss
                target = noise - pixel_latents
                target = target.permute(0, 2, 1, 3, 4)
                loss = torch.mean(
                    (weighting.float() * (model_pred.float() - target.float()) ** 2).reshape(target.shape[0], -1),
                    1, )
                loss = loss.mean()

                # Gather loss across processes
                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps

                # Safeguard against NaN or Inf loss values
                if torch.isnan(loss) or torch.isinf(loss):
                    logger.warning(f"[Step {global_step}] Loss is NaN/Inf! Skipping backward pass.")
                    optimizer.zero_grad()
                    continue

                # Backpropagate
                accelerator.backward(loss)

                # Calculate Gradient Norm & Clip
                total_grad_norm = 0.0
                if accelerator.sync_gradients:
                    # Calculate norm manually before clipping for logging
                    for p in flux_transformer.parameters():
                        if p.grad is not None:
                            param_norm = p.grad.detach().data.norm(2)
                            total_grad_norm += param_norm.item() ** 2
                    total_grad_norm = total_grad_norm ** 0.5

                    # Clip gradients
                    accelerator.clip_grad_norm_(flux_transformer.parameters(), args.max_grad_norm)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                # Safely get logging_steps (defaults to 1 if missing from config)
                logging_steps = getattr(args, "logging_steps", 1)

                if global_step % logging_steps == 0:
                    accelerator.log(
                        {
                            "train_loss": loss.detach().item(),
                            "grad_norm": total_grad_norm,
                            "lr": lr_scheduler.get_last_lr()[0],
                        },
                        step=global_step,)
                        
                # 1. Print formatted text log in terminal every 10 steps
                if global_step % 10 == 0 and accelerator.is_main_process:
                    logger.info(
                        f"[Step {global_step}/{args.max_train_steps}] "
                        f"Loss: {loss.detach().item():.4f} | "
                        f"Grad Norm: {total_grad_norm:.4f} | "
                        f"LR: {lr_scheduler.get_last_lr()[0]:.6f}" )

                # 2. Log metrics to WandB / TensorBoard
                accelerator.log({
                    "train_loss": train_loss,
                    "grad_norm": total_grad_norm,
                    "lr": lr_scheduler.get_last_lr()[0],
                }, step=global_step)
                train_loss = 0.0

                # 3. Update tqdm progress bar postfix
                logs = {
                    "loss": f"{loss.detach().item():.4f}",
                    "grad_norm": f"{total_grad_norm:.2f}",
                    "lr": f"{lr_scheduler.get_last_lr()[0]:.2e}", }
                progress_bar.set_postfix(**logs)

                # Save Checkpoints
                if global_step % args.checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(f"Removing old checkpoints: {', '.join(removing_checkpoints)}")
                                for removing_checkpoint in removing_checkpoints:
                                    shutil.rmtree(os.path.join(args.output_dir, removing_checkpoint))

                    save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                    try:
                        os.makedirs(save_path, exist_ok=True)
                    except Exception:
                        pass

                    unwrapped_flux_transformer = unwrap_model(flux_transformer)
                    flux_transformer_lora_state_dict = convert_state_dict_to_diffusers(
                        get_peft_model_state_dict(unwrapped_flux_transformer))

                    QwenImagePipeline.save_lora_weights(
                        save_path,
                        flux_transformer_lora_state_dict,
                        safe_serialization=True,)
                    logger.info(f"Saved state to {save_path}")

            if global_step >= args.max_train_steps:
                break

    accelerator.wait_for_everyone()
    accelerator.end_training()


if __name__ == "__main__":
    main()
