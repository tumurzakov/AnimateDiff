import argparse
import datetime
import logging
import inspect
import math
import os
from typing import Dict, Optional, Tuple
from omegaconf import OmegaConf
from collections import OrderedDict

import torch
import torch.nn.functional as F
import torch.utils.checkpoint

import diffusers
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from diffusers import AutoencoderKL, DDPMScheduler, DDIMScheduler
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version
from diffusers.utils.import_utils import is_xformers_available
from diffusers.loaders import AttnProcsLayers
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

from animatediff.models.unet import UNet3DConditionModel
from tuneavideo.data.frames_dataset import FramesDataset
from tuneavideo.data.multi_dataset import MultiTuneAVideoDataset
from animatediff.pipelines.pipeline_animation import AnimationPipeline
from tuneavideo.util import save_videos_grid, ddim_inversion
from einops import rearrange, repeat

from diffusers.models.attention_processor import LoRAAttnProcessor
from animatediff.utils.convert_from_ckpt import convert_ldm_unet_checkpoint, convert_ldm_clip_checkpoint, convert_ldm_vae_checkpoint
from safetensors import safe_open

from PIL import Image

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.10.0.dev0")

logger = get_logger(__name__, log_level="INFO")

def load_checkpoint(path, unet, vae, text_encoder):
    if path.endswith(".ckpt"):
        state_dict = torch.load(path)
        unet.load_state_dict(state_dict)

    elif path.endswith(".safetensors"):
        state_dict = {}
        with safe_open(path, framework="pt", device="cpu") as f:
            for key in f.keys():
                state_dict[key] = f.get_tensor(key)

        base_state_dict = state_dict

        # vae
        converted_vae_checkpoint = convert_ldm_vae_checkpoint(base_state_dict, vae.config)
        #vae.load_state_dict(converted_vae_checkpoint)
        # unet
        converted_unet_checkpoint = convert_ldm_unet_checkpoint(base_state_dict, unet.config)
        unet.load_state_dict(converted_unet_checkpoint, strict=False)
        # text_model
        #text_encoder = convert_ldm_clip_checkpoint(base_state_dict)

        return unet, vae, text_encoder

def main(
    pretrained_model_path: str,
    output_dir: str,
    train_data: Dict,
    validation_data: Dict,
    validation_steps: int = 100,
    dreambooth_path: str = None,
    train_whole_module: bool = False,
    trainable_modules: Tuple[str] = (
        "to_q",
    ),
    train_batch_size: int = 1,
    max_train_steps: int = 500,
    learning_rate: float = 3e-5,
    scale_lr: bool = False,
    lr_scheduler: str = "constant",
    lr_warmup_steps: int = 0,
    adam_beta1: float = 0.9,
    adam_beta2: float = 0.999,
    adam_weight_decay: float = 1e-2,
    adam_epsilon: float = 1e-08,
    max_grad_norm: float = 1.0,
    gradient_accumulation_steps: int = 1,
    gradient_checkpointing: bool = True,
    checkpointing_steps: int = 500,
    start_global_step: int = 0,
    resume_from_checkpoint: Optional[str] = None,
    mixed_precision: Optional[str] = "fp16",
    use_optimizer: str = 'AdamW',
    enable_xformers_memory_efficient_attention: bool = True,
    seed: Optional[int] = None,

    motion_module: str = "models/Motion_Module/mm_sd_v15.ckpt",
    inference_config_path: str = "configs/inference/inference.yaml",
    motion_module_pe_multiplier: int = 1,
    dataset_class: str = 'MultiTuneAVideoDataset',

    train_dreambooth: bool = False,
    train_lora: bool = False,
    lora_rank: int = 4,
    lora_resume_from_checkpoint: Optional[str] = None,
    report_to: str = None,
    report_facenet_distance: bool = False,
    report_facenet_reference_path: str = None,
    report_aesthetic_score: bool = False,
):
    *_, config = inspect.getargvalues(inspect.currentframe())

    inference_config = OmegaConf.load(inference_config_path)

    accelerator = Accelerator(
        gradient_accumulation_steps=gradient_accumulation_steps,
        mixed_precision=mixed_precision,
        log_with=report_to,
    )

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if seed is not None:
        set_seed(seed)

    # Handle the output folder creation
    if accelerator.is_main_process:
        # now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        # output_dir = os.path.join(output_dir, now)
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(f"{output_dir}/samples", exist_ok=True)
        os.makedirs(f"{output_dir}/inv_latents", exist_ok=True)
        OmegaConf.save(config, os.path.join(output_dir, 'config.yaml'))

    # Load scheduler, tokenizer and models.
    noise_scheduler = DDPMScheduler.from_pretrained(pretrained_model_path, subfolder="scheduler")
    tokenizer = CLIPTokenizer.from_pretrained(pretrained_model_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(pretrained_model_path, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(pretrained_model_path, subfolder="vae")
    unet = UNet3DConditionModel.from_pretrained_2d(
            pretrained_model_path,
            subfolder="unet",
            unet_additional_kwargs=OmegaConf.to_container(inference_config.unet_additional_kwargs))

    if dreambooth_path != None and dreambooth_path != "":
        unet, vae, text_encoder = load_checkpoint(dreambooth_path, unet, vae, text_encoder)

    if motion_module != None and motion_module != "":
        motion_module_state_dict = torch.load(motion_module, map_location="cpu")

        # Multiply pe weights by multiplier for training more than 24 frames
        if motion_module_pe_multiplier > 1:
            for key in motion_module_state_dict:
              if 'pe' in key:
                t = motion_module_state_dict[key]
                t = repeat(t, "b f d -> b (f m) d", m=motion_module_pe_multiplier)
                motion_module_state_dict[key] = t

        if "global_step" in motion_module_state_dict: func_args.update({"global_step": motion_module_state_dict["global_step"]})
        missing, unexpected = unet.load_state_dict(motion_module_state_dict, strict=False)
        assert len(unexpected) == 0

    if train_lora:
        # now we will add new LoRA weights to the attention layers
        # It's important to realize here how many attention weights will be added and of which sizes
        # The sizes of the attention layers consist only of two different variables:
        # 1) - the "hidden_size", which is increased according to `unet.config.block_out_channels`.
        # 2) - the "cross attention size", which is set to `unet.config.cross_attention_dim`.

        # Let's first see how many attention processors we will have to set.
        # For Stable Diffusion, it should be equal to:
        # - down blocks (2x attention layers) * (2x transformer layers) * (3x down blocks) = 12
        # - mid blocks (2x attention layers) * (1x transformer layers) * (1x mid blocks) = 2
        # - up blocks (2x attention layers) * (3x transformer layers) * (3x down blocks) = 18
        # => 32 layers

        # Set correct lora layers
        lora_attn_procs = {}
        for name in unet.attn_processors.keys():
            cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
            if name.startswith("mid_block"):
                hidden_size = unet.config.block_out_channels[-1]
            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks.")])
                hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks.")])
                hidden_size = unet.config.block_out_channels[block_id]

            lora_attn_procs[name] = LoRAAttnProcessor(
                hidden_size=hidden_size,
                cross_attention_dim=cross_attention_dim,
                rank=lora_rank,
            )

        if lora_resume_from_checkpoint != None:
            unet.load_attn_procs(lora_resume_from_checkpoint)
        else:
            unet.set_attn_processor(lora_attn_procs)

    # Freeze vae and text_encoder
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(train_dreambooth)

    for name, module in unet.named_modules():
        if "motion_modules" in name and (train_whole_module or name.endswith(tuple(trainable_modules))):
            for params in module.parameters():
                params.requires_grad = True

    if train_lora:
        lora_layers = AttnProcsLayers(unet.attn_processors)
        for param in lora_layers.parameters():
            param.requires_grad = True

    if enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    if gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    if scale_lr:
        learning_rate = (
            learning_rate * gradient_accumulation_steps * train_batch_size * accelerator.num_processes
        )

    # Initialize the optimizer
    optimizer = None
    if use_optimizer == 'AdamW8bit':
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
            )

        optimizer_cls = bnb.optim.AdamW8bit
    elif use_optimizer == 'DAdaptation':
        try:
            import dadaptation
        except ImportError:
            raise ImportError(
                "Please install dadaptation. You can do so by running `pip install dadaptation`"
            )

        optimizer = dadaptation.DAdaptAdam(
            unet.parameters(),
            lr=1.,
            weight_decay=adam_weight_decay,
        )
    elif use_optimizer == 'Prodigy':
        try:
            import prodigyopt
        except ImportError:
            raise ImportError(
                "Please install prodigyopt. You can do so by running `pip install prodigyopt`"
            )

        # https://rentry.org/59xed3
        optimizer = prodigyopt.Prodigy(
            unet.parameters(),
            lr=1.,
            decouple=True,
            weight_decay=0.01,
            d_coef=2,
            use_bias_correction=True,
            safeguard_warmup=True,
        )
    else:
        optimizer_cls = torch.optim.AdamW

    if optimizer == None:
        optimizer = optimizer_cls(
            unet.parameters(),
            lr=learning_rate,
            betas=(adam_beta1, adam_beta2),
            weight_decay=adam_weight_decay,
            eps=adam_epsilon,
        )

    # Get the training dataset
    train_dataset = None
    if dataset_class == 'MultiTuneAVideoDataset':
        train_dataset = MultiTuneAVideoDataset(**train_data)

        # Preprocessing the dataset
        train_dataset.prompt_ids = [None] * len(train_dataset.prompt)
        for index, prompt in enumerate(train_dataset.prompt):
            train_dataset.prompt_ids[index] = tokenizer(
                prompt,max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
            ).input_ids[0]
    else:
        train_dataset = FramesDataset(tokenizer=tokenizer, **train_data)
        train_dataset.load()

    # DataLoaders creation:
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=train_batch_size
    )

    # Get the validation pipeline
    validation_pipeline = AnimationPipeline(
        vae=vae, text_encoder=text_encoder, tokenizer=tokenizer, unet=unet,
        scheduler=DDIMScheduler(**OmegaConf.to_container(inference_config.noise_scheduler_kwargs['DDIMScheduler'])),
    )
    validation_pipeline.enable_vae_slicing()
    ddim_inv_scheduler = DDIMScheduler.from_pretrained(pretrained_model_path, subfolder='scheduler')
    ddim_inv_scheduler.set_timesteps(validation_data.num_inv_steps)

    # Scheduler
    lr_scheduler = get_scheduler(
        lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=lr_warmup_steps * gradient_accumulation_steps,
        num_training_steps=max_train_steps * gradient_accumulation_steps,
    )

    # Prepare everything with our `accelerator`.
    unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet, optimizer, train_dataloader, lr_scheduler
    )

    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move text_encode and vae to gpu and cast to weight_dtype
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / gradient_accumulation_steps)
    # Afterwards we recalculate our number of training epochs
    num_train_epochs = math.ceil(max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers("text2video-fine-tune")

    # Train!
    total_batch_size = train_batch_size * accelerator.num_processes * gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {max_train_steps}")
    global_step = 0
    first_epoch = 0

    if start_global_step > 0:
        global_step = start_global_step
        first_epoch = global_step // num_update_steps_per_epoch
        resume_step = global_step % num_update_steps_per_epoch

    # Potentially load in the weights and states from a previous save
    if resume_from_checkpoint:
        if resume_from_checkpoint != "latest":
            path = os.path.basename(resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1]
        accelerator.print(f"Resuming from checkpoint {path}")
        accelerator.load_state(os.path.join(output_dir, path))
        global_step = int(path.split("-")[1])

        first_epoch = global_step // num_update_steps_per_epoch
        resume_step = global_step % num_update_steps_per_epoch

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(global_step, max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")

    facenet = None
    if report_facenet_distance:
        from animatediff.utils.facenet import Facenet
        ref = Image.open(report_facenet_reference_path)
        ref = torch.tensor(np.array(ref)).to(accelerator.device)
        facenet = Facenet(ref, 512, accelerator.device)

    aesthetic = None
    if report_aesthetic_score:
        from animatediff.utils.facenet import Aesthetic
        aesthetic = Aesthetic(accelerator.device)

    for epoch in range(first_epoch, num_train_epochs):
        unet.train()
        train_loss = 0.0
        for step, batch in enumerate(train_dataloader):
            # Skip steps until we reach the resumed step
            if resume_from_checkpoint and epoch == first_epoch and step < resume_step:
                if step % gradient_accumulation_steps == 0:
                    progress_bar.update(1)
                continue

            with accelerator.accumulate(unet):
                # Convert videos to latent space
                pixel_values = batch["pixel_values"].to(weight_dtype)
                video_length = pixel_values.shape[1]
                pixel_values = rearrange(pixel_values, "b f c h w -> (b f) c h w")
                latents = vae.encode(pixel_values).latent_dist.sample()
                latents = rearrange(latents, "(b f) c h w -> b c f h w", f=video_length)
                latents = latents * 0.18215

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                # Sample a random timestep for each video
                timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (bsz,), device=latents.device)
                timesteps = timesteps.long()

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # Get the text embedding for conditioning
                encoder_hidden_states = text_encoder(batch["prompt_ids"])[0]

                # Get the target for loss depending on the prediction type
                if noise_scheduler.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.prediction_type}")

                # Predict the noise residual and compute loss
                model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(train_batch_size)).mean()
                train_loss += avg_loss.item() / gradient_accumulation_steps

                # Backpropagate
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(unet.parameters(), max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": train_loss}, step=global_step)
                train_loss = 0.0

                if global_step % checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        if train_whole_module or len(trainable_modules) > 0:
                            save_path = os.path.join(output_dir, f"mm-{global_step}.pth")
                            save_mm_checkpoint(unet, save_path)
                            logger.info(f"Saved mm state to {save_path}")

                        if train_lora:
                            save_path = os.path.join(output_dir, f"lora-{global_step}")
                            save_lora_checkpoint(unet, save_path)
                            logger.info(f"Saved lora state to {save_path}")

                        if train_dreambooth:
                            save_path = os.path.join(output_dir, f"db-{global_step}")
                            save_dreambooth_checkpoint(validation_pipeline, save_path)

                if global_step % validation_steps == 0:
                    if accelerator.is_main_process:
                        samples = []
                        generator = torch.Generator(device=latents.device)
                        generator.manual_seed(seed)

                        ddim_inv_latent = None
                        if validation_data.use_inv_latent:
                            inv_latents_path = os.path.join(output_dir, f"inv_latents/ddim_latent-{global_step}.pt")
                            ddim_inv_latent = ddim_inversion(
                                validation_pipeline, ddim_inv_scheduler, video_latent=latents,
                                num_inv_steps=validation_data.num_inv_steps, prompt="")[-1].to(weight_dtype)
                            torch.save(ddim_inv_latent, inv_latents_path)

                        for idx, prompt in enumerate(set(validation_data.prompts)):
                            sample = validation_pipeline(prompt, generator=generator,
                                                         latents=ddim_inv_latent,
                                                         fp16=True,
                                                         **validation_data).videos
                            save_videos_grid(sample, f"{output_dir}/samples/sample-{global_step}/{idx}.gif")
                            samples.append(sample)

                            if report_facenet_distance:
                                distance = facenet.distance(sample[0])
                                tracker.log({"facenet_distance": distance})

                            if report_aesthetic_score:
                                score = aethetic.score(sample[0])
                                tracker.log({"aesthetic_score": score})

                            for tracker in accelerator.trackers:
                                if tracker.name == "wandb":
                                    tracker.log(
                                        {
                                            "validation": [
                                                wandb.Image(image, caption=f"{i}: {args.validation_prompt}")
                                                for i, image in enumerate(sample)
                                            ]
                                        }
                                    )


                        samples = torch.concat(samples)
                        save_path = f"{output_dir}/samples/sample-{global_step}.gif"
                        save_videos_grid(samples, save_path)
                        logger.info(f"Saved samples to {save_path}")

            logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)

            if global_step >= max_train_steps:
                break

    # Create the pipeline using the trained modules and save it.
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unet = accelerator.unwrap_model(unet)
        pipeline = AnimationPipeline.from_pretrained(
            pretrained_model_path,
            text_encoder=text_encoder,
            vae=vae,
            unet=unet,
        )

        if train_whole_module or len(trainable_modules) > 0:
            mm_path = "%s/mm.pth" % output_dir
            save_mm_checkpoint(unet, mm_path)

        if train_lora:
            save_path = os.path.join(output_dir, f"lora")
            save_lora_checkpoint(unet, save_path)

        if train_dreambooth:
            save_path = os.path.join(output_dir, f"db")
            save_dreambooth_checkpoint(validation_pipeline, save_path)


    accelerator.end_training()

def save_mm_checkpoint(unet, mm_path):
    mm_state_dict = OrderedDict()
    state_dict = unet.state_dict()
    for key in state_dict:
        if "motion_module" in key:
            mm_state_dict[key] = state_dict[key]

    torch.save(mm_state_dict, mm_path)

def save_lora_checkpoint(unet, lora_path):
    unet.save_attn_procs(lora_path)

def save_dreambooth_checkpoint(pipeline, output_dir, use_safetensors = False):
    pipeline.save_pretrained(output_dir, safe_serialization=use_safetensors)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./configs/tuneavideo.yaml")
    args = parser.parse_args()

    main(**OmegaConf.load(args.config))
