# Adapted from https://github.com/showlab/Tune-A-Video/blob/main/tuneavideo/pipelines/pipeline_tuneavideo.py

import inspect
import os
from typing import Tuple, Callable, List, Optional, Union, Dict, Any
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from tqdm import tqdm

from diffusers.utils import is_accelerate_available
from packaging import version
from transformers import CLIPTextModel, CLIPTokenizer

from diffusers.configuration_utils import FrozenDict
from diffusers.models import AutoencoderKL, ControlNetModel
from diffusers import DiffusionPipeline
from diffusers.schedulers import (
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,
)
from diffusers.utils import deprecate, logging, BaseOutput
from diffusers.utils.torch_utils import is_compiled_module
from diffusers.pipelines.controlnet import MultiControlNetModel

from ..models.unet import UNet3DConditionOutput

from diffusers.utils import (
    USE_PEFT_BACKEND,
    deprecate,
    logging,
    replace_example_docstring,
    scale_lora_layers,
    unscale_lora_layers,
)

from einops import rearrange, repeat

from ..models.unet import UNet3DConditionModel

from ..utils import overlap_policy
from ..utils.path import get_absolute_path
from ..utils.util import preprocess_image, tensor_hash
from ..utils.adain import adaptive_instance_normalization
from diffusers.loaders import LoraLoaderMixin, TextualInversionLoaderMixin
from diffusers.utils.torch_utils import randn_tensor

from diffusers.image_processor import VaeImageProcessor

from compel import Compel, DiffusersTextualInversionManager
import PIL

import torch.nn.functional as F

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

class MaskedPrompt:
    """
    prompt = MaskedPrompt(prompt, height, width)
    prompt.addMask(mask, "prompt")
    """

    def __init__(self,
            prompt,
            negative_prompt,
            width,
            height,
            embeddings = None,
            controlnet_scale=1.0,
            loras=[],
            mask_latents=[],
            mask_crops=[],
            mask_timesteps=[],
            ):

        mask = torch.ones((1, 1, 1, height//8, width//8))
        self.prompts = [{
            'mask': mask,
            'prompt': prompt,
            'negative_prompt': negative_prompt,
            'embeddings': embeddings,
            'controlnet_scale': controlnet_scale,
            'loras': loras,
            'mask_latents': mask_latents,
            'mask_crops': mask_crops,
            'mask_timesteps': mask_timesteps,
        }]

    def addMask(self,
            mask,
            prompt,
            negative_prompt,
            embeddings = None,
            controlnet_scale=1.0,
            loras=[],
            mask_latents=[],
            mask_crops=[],
            mask_timesteps=[],
            ):

        for i, prev_prompt in enumerate(self.prompts):
            prev_prompt['mask'] = torch.clamp(prev_prompt['mask'] - mask, 0, 1)

        self.prompts.append({
            'mask': mask,
            'prompt': prompt,
            'negative_prompt': negative_prompt,
            'embeddings': embeddings,
            'controlnet_scale': controlnet_scale,
            'loras': loras,
            'mask_latents': mask_latents,
            'mask_crops': mask_crops,
            'mask_timesteps': mask_timesteps,
        })

class MaskedPromptHelper:
    def __init__(self, masked_prompts, device):
        self.prompts = masked_prompts
        self.length = len(self.prompts[0].prompts)
        self.iter = 0
        self.device = device

    def part(self, seq):
        parted = [self.prompts[i] for i in seq]
        return MaskedPromptHelper(parted, self.device)

    def embeddings(self):
        embeddings = []
        for i, frame_prompt in enumerate(self.prompts):
            embeddings.append(frame_prompt.prompts[self.layer]['embeddings'])

        embeddings = torch.stack(embeddings).to(self.device)
        embeddings = rearrange(embeddings, 'f b n c -> (b f) n c')
        return embeddings

    def mask(self):
        masks = []
        for i, frame_prompt in enumerate(self.prompts):
            mask = frame_prompt.prompts[self.layer]['mask']
            if len(mask.shape) == 3:
                mask = mask[i]
            masks.append(mask)

        masks = torch.stack(masks, dim=3).squeeze(0).to(self.device)

        return masks

    def controlnet_scale(self):
        scale = []
        for frame_prompt in self.prompts:
            scale.append(frame_prompt.prompts[self.layer]['controlnet_scale'])

        return scale

    def loras(self):
        loras = []

        for frame_prompt in self.prompts:
            loras.append(frame_prompt.prompts[self.layer]['loras'])

        return []

    def mask_latents(self):
        for frame_prompt in self.prompts:
            if frame_prompt.prompts[self.layer]['mask_latents'] is not None:
                return frame_prompt.prompts[self.layer]['mask_latents']

        return None

    def mask_crops(self):
        for frame_prompt in self.prompts:
            if frame_prompt.prompts[self.layer]['mask_crops'] is not None:
                return frame_prompt.prompts[self.layer]['mask_crops']

        return None

    def mask_timesteps(self):
        for frame_prompt in self.prompts:
            if frame_prompt.prompts[self.layer]['mask_timesteps'] is not None:
                return frame_prompt.prompts[self.layer]['mask_timesteps']

        return None

    def __iter__(self):
        self.layer = 0
        return self

    def __next__(self):
        if self.layer >= self.length:
            raise StopIteration

        embeddings = self.embeddings()
        latent_mask = self.mask()
        controlnet_scale = self.controlnet_scale()
        loras = self.loras()
        mask_latents = self.mask_latents()
        mask_crops = self.mask_crops()
        mask_timesteps = self.mask_timesteps()

        self.layer += 1

        return embeddings, latent_mask, controlnet_scale, loras, mask_latents, mask_crops, mask_timesteps

@dataclass
class AnimationPipelineOutput(BaseOutput):
    videos: Union[torch.Tensor, np.ndarray]

class AnimationPipeline(DiffusionPipeline, TextualInversionLoaderMixin, LoraLoaderMixin):
    _optional_components = []

    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet: UNet3DConditionModel,
        scheduler: Union[
            DDIMScheduler,
            PNDMScheduler,
            LMSDiscreteScheduler,
            EulerDiscreteScheduler,
            EulerAncestralDiscreteScheduler,
            DPMSolverMultistepScheduler,
        ],
        scan_inversions: bool = True,
        controlnet: Union[ControlNetModel, List[ControlNetModel], Tuple[ControlNetModel], MultiControlNetModel] = None,
    ):
        super().__init__()

        if hasattr(scheduler.config, "steps_offset") and scheduler.config.steps_offset != 1:
            deprecation_message = (
                f"The configuration file of this scheduler: {scheduler} is outdated. `steps_offset`"
                f" should be set to 1 instead of {scheduler.config.steps_offset}. Please make sure "
                "to update the config accordingly as leaving `steps_offset` might led to incorrect results"
                " in future versions. If you have downloaded this checkpoint from the Hugging Face Hub,"
                " it would be very nice if you could open a Pull request for the `scheduler/scheduler_config.json`"
                " file"
            )
            deprecate("steps_offset!=1", "1.0.0", deprecation_message, standard_warn=False)
            new_config = dict(scheduler.config)
            new_config["steps_offset"] = 1
            scheduler._internal_dict = FrozenDict(new_config)

        if hasattr(scheduler.config, "clip_sample") and scheduler.config.clip_sample is True:
            deprecation_message = (
                f"The configuration file of this scheduler: {scheduler} has not set the configuration `clip_sample`."
                " `clip_sample` should be set to False in the configuration file. Please make sure to update the"
                " config accordingly as not setting `clip_sample` in the config might lead to incorrect results in"
                " future versions. If you have downloaded this checkpoint from the Hugging Face Hub, it would be very"
                " nice if you could open a Pull request for the `scheduler/scheduler_config.json` file"
            )
            deprecate("clip_sample not set", "1.0.0", deprecation_message, standard_warn=False)
            new_config = dict(scheduler.config)
            new_config["clip_sample"] = False
            scheduler._internal_dict = FrozenDict(new_config)

        is_unet_version_less_0_9_0 = hasattr(unet.config, "_diffusers_version") and version.parse(
            version.parse(unet.config._diffusers_version).base_version
        ) < version.parse("0.9.0.dev0")
        is_unet_sample_size_less_64 = hasattr(unet.config, "sample_size") and unet.config.sample_size < 64
        if is_unet_version_less_0_9_0 and is_unet_sample_size_less_64:
            deprecation_message = (
                "The configuration file of the unet has set the default `sample_size` to smaller than"
                " 64 which seems highly unlikely. If your checkpoint is a fine-tuned version of any of the"
                " following: \n- CompVis/stable-diffusion-v1-4 \n- CompVis/stable-diffusion-v1-3 \n-"
                " CompVis/stable-diffusion-v1-2 \n- CompVis/stable-diffusion-v1-1 \n- runwayml/stable-diffusion-v1-5"
                " \n- runwayml/stable-diffusion-inpainting \n you should change 'sample_size' to 64 in the"
                " configuration file. Please make sure to update the config accordingly as leaving `sample_size=32`"
                " in the config might lead to incorrect results in future versions. If you have downloaded this"
                " checkpoint from the Hugging Face Hub, it would be very nice if you could open a Pull request for"
                " the `unet/config.json` file"
            )
            deprecate("sample_size<64", "1.0.0", deprecation_message, standard_warn=False)
            new_config = dict(unet.config)
            new_config["sample_size"] = 64
            unet._internal_dict = FrozenDict(new_config)

        if isinstance(controlnet, (list, tuple)):
            controlnet = MultiControlNetModel(controlnet)

        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler,
            controlnet=controlnet,
        )
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.embeddings_dir = get_absolute_path('models', 'embeddings')
        self.embeddings_dict = {}
        self.default_tokens = len(self.tokenizer)
        self.scan_inversions = scan_inversions

        textual_inversion_manager = DiffusersTextualInversionManager(self)
        self.compel = Compel(
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            textual_inversion_manager=textual_inversion_manager,
        )

        self.control_image_processor = VaeImageProcessor(
            vae_scale_factor=self.vae_scale_factor, do_convert_rgb=True, do_normalize=False
        )

        self.latent_cache = {}

    def update_embeddings(self):
        if not self.scan_inversions:
            return
        names = [p for p in os.listdir(self.embeddings_dir) if p.endswith('.pt')]
        weight = self.text_encoder.text_model.embeddings.token_embedding.weight
        added_embeddings = []
        for name in names:
            embedding_path = os.path.join(self.embeddings_dir, name)
            embedding = torch.load(embedding_path)
            key = os.path.splitext(name)[0]
            if key in self.tokenizer.encoder:
                idx = self.tokenizer.encoder[key]
            else:
                idx = len(self.tokenizer)
                self.tokenizer.add_tokens([key])
            embedding = embedding['string_to_param']['*']
            if idx not in self.embeddings_dict:
                added_embeddings.append(name)
                self.embeddings_dict[idx] = torch.arange(weight.shape[0], weight.shape[0] + embedding.shape[0])
                weight = torch.cat([weight, embedding.to(weight.device, weight.dtype)], dim=0)
                self.tokenizer.add_tokens([key])
        if added_embeddings:
            self.text_encoder.text_model.embeddings.token_embedding = nn.Embedding(
                weight.shape[0], weight.shape[1], _weight=weight)
            logger.info(f'Added {len(added_embeddings)} embeddings: {added_embeddings}')

    def enable_vae_slicing(self):
        self.vae.enable_slicing()

    def disable_vae_slicing(self):
        self.vae.disable_slicing()

    def enable_sequential_cpu_offload(self, gpu_id=0):
        if is_accelerate_available():
            from accelerate import cpu_offload
        else:
            raise ImportError("Please install accelerate via `pip install accelerate`")

        device = torch.device(f"cuda:{gpu_id}")

        for cpu_offloaded_model in [self.unet, self.text_encoder, self.vae]:
            if cpu_offloaded_model is not None:
                cpu_offload(cpu_offloaded_model, device)


    @property
    def _execution_device(self):
        if self.device != torch.device("meta") or not hasattr(self.unet, "_hf_hook"):
            return self.device
        for module in self.unet.modules():
            if (
                hasattr(module, "_hf_hook")
                and hasattr(module._hf_hook, "execution_device")
                and module._hf_hook.execution_device is not None
            ):
                return torch.device(module._hf_hook.execution_device)
        return self.device

    def insert_inversions(self, ids, attention_mask):
        larger = ids >= self.default_tokens
        for idx in reversed(torch.where(larger)[1]):
            ids = torch.cat([
                ids[:, :idx],
                self.embeddings_dict[ids[:, idx].item()].unsqueeze(0),
                ids[:, idx + 1:],
            ], 1)
            if attention_mask is not None:
                attention_mask = torch.cat([
                    attention_mask[:, :idx],
                    torch.ones(1, 1, dtype=attention_mask.dtype, device=attention_mask.device),
                    attention_mask[:, idx + 1:],
                ], 1)
        if ids.shape[1] > self.tokenizer.model_max_length:
            logger.warning(f"After inserting inversions, the sequence length is larger than the max length. Cutting off"
                           f" {ids.shape[1] - self.tokenizer.model_max_length} tokens.")
            ids = torch.cat([ids[:, :self.tokenizer.model_max_length - 1], ids[:, -1:]], 1)
            if attention_mask is not None:
                attention_mask = attention_mask[:, :self.tokenizer.model_max_length]
        return ids, attention_mask

    def _encode_prompt_compel(self, prompt, device, num_videos_per_prompt, do_classifier_free_guidance, negative_prompt):
        batch_size = len(prompt) if isinstance(prompt, list) else 1

        self.update_embeddings()

        text_embeddings = self.compel(prompt)

        # duplicate text embeddings for each generation per prompt, using mps friendly method
        bs_embed, seq_len, _ = text_embeddings.shape
        text_embeddings = text_embeddings.repeat(1, num_videos_per_prompt, 1)
        text_embeddings = text_embeddings.view(bs_embed * num_videos_per_prompt, seq_len, -1)

        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance:
            uncond_embeddings = self.compel(negative_prompt)

            # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
            seq_len = uncond_embeddings.shape[1]
            uncond_embeddings = uncond_embeddings.repeat(1, num_videos_per_prompt, 1)
            uncond_embeddings = uncond_embeddings.view(batch_size * num_videos_per_prompt, seq_len, -1)

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

        return text_embeddings

    def encode_prompt(
        self,
        prompt,
        device,
        num_images_per_prompt,
        do_classifier_free_guidance,
        negative_prompt=None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        lora_scale: Optional[float] = None,
        clip_skip: Optional[int] = None,
    ):
        num_videos_per_prompt = num_images_per_prompt

        batch_size = len(prompt) if isinstance(prompt, list) else 1

        self.update_embeddings()

        text_embeddings = self.compel(prompt)

        # duplicate text embeddings for each generation per prompt, using mps friendly method
        bs_embed, seq_len, _ = text_embeddings.shape
        text_embeddings = text_embeddings.repeat(1, num_videos_per_prompt, 1)
        text_embeddings = text_embeddings.view(bs_embed * num_videos_per_prompt, seq_len, -1)

        uncond_embeddings = None
        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance:
            uncond_embeddings = self.compel(negative_prompt)

            # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
            seq_len = uncond_embeddings.shape[1]
            uncond_embeddings = uncond_embeddings.repeat(1, num_videos_per_prompt, 1)
            uncond_embeddings = uncond_embeddings.view(batch_size * num_videos_per_prompt, seq_len, -1)

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            # text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

        return text_embeddings, uncond_embeddings

    def _encode_prompt(self,
                       compel,
                       prompt,
                       device,
                       num_videos_per_prompt,
                       do_classifier_free_guidance,
                       negative_prompt,
                       lora_scale: Optional[float] = None,
                       ):
        # set lora scale so that monkey patched LoRA
        # function of text encoder can correctly access it
        if lora_scale is not None and isinstance(self, LoraLoaderMixin):
            self._lora_scale = lora_scale


        if not compel:
          text_embeddings = self._encode_prompt_orig(
              prompt, device, num_videos_per_prompt, do_classifier_free_guidance, negative_prompt
          )

        if compel:
          text_embeddings = self._encode_prompt_compel(
              prompt, device, num_videos_per_prompt, do_classifier_free_guidance, negative_prompt
          )
        return text_embeddings

    def get_tokens(self, prompt):
        text_inputs = self.tokenizer(
            prompt,
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        return text_inputs.input_ids

    def _encode_prompt_orig(self, prompt, device, num_videos_per_prompt, do_classifier_free_guidance, negative_prompt):
        batch_size = len(prompt) if isinstance(prompt, list) else 1

        self.update_embeddings()
        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        untruncated_ids = self.tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(text_input_ids, untruncated_ids):
            removed_text = self.tokenizer.batch_decode(untruncated_ids[:, self.tokenizer.model_max_length - 1 : -1])
            logger.warning(
                "The following part of your input was truncated because CLIP can only handle sequences up to"
                f" {self.tokenizer.model_max_length} tokens: {removed_text}"
            )

        if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
            attention_mask = text_inputs.attention_mask.to(device)
        else:
            attention_mask = None

        text_input_ids, attention_mask = self.insert_inversions(text_input_ids, attention_mask)
        text_embeddings = self.text_encoder(
            text_input_ids.to(device),
            attention_mask=attention_mask,
        )
        text_embeddings = text_embeddings[0]

        # duplicate text embeddings for each generation per prompt, using mps friendly method
        bs_embed, seq_len, _ = text_embeddings.shape
        text_embeddings = text_embeddings.repeat(1, num_videos_per_prompt, 1)
        text_embeddings = text_embeddings.view(bs_embed * num_videos_per_prompt, seq_len, -1)

        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance:
            uncond_tokens: List[str]
            if negative_prompt is None:
                uncond_tokens = [""] * batch_size
            elif type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt]
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )
            else:
                uncond_tokens = negative_prompt

            max_length = text_input_ids.shape[-1]
            uncond_input = self.tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
            )

            if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
                attention_mask = uncond_input.attention_mask.to(device)
            else:
                attention_mask = None

            uncond_input_ids = uncond_input.input_ids
            uncond_input_ids, attention_mask = self.insert_inversions(uncond_input_ids, attention_mask)
            uncond_embeddings = self.text_encoder(
                uncond_input_ids.to(device),
                attention_mask=attention_mask,
            )
            uncond_embeddings = uncond_embeddings[0]

            # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
            seq_len = uncond_embeddings.shape[1]
            uncond_embeddings = uncond_embeddings.repeat(1, num_videos_per_prompt, 1)
            uncond_embeddings = uncond_embeddings.view(batch_size * num_videos_per_prompt, seq_len, -1)

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

        return text_embeddings

    def decode_latents(self, latents):
        video_length = latents.shape[2]
        latents = 1 / 0.18215 * latents
        latents = rearrange(latents, "b c f h w -> (b f) c h w")
        # video = self.vae.decode(latents).sample
        video = []
        device = self._execution_device
        for frame_idx in tqdm(range(latents.shape[0])):
            video.append(self.vae.decode(latents[frame_idx:frame_idx+1].to(device)).sample)
        video = torch.cat(video)
        video = rearrange(video, "(b f) c h w -> b c f h w", f=video_length)
        video = (video / 2 + 0.5).clamp(0, 1)
        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloa16
        video = video.cpu().float().numpy()
        return video

    def prepare_extra_step_kwargs(self, generator, eta):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    def check_inputs(self, prompt, height, width, callback_steps):
        if not isinstance(prompt, str) and not isinstance(prompt, list)  and not isinstance(prompt, dict):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

        if (callback_steps is None) or (
            callback_steps is not None and (not isinstance(callback_steps, int) or callback_steps <= 0)
        ):
            raise ValueError(
                f"`callback_steps` has to be a positive integer but is {callback_steps} of type"
                f" {type(callback_steps)}."
            )

    def check_cnet_inputs(
        self,
        prompt,
        image,
        callback_steps,
        negative_prompt=None,
        controlnet_conditioning_scale=1.0,
        control_guidance_start=0.0,
        control_guidance_end=1.0,
    ):
        # `prompt` needs more sophisticated handling when there are multiple
        # conditionings.
        if isinstance(self.controlnet, MultiControlNetModel):
            if isinstance(prompt, list):
                logger.warning(
                    f"You have {len(self.controlnet.nets)} ControlNets and you have passed {len(prompt)}"
                    " prompts. The conditionings will be fixed across the prompts."
                )

        # Check `image`
        is_compiled = hasattr(F, "scaled_dot_product_attention") and isinstance(
            self.controlnet, torch._dynamo.eval_frame.OptimizedModule
        )
        if (
            isinstance(self.controlnet, ControlNetModel)
            or is_compiled
            and isinstance(self.controlnet._orig_mod, ControlNetModel)
        ):
            self.check_image(image, prompt)
        elif (
            isinstance(self.controlnet, MultiControlNetModel)
            or is_compiled
            and isinstance(self.controlnet._orig_mod, MultiControlNetModel)
        ):
            if not isinstance(image, list):
                raise TypeError("For multiple controlnets: `image` must be type `list`")

            # When `image` is a nested list:
            # (e.g. [[canny_image_1, pose_image_1], [canny_image_2, pose_image_2]])
            elif any(isinstance(i, list) for i in image):
                raise ValueError("A single batch of multiple conditionings are supported at the moment.")
            elif len(image) != len(self.controlnet.nets):
                raise ValueError(
                    f"For multiple controlnets: `image` must have the same length as the number of controlnets, but got {len(image)} images and {len(self.controlnet.nets)} ControlNets."
                )

            for image_ in image:
                self.check_image(image_, prompt)
        else:
            assert False

        # Check `controlnet_conditioning_scale`
        if (
            isinstance(self.controlnet, ControlNetModel)
            or is_compiled
            and isinstance(self.controlnet._orig_mod, ControlNetModel)
        ):
            if not isinstance(controlnet_conditioning_scale, float):
                raise TypeError("For single controlnet: `controlnet_conditioning_scale` must be type `float`.")
        elif (
            isinstance(self.controlnet, MultiControlNetModel)
            or is_compiled
            and isinstance(self.controlnet._orig_mod, MultiControlNetModel)
        ):
            if isinstance(controlnet_conditioning_scale, list):
                if any(isinstance(i, list) for i in controlnet_conditioning_scale):
                    raise ValueError("A single batch of multiple conditionings are supported at the moment.")
            elif isinstance(controlnet_conditioning_scale, list) and len(controlnet_conditioning_scale) != len(
                self.controlnet.nets
            ):
                raise ValueError(
                    "For multiple controlnets: When `controlnet_conditioning_scale` is specified as `list`, it must have"
                    " the same length as the number of controlnets"
                )
        else:
            assert False

    def check_image(self, image, prompt):
        image_is_pil = isinstance(image, PIL.Image.Image)
        image_is_tensor = isinstance(image, torch.Tensor)
        image_is_np = isinstance(image, np.ndarray)
        image_is_pil_list = isinstance(image, list) and isinstance(image[0], PIL.Image.Image)
        image_is_tensor_list = isinstance(image, list) and isinstance(image[0], torch.Tensor)
        image_is_np_list = isinstance(image, list) and isinstance(image[0], np.ndarray)

        if (
            not image_is_pil
            and not image_is_tensor
            and not image_is_np
            and not image_is_pil_list
            and not image_is_tensor_list
            and not image_is_np_list
        ):
            raise TypeError(
                f"image must be passed and be one of PIL image, numpy array, torch tensor, list of PIL images, list of numpy arrays or list of torch tensors, but is {type(image)}"
            )

    def prepare_image(
        self,
        image,
        width,
        height,
        batch_size,
        video_length,
        device,
        dtype,
        do_classifier_free_guidance=False,
        guess_mode=False,
    ):

        images = []
        for image_ in image:
            pil_image = None
            if isinstance(image_, str):
                pil_image = PIL.Image.open(image_)
            elif isinstance(image_, torch.Tensor):
                pil_image = PIL.Image.fromarray(image_.cpu().detach().numpy())
            else:
                raise TypeError('Unknown type')

            processed_image = self.control_image_processor.preprocess(pil_image, height=height, width=width).to(dtype=torch.float32)
            numpy_image = np.array(processed_image)
            tensor_image = torch.tensor(numpy_image).squeeze(0)
            images.append(tensor_image)

        image = torch.stack(images)
        image = image.squeeze(0)

        image = image.to(device=device, dtype=dtype)

        if do_classifier_free_guidance and not guess_mode:
            image = torch.cat([image] * 2)

        return image

    def run_safety_checker(self, image, device, dtype):
        has_nsfw_concept = None
        return image, has_nsfw_concept

    def prepare_latents(self,
            init_image,
            init_image_mask,
            timestep,
            batch_size,
            num_channels_latents,
            video_length,
            temporal_context,
            height,
            width,
            dtype,
            device,
            generator,
            latents=None):

        rand_device = "cpu" if device.type == "mps" else device
        shape = (batch_size, num_channels_latents, video_length, height // self.vae_scale_factor, width // self.vae_scale_factor)

        init_latents = None
        overlap_frames = 0

        if init_image is not None:
            if isinstance(init_image, str):
                init_image = PIL.Image.open(init_image)

            if isinstance(init_image, PIL.Image.Image):
                init_image = preprocess_image(init_image)

            if not isinstance(init_image, (torch.Tensor, PIL.Image.Image, list)):
                raise ValueError(
                    f"`image` has to be of type `torch.Tensor`, `PIL.Image` or list but is {type(image)}"
                )

            if isinstance(init_image, list):
                init_image = [preprocess_image(x).to(self.vae.device) for x in init_image]

                init_latents = []
                for x in init_image:
                    init_latents.append(self.vae.encode(x).latent_dist.sample(generator))

            elif isinstance(init_image, torch.Tensor):
                latent_size=(height // self.vae_scale_factor, width // self.vae_scale_factor)
                if init_image.shape[-2:] == latent_size:
                    latent = init_image[:,:,-1,:,:]
                    init_latents = [latent] * video_length
                    overlap_frames = min(init_image.shape[2], video_length)
                    for i in range(overlap_frames):
                        latent = init_image[:,:,i,:,:]
                        init_latents[i] = latent

                elif init_image.shape[0] > 1:
                    init_latents = []
                    for i in tqdm(range(init_image.shape[0]), desc='encode'):
                        init_image_batch = torch.stack([init_image[i]]).to(self.vae.device)
                        latent = self.vae.encode(init_image_batch).latent_dist.sample(generator)
                        latent = self.vae.config.scaling_factor * latent
                        init_latents.append(latent)

                else:
                    latent = self.vae.encode(init_image.to(self.vae.device)).latent_dist.sample(generator)
                    latent = self.vae.config.scaling_factor * latent
                    init_latents = [latent] * video_length

            init_latents = torch.cat(init_latents, dim=0)
            init_latents = rearrange(init_latents, '(b f) c h w -> b c f h w', f=video_length)

            self.init_latents = init_latents

            if init_image_mask is not None:
                if isinstance(init_image_mask, PIL.Image.Image):
                    init_image_mask = init_image_mask
                    init_image_mask = torch.tensor(np.array(init_image_mask))

                if len(init_image_mask.shape) == 3:
                    init_image_mask = torch.stack([init_image_mask] * video_length)

                init_image_mask = rearrange(init_image_mask, 'f h w c -> f c h w')
                init_image_mask = F.interpolate(
                        init_image_mask,
                        size=(height//8, width//8),
                        mode='nearest-exact').squeeze()

                init_image_mask = init_image_mask.sum(dim=1)
                init_image_mask = (init_image_mask > 100).to(torch.float)
                init_image_mask = init_image_mask.unsqueeze(0)
                init_image_mask = init_image_mask.unsqueeze(0)
                init_image_mask = init_image_mask.to(init_image.device)

            self.init_image_mask = init_image_mask


        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if latents is None:

            if isinstance(generator, list):
                shape = shape
                latents = [
                    torch.randn(shape, generator=generator[i], device=rand_device, dtype=dtype)
                    for i in range(batch_size)
                ]
                latents = torch.cat(latents, dim=0).to(device)
            else:
                noise = torch.randn(shape, generator=generator, device=rand_device, dtype=dtype).to(self.vae.device)
                self.noise = noise

                if init_latents is None:
                    logger.debug("Using random noise")
                    latents = noise
                else:
                    logger.debug("Using init_latents noise")
                    latents = self.scheduler.add_noise(init_latents, noise, timestep)

                    noise = torch.randn(shape, generator=generator, device=rand_device, dtype=dtype).to(self.vae.device)
                    noise_length = video_length - overlap_frames
                    frame_noise_scale = 30
                    for i in range(noise_length):
                        frame = i + overlap_frames
                        init_alpha = float(i) / noise_length / frame_noise_scale
                        latents[:, :, frame, :, :] += (noise[:, :, frame, :, :] * init_alpha)

        else:
            logger.debug("Using latents")
            if latents.shape != shape:
                raise ValueError(f"Unexpected latents shape, got {latents.shape}, expected {shape}")

        # scale the initial noise by the standard deviation required by the scheduler
        if init_latents is None:
            latents = latents * torch.tensor(self.scheduler.init_noise_sigma).to(device)

        return latents

    def get_timesteps(self, num_inference_steps, strength, device):
        # get the original timestep using init_timestep
        init_timestep = min(int(num_inference_steps * strength), num_inference_steps)

        t_start = max(num_inference_steps - init_timestep, 0)
        timesteps = self.scheduler.timesteps[t_start * self.scheduler.order :]

        return timesteps, num_inference_steps - t_start

    def calc_cnet_residuals(
            self,
            i,
            t,
            seq,
            multi_text_embeddings,
            image,
            latent_model_input,
            controlnet_keep,
            controlnet_conditioning_scale,
            guess_mode,
            temporal_context,
            do_classifier_free_guidance):

        down_block_res_samples = None
        mid_block_res_sample = None

        # controlnet(s) inference
        if guess_mode and do_classifier_free_guidance:
            # Infer ControlNet only for the conditional batch.
            control_model_input = latent_model_input
            control_model_input = self.scheduler.scale_model_input(control_model_input, t)
        else:
            control_model_input = latent_model_input

        if isinstance(controlnet_keep[i], list):
            cond_scale = [c * s for c, s in zip(controlnet_conditioning_scale, controlnet_keep[i])]
        else:
            controlnet_cond_scale = controlnet_conditioning_scale
            if isinstance(controlnet_cond_scale, list):
                controlnet_cond_scale = controlnet_cond_scale[0]
            cond_scale = controlnet_cond_scale * controlnet_keep[i]

        controlnet_image = []
        for img in image:
            cnet_image = img[seq]
            if do_classifier_free_guidance and not guess_mode:
                cnet_image = torch.cat([cnet_image]*2)
            controlnet_image.append(cnet_image)

        control_model_input = rearrange(control_model_input, "b c f h w -> (b f) c h w")
        down_block_res_samples, mid_block_res_sample = self.controlnet(
            control_model_input,
            t,
            encoder_hidden_states=multi_text_embeddings,
            controlnet_cond=controlnet_image,
            conditioning_scale=cond_scale,
            guess_mode=guess_mode,
            return_dict=False,
        )

        for down_idx in range(len(down_block_res_samples)):
            down_block_res_samples[down_idx] = rearrange(
                    down_block_res_samples[down_idx],
                    '(b f) c h w -> b c f h w',
                    f=temporal_context)

        mid_block_res_sample = rearrange(
                mid_block_res_sample,
                '(b f) c h w -> b c f h w',
                f=temporal_context)

        if guess_mode and do_classifier_free_guidance:
            # Infered ControlNet only for the conditional batch.
            # To apply the output of ControlNet to both the unconditional and conditional batches,
            # add 0 to the unconditional batch to keep it unchanged.
            down_block_res_samples = [torch.cat([torch.zeros_like(d), d]) for d in down_block_res_samples]
            mid_block_res_sample = torch.cat([torch.zeros_like(mid_block_res_sample), mid_block_res_sample])

        return down_block_res_samples, mid_block_res_sample

    def get_masked_prompts(self,
                           prompt,
                           negative_prompt,
                           video_length,
                           width,
                           height,
                           compel,
                           device,
                           num_videos_per_prompt,
                           do_classifier_free_guidance,
                           text_encoder_lora_scale,
                           controlnet_conditioning_scale,
                           ):

        masked_prompts = [None for x in range(video_length)]

        self.tokens = {}

        for i in range(video_length):
            if isinstance(prompt, str):
                masked_prompts[i] = MaskedPrompt(prompt, negative_prompt, width, height, controlnet_scale=controlnet_conditioning_scale)
            elif isinstance(prompt, MaskedPrompt):
                masked_prompts[i] = prompt
            elif isinstance(prompt, dict):
                for start_frame in prompt:
                    if i >= int(start_frame):
                        part_prompt = prompt[start_frame]
                        if isinstance(part_prompt, str):
                            masked_prompts[i] = MaskedPrompt(part_prompt,
                                    negative_prompt,
                                    width,
                                    height,
                                    controlnet_scale=controlnet_conditioning_scale)

                        elif isinstance(part_prompt, MaskedPrompt):
                            masked_prompts[i] = part_prompt
            else:
                t = str(type(prompt))
                raise TypeError(f"Unknown prompt type {t}")

        for i, masked_prompt in enumerate(masked_prompts):
            if masked_prompt != None:
                for prompt in masked_prompt.prompts:

                    if prompt['prompt'] not in self.tokens:
                        self.tokens[prompt['prompt']] = self.get_tokens(prompt['prompt']).tolist()

                    prompt['embeddings'] = self._encode_prompt(
                        compel,
                        prompt['prompt'], device, num_videos_per_prompt,
                        do_classifier_free_guidance, prompt['negative_prompt'],
                        lora_scale=text_encoder_lora_scale,
                    )

        for t in self.tokens:
            logger.debug("Tokens %s, %s", t, self.tokens[t])

        filled_masked_prompts = []
        last_masked_prompt = None
        for masked_prompt in masked_prompts:
            if masked_prompt != None:
                last_masked_prompt = masked_prompt
            else:
                masked_prompt = last_masked_prompt

            filled_masked_prompts.append(masked_prompt)

        return MaskedPromptHelper(filled_masked_prompts, device)

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]],
        video_length: Optional[int],
        init_image: Union[
            torch.FloatTensor,
            PIL.Image.Image,
            np.ndarray,
            List[torch.FloatTensor],
            List[PIL.Image.Image],
            List[np.ndarray],
        ] = None,
        init_image_mask: Union[
            torch.FloatTensor,
            PIL.Image.Image,
            np.ndarray,
            List[torch.FloatTensor],
            List[PIL.Image.Image],
            List[np.ndarray],
        ] = None,
        image: Union[
            torch.FloatTensor,
            PIL.Image.Image,
            np.ndarray,
            List[torch.FloatTensor],
            List[PIL.Image.Image],
            List[np.ndarray],
        ] = None,
        strength: float = 0.8,
        temporal_context: Optional[int] = None,
        strides: int = 3,
        overlap: int = 4,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        prompt_embeds = None,
        negative_prompt_embeds = None,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_videos_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "tensor",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: Optional[int] = 1,
        seq_policy=overlap_policy.uniform,
        fp16=False,
        compel=False,
        controlnet_conditioning_scale: Union[float, List[float]] = 1.0,
        guess_mode: bool = False,
        control_guidance_start: Union[float, List[float]] = 0.0,
        control_guidance_end: Union[float, List[float]] = 1.0,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,

        read_latent_cache: bool = False,
        latent_cache_size: int = 0,

        latents_freq_filter: bool = False,

        adain_style_in_latents = None,
        adain_style_out_latents = None,

        **kwargs,
    ):
        if self.controlnet:
            controlnet = self.controlnet._orig_mod if is_compiled_module(self.controlnet) else self.controlnet

            # align format for control guidance
            if not isinstance(control_guidance_start, list) and isinstance(control_guidance_end, list):
                control_guidance_start = len(control_guidance_end) * [control_guidance_start]
            elif not isinstance(control_guidance_end, list) and isinstance(control_guidance_start, list):
                control_guidance_end = len(control_guidance_start) * [control_guidance_end]
            elif not isinstance(control_guidance_start, list) and not isinstance(control_guidance_end, list):
                mult = len(controlnet.nets) if isinstance(controlnet, MultiControlNetModel) else 1
                control_guidance_start, control_guidance_end = mult * [control_guidance_start], mult * [
                    control_guidance_end
                ]

        # Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        # Check inputs. Raise error if not correct
        # self.check_inputs(prompt, height, width, callback_steps)

        if self.controlnet is not None:
            self.check_cnet_inputs(
                prompt,
                image,
                callback_steps,
                negative_prompt,
                controlnet_conditioning_scale,
                control_guidance_start,
                control_guidance_end,
            )

        # Define call parameters
        # batch_size = 1 if isinstance(prompt, str) else len(prompt)
        batch_size = 1
        if latents is not None:
            batch_size = latents.shape[0]
        if isinstance(prompt, list):
            batch_size = len(prompt)

        device = self._execution_device
        cpu = torch.device('cpu')
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        if self.controlnet is not None:
            if isinstance(controlnet, MultiControlNetModel) and isinstance(controlnet_conditioning_scale, float):
                controlnet_conditioning_scale = [controlnet_conditioning_scale] * len(controlnet.nets)


            global_pool_conditions = (
                controlnet.config.global_pool_conditions
                if isinstance(controlnet, ControlNetModel)
                else controlnet.nets[0].config.global_pool_conditions
            )
            guess_mode = guess_mode or global_pool_conditions

        text_encoder_lora_scale = (
            cross_attention_kwargs.get("scale", None) if cross_attention_kwargs is not None else None
        )

        masked_prompts = self.get_masked_prompts(
                prompt,
                negative_prompt,
                video_length,
                width,
                height,
                compel,
                device,
                num_videos_per_prompt,
                do_classifier_free_guidance,
                text_encoder_lora_scale,
                controlnet_conditioning_scale,
                )

        if self.controlnet is not None:
            # 4. Prepare image
            if isinstance(controlnet, ControlNetModel):
                image = self.prepare_image(
                    image=image,
                    width=width,
                    height=height,
                    batch_size=batch_size,
                    video_length=video_length,
                    device=device,
                    dtype=controlnet.dtype,
                    do_classifier_free_guidance=do_classifier_free_guidance,
                    guess_mode=guess_mode,
                )
            elif isinstance(controlnet, MultiControlNetModel):
                images = []

                for image_ in image:
                    image_ = self.prepare_image(
                        image=image_,
                        width=width,
                        height=height,
                        batch_size=batch_size,
                        video_length=video_length,
                        device=device,
                        dtype=controlnet.dtype,
                        do_classifier_free_guidance=do_classifier_free_guidance,
                        guess_mode=guess_mode,
                    )

                    images.append(image_)

                image = images
            else:
                assert False

        # Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        if init_image is not None:
            timesteps, num_inference_steps = self.get_timesteps(num_inference_steps, strength, device)

        latent_timestep = timesteps[:1].repeat(batch_size * num_videos_per_prompt)

        # Prepare latent variables
        num_channels_latents = self.unet.in_channels
        self.init_image_mask = None
        latents = self.prepare_latents(
            init_image,
            init_image_mask,
            latent_timestep,
            batch_size * num_videos_per_prompt,
            num_channels_latents,
            video_length,
            temporal_context,
            height,
            width,
            torch.float32,
            generator.device if generator is not None else cpu,
            generator,
            latents,
        )
        latents_dtype = latents.dtype

        latents = latents.to(device)

        if adain_style_in_latents is not None:
            latents = adaptive_instance_normalization(latents, adain_style_in_latents)

        # Prepare extra step kwargs.
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        total = sum(
            len(list(seq_policy(i, num_inference_steps, latents.shape[2], temporal_context, strides, overlap)))
            for i in range(len(timesteps))
        )

        if self.controlnet is not None:
            # 7.1 Create tensor stating which controlnets to keep
            controlnet_keep = []
            for i in range(len(timesteps)):
                keeps = [
                    1.0 - float(i / len(timesteps) < s or (i + 1) / len(timesteps) > e)
                    for s, e in zip(control_guidance_start, control_guidance_end)
                ]
                controlnet_keep.append(keeps[0] if isinstance(controlnet, ControlNetModel) else keeps)

            self.controlnet.to(device, dtype=latents.dtype)

        initial_noise = latents.detach().clone()

        # Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        with self.progress_bar(total=total) as progress_bar:
            desc = []
            if 'desc' in kwargs:
                desc.append(kwargs['desc'])

            if len(desc) > 0:
                progress_bar.set_description(",".join(desc))

            for i, t in enumerate(timesteps):

                if latent_cache_size > 0:
                    latent_cache = latents[:,:,-latent_cache_size:,:,:].clone().cpu()

                if read_latent_cache and latent_cache_size > 0:
                    for cache_latent_idx in range(latent_cache_size):
                        latents[:,:,cache_latent_idx,:,:] = \
                            self.latent_cache[t.item()][:,:,cache_latent_idx,:,:] \
                                .clone().to(self.unet.device)

                if latent_cache_size > 0:
                    self.latent_cache[t.item()] = latent_cache

                noise_pred = torch.zeros((latents.shape[0] * (2 if do_classifier_free_guidance else 1),
                                          *latents.shape[1:]), device=latents.device, dtype=latents_dtype)
                counter = torch.zeros((1, 1, latents.shape[2], 1, 1), device=latents.device, dtype=latents_dtype)
                for seq in seq_policy(i, num_inference_steps, latents.shape[2], temporal_context, strides, overlap):
                    # expand the latents if we are doing classifier free guidance
                    latent_model_input = latents[:, :, seq].to(device)\
                        .repeat(2 if do_classifier_free_guidance else 1, 1, 1, 1, 1)
                    latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                    down_block_res_samples = None
                    mid_block_res_sample = None

                    prompt_idx = 0
                    noise_preds = []
                    with torch.autocast('cuda', enabled=fp16, dtype=torch.float16):

                        parted_prompts = masked_prompts.part(seq)
                        for embeddings, latent_mask, controlnet_scale, loras, mask_latents, mask_crops, mask_timesteps in parted_prompts:

                            if self.controlnet != None and controlnet_scale[0] > 0:
                                down_block_res_samples, mid_block_res_sample = self.calc_cnet_residuals(
                                    i,
                                    t,
                                    seq,
                                    embeddings,
                                    image,
                                    latent_model_input,
                                    controlnet_keep,
                                    [controlnet_scale[0]]*len(self.controlnet.nets),
                                    guess_mode,
                                    min(temporal_context, video_length),
                                    do_classifier_free_guidance)

                            # predict the noise residual
                            pred = self.unet(latent_model_input,
                                             t,
                                             cross_attention_kwargs=cross_attention_kwargs,
                                             down_block_additional_residuals=down_block_res_samples,
                                             mid_block_additional_residual=mid_block_res_sample,
                                             encoder_hidden_states=embeddings)

                            predict = pred.sample.to(dtype=latents_dtype, device=device)
                            predict = predict * latent_mask

                            noise_pred[:, :, seq] += predict
                            prompt_idx += 1

                    counter[:, :, seq] += 1
                    progress_bar.update()

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = (noise_pred / counter).chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

                if self.init_image_mask is not None:
                    init_latents_proper = self.init_latents

                    if i < len(timesteps) - 1:
                        noise_timestep = timesteps[i + 1]
                        init_latents_proper = self.scheduler.add_noise(
                            init_latents_proper, self.noise, torch.tensor([noise_timestep])
                        )

                    latents = (1 - self.init_image_mask) * init_latents_proper + self.init_image_mask * latents

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    if callback is not None and i % callback_steps == 0:
                        callback(i, t, latent_model_input)

        if self.init_image_mask is not None:
            latents = (1 - self.init_image_mask) * self.init_latents + self.init_image_mask * latents

        if adain_style_out_latents is not None:
            latents = adaptive_instance_normalization(latents, adain_style_out_latents)

        if output_type == 'latents':
            return latents

        # Post-processing
        latents = latents.to(self.vae.dtype)
        video = self.decode_latents(latents)

        # Convert to tensor
        if output_type == "tensor":
            video = torch.from_numpy(video)

        if not return_dict:
            return video

        return AnimationPipelineOutput(videos=video)
