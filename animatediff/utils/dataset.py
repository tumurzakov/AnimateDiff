import os
import re
import argparse
import sys
import copy
import gc
import hashlib
import importlib
import itertools
import logging
import math
import os
import shutil
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from huggingface_hub import create_repo, model_info, upload_folder
from packaging import version
from PIL import Image, ImageFilter
from PIL.ImageOps import exif_transpose
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import AutoTokenizer, PretrainedConfig
from einops import rearrange
from datetime import datetime
import random

import diffusers
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    DiffusionPipeline,
    StableDiffusionPipeline,
    UNet2DConditionModel,
)
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version, is_wandb_available
from diffusers.utils.import_utils import is_xformers_available

import json

from typing import Callable, List, Optional, Union

import decord
decord.bridge.set_bridge('torch')

import tempfile

from transformers import CLIPTokenizer

def read_meta(file_path):
    meta = {}
    with open(file_path, 'r') as f:
        content = f.read()
        if content != '':
            candidate = json.loads(content)
            if isinstance(candidate, dict) or isinstance(candidate, list):
                meta = candidate
    return meta

class AnimateDiffDataset(Dataset):
    def __init__(
            self,
            samples_dir: str,
            video_length: int,
            tokenizer: CLIPTokenizer = None,
            width: int = -1,
            height: int = -1,
            randomize: bool = True,
    ):
        self.samples_dir = samples_dir
        self.video_length = video_length
        self.tokenizer = tokenizer
        self.width = width
        self.height = height
        self.randomize = randomize

        self.samples = []

        files = [x for x in os.listdir(samples_dir) if 'json' not in x]
        for file_name in files:
            file_path = f"{samples_dir}/{file_name}"
            name, ext = os.path.splitext(file_name)
            meta_path = f"{samples_dir}/{name}.json"
            self.samples.append((file_path, meta_path))

    def tokenize(self, prompt):
        input_ids = self.tokenizer(
            prompt,
            max_length=self.tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        ).input_ids[0]

        return input_ids

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        file_path, meta_path = self.samples[index]
        vr = decord.VideoReader(file_path, width=self.width, height=self.height)

        start = 0
        if self.randomize and len(vr) > self.video_length:
            start = random.randint(0, len(vr) - self.video_length)

        sample_index = list(range(0, len(vr)))[start:start+self.video_length]
        video = vr.get_batch(sample_index)
        video = rearrange(video, "f h w c -> f c h w")

        meta = read_meta(meta_path)

        prompt = ""
        if 'prompt' in meta:
            prompt = meta['prompt']
        elif 'sample' in meta:
            prompt = meta['sample']['prompt']

        example = {}
        example['pixel_values'] = (video / 127.5 - 1.0)
        example["prompt_ids"] = self.tokenize(prompt)
        return example
