from typing import Callable, List, Optional, Union
from torch.utils.data import Dataset

import decord
decord.bridge.set_bridge('torch')

from einops import rearrange

import random
import os
import json
from PIL import Image, ImageFilter
import numpy as np
import cv2
from scipy import ndimage
import tempfile
import ffmpeg

from transformers import CLIPTokenizer


class FramesDataset(Dataset):
    def __init__(
            self,
            samples_dir: str,
            prompt_map_path: Union[str, list[str]],
            width: int = 512,
            height: int = 512,
            video_length: int = 16,
            sample_start_index: int = 0,
            sample_count: int = 1,
            sample_frame_rate: int = 8,
            variance_threshold: int = 50,
            tokenizer: CLIPTokenizer = None,
            mode: str = 'random',
            prompt_prefix = '',
            prompt_postfix = '',
    ):

        print("FramesDataset", "init", width, height, video_length, sample_count, mode)

        self.width = width
        self.height = height
        self.video_length = video_length
        self.sample_count = sample_count
        self.tokenizer = tokenizer
        self.samples_dir = samples_dir
        self.sample_start_index = sample_start_index
        self.sample_frame_rate = sample_frame_rate
        self.variance_threshold = variance_threshold
        self.mode = mode
        self.prompt_prefix = prompt_prefix
        self.prompt_postfix = prompt_postfix

        self.samples = []

        self.prompt_map = None

        with open(prompt_map_path, 'r') as f:
            self.prompt_map = json.loads(f.read())

        self.frames_path = [str(k) for k in self.prompt_map.keys()]

        print("FramesDataset", "init", "frames_path", len(self.frames_path))

    def load(self):
        print("FramesDataset", "load", "samples_dir", self.samples_dir)

        self.samples = []
        for filename in os.listdir(self.samples_dir):
            if 'json' in filename:
                full_path = f"{self.samples_dir}/{filename}"
                with open(full_path, 'r') as f:
                    sample = json.loads(f.read())
                    prompt = "%s %s %s" % (self.prompt_prefix, sample['prompt'], self.prompt_postfix)
                    sample['prompt'] = prompt
                    sample['prompt_ids'] = self.tokenize(sample['prompt'])
                    sample['video_file'] = full_path.replace("json", "mp4")
                    self.samples.append(sample)
        print("FramesDataset", "load", "samples", len(self.samples))

    def tokenize(self, prompt):
        input_ids = self.tokenizer(
            prompt,
            max_length=self.tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        ).input_ids[0]

        return input_ids

    def prepare(self):
        print("FramesDataset", "prepare")

        candidates = []
        for dir_path in self.frames_path:
            candidates = candidates + self.load_key_frames(dir_path)

        print("FramesDataset", "prepare", "candidates", len(candidates))

        self.pick(self.sample_count, candidates)

    def pick(self, count, candidates):
        print("FramesDataset", "pick", count, len(candidates))

        sample_index = self.sample_start_index
        while True:
            if self.mode == 'random':
                key_frame = random.choice(candidates)
            else:
                key_frame = candidates[sample_index]

            print("FramesDataset", "pick", "key_frame", key_frame)

            dir_name = os.path.dirname(key_frame)
            file_name = os.path.basename(key_frame)
            frame_number = int(file_name.split(".")[0])

            sample = []
            for i in range(frame_number, frame_number + self.video_length):
                frame_path = f"{dir_name}/{i}.png"
                frame = Image.open(frame_path)
                frame = frame.resize((self.width, self.height))
                sample.append(np.array(frame))

            sample = np.array(sample)

            print("FramesDataset", "pick", "reading sample", sample.shape)

            if self.mode == 'random' and not self.check(sample):
                print("FramesDataset", "pick", "skip")
                continue

            print("FramesDataset", "pick", "checked")

            prompt = self.get_prompt(key_frame)

            sample_file = f"{self.samples_dir}/{sample_index}.mp4"
            self.write_video(sample, sample_file, self.sample_frame_rate)
            print("FramesDataset", "pick", "sample_file", sample_file)

            meta_file = f"{self.samples_dir}/{sample_index}.json"
            with open(meta_file, 'w') as f:
                f.write(json.dumps({
                    'key_frame': key_frame,
                    'video_file': sample_file,
                    'prompt': prompt,
                }))
            print("FramesDataset", "pick", "meta_file", meta_file)

            sample_index = sample_index + 1
            if sample_index == self.sample_start_index + self.sample_count:
                print("FramesDataset", "pick", "done")
                break

    def write_video(self, frames, video_file, video_fps):
        with tempfile.TemporaryDirectory() as frames_dir:
            for index, frame in enumerate(frames):
                Image.fromarray(frame).save(f"{frames_dir}/{index}.png")

            (ffmpeg
                .input(f"{frames_dir}/%d.png")
                .output(video_file, vcodec='libx264', vf=f"fps={video_fps}")
                .overwrite_output()
                .run())

    def get_prompt(self, key_frame):
        print("FramesDataset", "get_prompt", key_frame)

        dir_name = os.path.dirname(key_frame)
        file_name = os.path.basename(key_frame)
        number = int(file_name.split(".")[0])
        prompt = ""
        if dir_name in self.prompt_map:
            prompt_map = self.prompt_map[dir_name]
            for k in prompt_map:
                if number >= int(k):
                    print("FramesDataset", "get_prompt", k, prompt_map[k])
                    return prompt_map[k]

        print("FramesDataset", "get_prompt", "not found")
        return prompt

    def check(self, sample):
        diffs = []
        for i in range(0, len(sample)-1):
            diffs.append(np.sum(self.blur(sample[i]) - self.blur(sample[i-1])))

        first_diff = diffs[0]
        variance = np.var(diffs)**(1/2)/first_diff * 100
        threshold = self.variance_threshold

        return variance < threshold

    def blur(self, frame):
        image = Image.fromarray(frame)
        image = image.filter(ImageFilter.GaussianBlur(radius=5))
        return np.array(image)

    def load_key_frames(self, dir_path):
        print("FramesDataset", "load_key_frames", dir_path)

        if not os.path.isdir(dir_path):
            raise Exception("Dir not exist")

        def extract_integer(filename):
            return int(filename.split('.')[0])

        candidates = []

        files = sorted(os.listdir(dir_path), key=extract_integer)
        print("FramesDataset", "load_key_frames", "files", len(files))
        count = len(files)
        for index, file_name in enumerate(files):
            file_path = f"{dir_path}/{file_name}"

            if 'png' in file_name and index + self.video_length <= count:
                candidates.append(file_path)

        print("FramesDataset", "load_key_frames", "candidates", len(candidates))
        return candidates

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        meta = self.samples[index]
        vr = decord.VideoReader(meta['video_file'])
        sample_index = list(range(0, len(vr)))[:self.video_length]
        video = vr.get_batch(sample_index)
        video = rearrange(video, "f h w c -> f c h w")
        meta['pixel_values'] = (video / 127.5 - 1.0)
        return meta

if __name__ == "__main__":

    tokenizer = CLIPTokenizer.from_pretrained('runwayml/stable-diffusion-v1-5', subfolder="tokenizer")

    dataset = FramesDataset(
        samples_dir = "test/FramesDataset/samples_dir",
        prompt_map_path = 'test/FramesDataset/prompt_map.json',
        width = 512,
        height = 512,
        video_length = 16,
        sample_count = 1,
        tokenizer = tokenizer,
        variance_threshold  = 40,
    )

    dataset.prepare()
    #dataset.load()
    #print(len(dataset), dataset[0]['key_frame'], dataset[0]['prompt'])

