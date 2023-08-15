from typing import Callable, List, Optional, Union
from torch.utils.data import Dataset

import random
import os
import json
from PIL import Image, ImageFilter
import numpy as np
import cv2
from scipy import ndimage

from transformers import CLIPTokenizer


class FramesDataset(Dataset):
    def __init__(
            self,
            prompt_map_path: Union[str, list[str]],
            width: int = 512,
            height: int = 512,
            video_length: int = 16,
            sample_count: int = 1,
            tokenizer: CLIPTokenizer = None,
    ):

        print("FramesDataset", "init", width, height, video_length, sample_count)

        self.width = width
        self.height = height
        self.video_length = video_length
        self.sample_count = sample_count
        self.tokenizer = tokenizer

        self.samples = []

        self.prompt_map = None

        with open(prompt_map_path, 'r') as f:
            self.prompt_map = json.loads(f.read())

        self.frames_path = [str(k) for k in self.prompt_map.keys()]

        print("FramesDataset", "init", "frames_path", len(self.frames_path))

    def load(self):

        print("FramesDataset", "load")
        self.samples = []

        candidates = []
        for dir_path in self.frames_path:
            candidates = candidates + self.load_key_frames(dir_path)

        print("FramesDataset", "load", "candidates", len(candidates))

        self.samples = self.pick(self.sample_count, candidates)

        print("FramesDataset", "load", "samples", len(self.samples))

    def pick(self, count, candidates):
        print("FramesDataset", "pick", count, len(candidates))

        samples = []

        recursion_control = self.sample_count * 5

        while True:
            recursion_control = recursion_control - 1

            if recursion_control < 0:
                raise Exception("Your dataset is garbage")

            key_frame = random.choice(candidates)
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

            if not self.check(sample):
                print("FramesDataset", "pick", "skip")
                continue

            prompt = self.get_prompt(key_frame)

            input_ids = self.tokenizer(
                prompt,
                max_length=self.tokenizer.model_max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            ).input_ids[0]

            samples.append({
                'key_frame': key_frame,
                'prompt': prompt,
                'pixel_values': (sample / 127.5 - 1.0),
                'prompt_ids': input_ids,
            })

            if len(samples) == self.sample_count:
                break

        return samples

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
        threshold = 50

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
        return self.sample_count

    def __getitem__(self, index):
        return self.samples[index]

if __name__ == "__main__":

    tokenizer = CLIPTokenizer.from_pretrained('runwayml/stable-diffusion-v1-5', subfolder="tokenizer")

    dataset = FramesDataset(
        prompt_map_path = 'test/FramesDataset/prompt_map.json',
        width = 512,
        height = 512,
        video_length = 16,
        sample_count = 1,
        tokenizer = tokenizer
    )

    dataset.load()
    print(len(dataset), dataset[0]['key_frame'], dataset[0]['prompt'])

