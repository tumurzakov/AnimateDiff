from typing import Callable, List, Optional, Union
from torch.utils.data import Dataset

import random
import os
import json
from PIL import Image, ImageFilter
import numpy as np
import cv2
from scipy import ndimage
import hickle

from transformers import CLIPTokenizer


class FramesDataset(Dataset):
    def __init__(
            self,
            samples_dir: str,
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
        self.samples_dir = samples_dir

        self.samples = []

        self.prompt_map = None

        with open(prompt_map_path, 'r') as f:
            self.prompt_map = json.loads(f.read())

        self.frames_path = [str(k) for k in self.prompt_map.keys()]

        print("FramesDataset", "init", "frames_path", len(self.frames_path))

    def load(self):
        def extract_integer(filename):
            return int(filename.split('.')[0])

        self.samples = sorted(os.listdir(self.samples_dir), key=extract_integer)

    def prepare(self):
        print("FramesDataset", "prepare")

        candidates = []
        for dir_path in self.frames_path:
            candidates = candidates + self.load_key_frames(dir_path)

        print("FramesDataset", "prepare", "candidates", len(candidates))

        self.pick(self.sample_count, candidates)

    def pick(self, count, candidates):
        print("FramesDataset", "pick", count, len(candidates))

        sample_index = 0
        while True:
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

            print("FramesDataset", "pick", "reading sample", sample.shape)

            if not self.check(sample):
                print("FramesDataset", "pick", "skip")
                continue

            print("FramesDataset", "pick", "checked")

            prompt = self.get_prompt(key_frame)

            input_ids = self.tokenizer(
                prompt,
                max_length=self.tokenizer.model_max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            ).input_ids[0]

            sample = {
                'key_frame': key_frame,
                'prompt': prompt,
                'pixel_values': (sample / 127.5 - 1.0),
                'prompt_ids': input_ids,
            }

            sample_file = f"{self.samples_dir}/{sample_index}.pkl"
            with open(sample_file, 'wb') as f:
                print("FramesDataset", "pick", "sample_file", sample_file)
                hickle.dump(sample, f)

            sample_index = sample_index + 1
            if sample_index == self.sample_count:
                print("FramesDataset", "pick", "done")
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
        pkl = self.samples[index]
        with open(pkl, 'rb') as f:
            return hickle.load(f)

if __name__ == "__main__":

    tokenizer = CLIPTokenizer.from_pretrained('runwayml/stable-diffusion-v1-5', subfolder="tokenizer")

    dataset = FramesDataset(
        samples_dir = "test/FramesDataset/samples_dir",
        prompt_map_path = 'test/FramesDataset/prompt_map.json',
        width = 512,
        height = 512,
        video_length = 16,
        sample_count = 1,
        tokenizer = tokenizer
    )

    dataset.load()
    print(len(dataset), dataset[0]['key_frame'], dataset[0]['prompt'])

