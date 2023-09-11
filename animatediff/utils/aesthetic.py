import copy
from hashlib import md5
import json
import logging
import os
from pathlib import Path
import torch
import torch.nn as nn
import clip
import platform

class AestheticPredictor(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.input_size = input_size
        self.layers = nn.Sequential(
            nn.Linear(self.input_size, 1024),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.Dropout(0.1),
            nn.Linear(64, 16),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.layers(x)

class Aesthetic:

    def __init__(self, device):
        self.fetch_model()

        self.device = device

        self.state_name = "sac+logos+ava1-l14-linearMSE.pth"

        pt_state = torch.load(self.state_name, map_location=torch.device(device=device))

        # CLIP embedding dim is 768 for CLIP ViT L 14
        self.predictor = AestheticPredictor(768)
        self.predictor.load_state_dict(pt_state)
        self.predictor.to(device)
        self.predictor.eval()

        self.clip_model, self.clip_preprocess = clip.load("ViT-L/14", device=device)

    def fetch_model(self):
        state_name = self.state_name
        if not Path(state_name).exists():
            url = f"https://github.com/christophschuhmann/improved-aesthetic-predictor/blob/main/{state_name}?raw=true"
            import requests
            r = requests.get(url)
            with open(state_name, "wb") as f:
                f.write(r.content)


    def get_image_features(self, image):
        image = self.clip_preprocess(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            image_features = self.clip_model.encode_image(image)
            # l2 normalize
            image_features /= image_features.norm(dim=-1, keepdim=True)
        image_features = image_features.cpu().detach().numpy()
        return image_features


    def get_score(self, image):
        image_features = self.get_image_features(image)
        score = self.predictor(torch.from_numpy(image_features).to(self.device).float())
        return score.item()
