import torch
from einops import rearrange

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

