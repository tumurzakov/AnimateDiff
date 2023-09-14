import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
from einops import rearrange

class Facenet:

    def __init__(self, reference, resolution, device):
        self.device = device
        self.mtcnn = MTCNN(image_size=resolution, device=device)
        self.resnet = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)
        self.reference = self.calc_embedding(reference)

    def calc_embedding(self, img):
        img_cropped = self.mtcnn(img)
        if img_cropped == None:
            return None

        img_embedding = self.resnet(img_cropped.unsqueeze(0).to(self.device))
        return img_embedding

    def calc_distance(self, e1, e2):
        if e1 == None or e2 == None:
            return None
        return (e1 - e2).norm()

    @torch.no_grad()
    def get_distance(self, image):
        images = [image]
        if isinstance(image, list):
            images = image

        distances = []
        for im in images:
            e = self.calc_embedding(im)
            d = self.calc_distance(self.reference, e)
            if d is None:
                d = 1
            distances.append(d)

        d = torch.stack(distances)
        avg = torch.mean(d.float(), dim=0)

        return avg

    @torch.no_grad()
    def calc_latent_embedding(self, latents, vae):
        latents_r = latents
        if len(latents.shape) == 5:
            latents_r = rearrange(latents, 'b c f h w -> (b f) c h w')

        decoded = vae.decode(latents_r).sample
        decoded = (decoded / 2 + 0.5).clamp(0, 1) * 255
        decoded = rearrange(decoded, 'b c h w -> b h w c')
        return self.calc_embedding(decoded)
