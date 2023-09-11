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
        print('facenet', 'input', img.shape)
        img_cropped = self.mtcnn(img)[0]
        print('facenet', 'mtcnn', img_cropped.shape if img_cropped != None else 'None')
        if img_cropped == None:
            return None

        img_embedding = self.resnet(img_cropped.unsqueeze(0).to(self.device))
        print('facenet', 'embedding', img_embedding.shape)
        return img_embedding

    def calc_distance(self, e1, e2):
        if e1 == None or e2 == None:
            return None
        return (e1 - e2).norm().item()

    @torch.no_grad()
    def distance(self, image):
        e = self.calc_embedding(image)
        return self.calc_distance(self.reference, e)

    @torch.no_grad()
    def calc_latent_embedding(self, latents, vae):
        latents_r = latents
        if len(latents.shape) == 5:
            latents_r = rearrange(latents, 'b c f h w -> (b f) c h w')

        decoded = vae.decode(latents_r).sample
        decoded = (decoded / 2 + 0.5).clamp(0, 1) * 255
        decoded = rearrange(decoded, 'b c h w -> b h w c')
        return self.calc_embedding(decoded)
