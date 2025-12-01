import torch
from diffusers.models.autoencoders import AutoencoderKL
from torch import nn


class VAEWrapper(nn.Module):
    """
    - encode method converts images to latents
    - decode method converts latents back to images.
    """

    def __init__(self, model_id):
        super().__init__()
        self.vae = AutoencoderKL.from_pretrained(model_id)
        self.vae.eval()
        self.vae.requires_grad_(False)

        self.scaling_factor = self.vae.config.scaling_factor
        self.shift_factor = self.vae.config.shift_factor or 0.0

    @torch.no_grad()
    def encode(self, images):
        """
        images: [B, C, H, W], normalized to [-1, 1]
        Returns: [B, latent_dim, H', W']

        Ref: https://github.com/haoningwu3639/SimpleSDM-3/blob/main/train.py#L266
        """
        vae = self.vae
        latents = vae.encode(images).latent_dist.sample()
        latents = (latents - self.shift_factor) * self.scaling_factor
        return latents

    @torch.no_grad()
    def decode(self, latents):
        """
        latents: [B, latent_dim, H', W']
        Returns: [B, C, H, W], in [-1, 1]

        Ref:
        huggingface diffusers.StableDiffusion3Pipeline
        https://github.com/huggingface/diffusers/blob/18c8f10f20f398a20de2d9dc34c8bc381dd6cc69/src/diffusers/pipelines/stable_diffusion_3/pipeline_stable_diffusion_3.py#L1129
        """
        vae = self.vae
        latents = (latents / self.scaling_factor) + self.shift_factor
        images = vae.decode(latents, return_dict=False)[0]
        return images


def get_vae(model_id="stabilityai/sd-vae-ft-ema"):
    vae = VAEWrapper(model_id)
    return vae
