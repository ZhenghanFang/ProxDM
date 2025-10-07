import math
import torch


def get_sinusoidal_embedding(scalars: torch.Tensor, embedding_dim: int):
    """
    Args:
        scalars: shape (b,), expected range [0, 1]
        embedding_dim: int

    Returns:
        shape (b, embedding_dim)

    Adapted from
    get_timestep_embedding in
    https://github.com/hojonathanho/diffusion/blob/master/diffusion_tf/nn.py
    https://github.com/yang-song/score_sde_pytorch/blob/main/models/layers.py
    """
    assert len(scalars.shape) == 1
    scalars = scalars * 1000.0  # scale to [0, 1000], to match the original range
    assert embedding_dim % 2 == 0

    half_dim = embedding_dim // 2
    max_positions = 10000  # magic number 10000 is from transformers
    emb = math.log(max_positions) / (half_dim - 1)
    emb = torch.exp(
        torch.arange(half_dim, dtype=torch.float32, device=scalars.device) * -emb
    )
    emb = scalars.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    assert emb.shape == (scalars.shape[0], embedding_dim)
    return emb
