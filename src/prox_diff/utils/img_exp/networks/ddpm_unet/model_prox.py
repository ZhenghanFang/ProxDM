"""Unet models for prox diffusion."""

import torch
from torch import nn

from .model import ScalarEmbedding, UNet


class UNetTimeLamb(nn.Module):
    """Time-Lambda conditioned UNet."""

    def __init__(
        self, ch, ch_mult, attn, num_res_blocks, dropout, in_ch, use_checkpoint=False
    ):
        super().__init__()
        edim = ch * 4  # embedding dimension is 4x feature channel number
        self.time_embedding = ScalarEmbedding(ch, edim // 2)
        self.lamb_embedding = ScalarEmbedding(ch, edim // 2)

        self.unet = UNet(
            ch,
            ch_mult,
            attn,
            num_res_blocks,
            dropout,
            edim,
            in_ch,
            use_checkpoint=use_checkpoint,
        )

    def forward(self, x, t, lamb):
        """
        Args:
            x: (b, *data)
            t: (b,)
            lamb: (b,)
        """
        # Time embedding
        temb = self.time_embedding(t)

        # Lambda embedding
        lemb = self.lamb_embedding(lamb)

        # UNet
        return self.unet(x, torch.cat([temb, lemb], dim=1))


def get_network(net_config):
    """
    Instantiate and return a time-lambda UNet based on config.

    Args:
        net_config: ml_collections.ConfigDict

    Returns:
        nn.Module instance of the selected network
    """
    net_config = net_config.to_dict()
    name = net_config.pop("name", "UNetTimeLamb")  # Default to UNetTimeLamb

    if name == "UNetTimeLamb":
        return UNetTimeLamb(**net_config)
    else:
        raise ValueError(f"Unknown network name: {name}")
