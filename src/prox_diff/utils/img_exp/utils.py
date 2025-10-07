import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.utils
from tqdm import tqdm


def plot_samples(samples: np.ndarray, nc=10):
    """
    Plot image samples.
    Args:
        samples: (n_samples, 3 or 1, w, h), normalized to [-1, 1]
        nc: number of columns
    """
    n_samples, _, w, h = samples.shape
    nr = (n_samples + nc - 1) // nc  # Compute number of columns

    samples = samples * 0.5 + 0.5  # Normalize to [0, 1]
    samples = np.clip(samples, 0, 1)  # Clip to [0, 1]

    fig, axes = plt.subplots(nr, nc, figsize=(nc * 2, nr * 2))
    axes = np.array(axes).reshape(nr, nc)  # Ensure it's always a 2D array

    for i, img in enumerate(samples):
        ax = axes.flat[i]
        img = img.transpose(1, 2, 0)  # Convert to (w, h, c)
        if img.shape[2] == 1:
            # Convert grayscale to RGB
            img = np.repeat(img, 3, axis=2)
        ax.imshow(img)

    for ax in axes.flat:
        ax.axis("off")  # Hide axes

    plt.tight_layout()
    return fig, ax


def moving_average(x, w):
    return np.convolve(x, np.ones(w), "valid") / w


def save_ckpt(ckpt_path, **kwargs):
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
    torch.save(kwargs, ckpt_path)


###############################################################################
# EMA implementation based on https://github.com/w86763777/pytorch-ddpm/blob/master/main.py#L58
###############################################################################
def ema(source, target, decay):
    source_dict = source.state_dict()
    target_dict = target.state_dict()
    for key in source_dict.keys():
        target_dict[key].data.copy_(
            target_dict[key].data * decay + source_dict[key].data * (1 - decay)
        )
