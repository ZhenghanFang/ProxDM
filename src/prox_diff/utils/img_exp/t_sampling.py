"""Methods for sampling t for score training."""

import logging
import math
import sys
from functools import partial

import torch

logger = logging.getLogger(__name__)


def get_sample_t_func(config):
    """Get the sampling function based on the config."""
    name = config.train.get("sample_t", {}).get("name", "continuous")  # fallback

    # Get the sampling function
    if name == "continuous":
        sample_t_func = sample_t_continuous
    elif name == "step_candidates":
        step_candidates = config.train.sample_t.step_candidates
        logger.debug(f"Step candidates: {step_candidates}")
        sample_t_func = partial(
            sample_t_n_steps_candidates,
            candidates=step_candidates,
            time_eps=1e-3,  # time epsilon in diffusion sampling
        )
    else:
        raise ValueError(f"Unknown t sampling method: {name}")
    return sample_t_func


def sample_t_continuous(batch_size, device):
    return torch.rand(batch_size, device=device)


def sample_t_n_steps_candidates(
    batch_size: int,
    device,
    candidates: list,
    time_eps: float,
):
    """Pre-define a set of candidates of n_steps."""
    n_steps_candidates = torch.tensor(candidates, device=device)
    weights = torch.ones(len(n_steps_candidates), device=device)
    indices = torch.multinomial(weights, batch_size, replacement=True)
    n_steps = n_steps_candidates[indices]

    ts = []
    for n in n_steps:
        t_vals = torch.linspace(1.0, time_eps, int(n))
        t_sample = t_vals[torch.randint(0, len(t_vals), (1,))]
        ts.append(t_sample)
    t = torch.tensor(ts, device=device)
    # print(t)
    # print(n_steps)
    # print(t.shape)

    return t
