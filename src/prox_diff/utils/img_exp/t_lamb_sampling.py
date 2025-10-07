"""Methods for sampling t and lambda for prox diffusion training.
The model learns prox_{-lamb * ln p_t}.
"""

import logging
import math
import warnings
from functools import partial

import torch

from ...sde import SDE

logger = logging.getLogger(__name__)


def get_sample_t_lamb_func(config):
    """Get the sampling function based on the config."""
    name = config.train.sample_t_lamb.name

    # Get the sampling function
    if name == "step_candidates":
        step_candidates = config.train.sample_t_lamb.step_candidates

        # Parse weights field from config
        raw_weights = config.train.sample_t_lamb.get("weights", None)

        if raw_weights is None:
            weights = None
        else:
            wtype = raw_weights["type"]
            if wtype == "uniform":
                weights = None
            elif wtype == "log":
                weights = [math.log10(x) for x in step_candidates]
            elif wtype == "root_p":
                weights = [
                    float(x) ** (1.0 / raw_weights["p"]) for x in step_candidates
                ]
            elif wtype == "list":
                weights = raw_weights["values"]
            else:
                raise ValueError(
                    f"Unknown weights type: {wtype}, expected 'log', 'root_p', or 'list'."
                )

        logger.debug(f"Step candidates: {step_candidates}")
        logger.debug(f"Computed weights: {weights}")

        sample_t_lamb_func = partial(
            sample_t_lamb_n_steps_candidates,
            candidates=step_candidates,
            weights=weights,
            t_mode=config.train.sample_t_lamb.get("t_mode", None),
            discretization=config.discretization,
        )
    else:
        raise ValueError(f"Unknown t-lamb sampling method: {name}")
    return sample_t_lamb_func


def sample_t_lamb_n_steps_candidates(
    batch_size: int,
    sde: SDE,
    device,
    candidates: list,
    weights: list,
    t_mode: str,
    discretization: str,
):
    """Baseline sampling method. Pre-define a set of candidates of n_steps."""
    n_steps_candidates = torch.tensor(candidates, device=device)
    if weights is None:
        weights = torch.ones(len(n_steps_candidates), device=device)
    else:
        weights = torch.tensor(weights, device=device)
    indices = torch.multinomial(weights, batch_size, replacement=True)
    n_steps = n_steps_candidates[indices]
    delta_t = 1 / n_steps

    t = torch.rand(batch_size, device=device)
    t_mode = t_mode or "discrete"  # default to discrete
    if t_mode == "discrete":
        t = torch.floor(t / delta_t) * delta_t  # t_i, shape: b, quantize by delta_t
        t = torch.clamp(t, max=1 - delta_t)  # cap at 1-delta_t
    elif t_mode == "continuous":
        t_max = 1 - delta_t
        t = t * t_max  # normalize to [0, 1-delta_t]
    else:
        raise ValueError(f"Unknown t sampling mode: {t_mode}")

    t_plus = t + delta_t  # t_i+1, shape: b
    assert (t <= t_plus).all() and (t_plus <= 1).all(), (
        t.min(),
        t.max(),
        t_plus.min(),
        t_plus.max(),
    )
    eff = sde.beta_integral(t, t_plus)  # effecitve step size, gamma in prox diffusion
    if discretization == "hybrid":
        lamb = eff  # lambda in prox diffusion, hybrid backward discretization
    elif discretization in ["backward", "full"]:
        lamb = eff / (
            1 - 0.5 * eff
        )  # lambda in prox diffusion, full backward discretization
        if discretization == "full":
            warnings.warn(
                "discretization=full is deprecated — use backward instead.",
                category=UserWarning,
            )
    else:
        raise ValueError(f"Unknown discretization method: {discretization}")
    meta = {"step_num": n_steps, "discretization": discretization}  # meta data
    return t, lamb, meta
