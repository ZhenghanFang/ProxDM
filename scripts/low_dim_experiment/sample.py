"""Sample from a mixture of Dirac delta distributions in 2D using score and prox based methods.
Calculate Wasserstein distance to the target distribution.
"""

import os

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from prox_diff.core import ProxSamplerVP, ScoreSamplerVP
from prox_diff.utils.low_dim.gmm_prox_scipy import compute_prox as compute_prox_numpy
from prox_diff.utils.low_dim.gmm_score_torch import compute_score_gmm_isotropic_nd
from prox_diff.utils.low_dim.wasserstein import wasserstein_distance

# Settings
data_name = "dino"
ddpm_betabar_min = 0.1
ddpm_betabar_max = 20.0
n_steps = [5, 7, 10, 20, 30, 40]
n_samples = 1000

output_root = f"output/low_dim/{data_name}"

# Set seed for reproducibility
seed = 0
torch.manual_seed(seed)
np.random.seed(seed)

###############################################################################
# Define target distribution
###############################################################################
dataset = pd.read_csv("data/datasaurus/datasaurus.csv")
dataset = dataset[dataset["dataset"] == data_name]
manifold_points = dataset[["x", "y"]].values
scale, offset = 50, 50
manifold_points = (manifold_points - offset) / scale
# fig, ax = plt.subplots(figsize=(3, 3))
# ax.scatter(manifold_points[:, 0], manifold_points[:, 1], s=10)
# ax.axis("equal")
# ax.grid(alpha=0.5)
# fig.savefig(f"{output_dir}/delta_points.png")

delta_points = torch.tensor(manifold_points).float()
delta_weights = torch.ones(len(delta_points)) / len(delta_points)
assert torch.allclose(delta_weights.sum(), torch.tensor(1.0))


###############################################################################
# Helper funcs
###############################################################################
# DDPM SDE parameters
def beta_func(t):
    return ddpm_betabar_min + t * (ddpm_betabar_max - ddpm_betabar_min)


def alpha_func(t):
    return torch.exp(
        -(ddpm_betabar_min * t + 0.5 * (ddpm_betabar_max - ddpm_betabar_min) * t**2)
    )


# for score sampler
def ddpm_score_func_delta(x, alpha, points, weights):
    """
    Compute the score for a delta distribution perturbed according to DDPM's VP SDE.

    Args:
        x: torch.Tensor, shape (batch_size, 2)
        alpha: float, the alpha parameter in DDPM
        points: torch.Tensor, shape (n, 2), the points of the delta distribution
        weights: torch.Tensor, shape (n,), the weights of the points

    Returns:
        torch.Tensor, shape (batch_size, 2)
    """
    mus = torch.sqrt(alpha) * points
    sigma = torch.sqrt(1 - alpha)
    # print(mus, sigma)
    sigmas = torch.ones(len(points)) * sigma

    # Compute the score for the GMM
    return compute_score_gmm_isotropic_nd(x, weights, mus, sigmas)


def ddpm_score_func(x, alpha):
    return ddpm_score_func_delta(x, alpha, delta_points, delta_weights)


def score_model(x, t):
    """
    Compute nabla_x log p_t(x) for the target distribution.
    """
    assert torch.all(t == t[0])
    t = t[0]
    alpha_t = alpha_func(t)
    score = ddpm_score_func(x, alpha_t)
    return score


# for prox sampler
def compute_prox_perturbed_vp(x, t, lamb, points, weights):
    """
    Compute the prox operator for a delta distribution perturbed according to DDPM's VP SDE.

    Args:
        x (torch.Tensor): The input tensor. Shape: (batch_size, dim).
        lamb (float): The regularization parameter for the prox.
        alpha (float): The alpha parameter given by VP SDE.
        points (torch.Tensor): The points of the delta distribution. Shape: (n, dim).
        weights (torch.Tensor): The weights of the points. Shape: (n,).

    Returns:
        torch.Tensor: Shape: (batch_size, dim).
    """
    alpha = alpha_func(t)
    mus = torch.sqrt(alpha) * points
    sigma = torch.sqrt(1 - alpha)
    # print(mus, sigma)
    sigmas = torch.ones(len(points)) * sigma

    x = x.numpy().astype("float64")
    if isinstance(lamb, torch.Tensor):
        lamb = lamb.item()
    weights = weights.numpy()
    mus = mus.numpy()
    sigmas = sigmas.numpy()

    y = np.zeros_like(x)
    for i in tqdm(range(x.shape[0])):
        y[i] = compute_prox_numpy(x[i], lamb, weights, mus, sigmas)

    return torch.tensor(y)


def prox_model(x, t, lamb):
    """
    Compute prox_{-lamb * log p_t}(x) for the target distribution.
    """
    assert torch.all(t == t[0]) and torch.all(lamb == lamb[0])
    t = t[0]
    lamb = lamb[0]
    if t > 0:
        return compute_prox_perturbed_vp(x, t, lamb, delta_points, delta_weights)
    else:
        # Projection onto centers
        distances = torch.cdist(x.float(), delta_points)
        closest_center = torch.argmin(distances, dim=1)
        return delta_points[closest_center]


# for metric computation
def compute_w_dist_for_a_trajectory(trajectory, delta_points, delta_weights):

    w1_list = []
    w2_list = []
    for samples in tqdm(trajectory):
        samples_x = np.array(samples)
        samples_y = np.array(delta_points)
        weights_y = np.array(delta_weights)
        w1_list.append(
            wasserstein_distance(
                samples_x, samples_y, weights_y=weights_y, p=1, method="emd"
            )
        )
        w2_list.append(
            wasserstein_distance(
                samples_x, samples_y, weights_y=weights_y, p=2, method="emd"
            )
        )
    return {"w1": w1_list, "w2": w2_list}


###############################################################################
# Initial samples
###############################################################################
# Initial state (e.g., sampled from isotropic Gaussian distribution)
device = "cpu"
dim = 2
x_init = torch.randn(n_samples, dim, device=device, dtype=torch.float64)

###############################################################################
# Loop over n_steps
###############################################################################
n_steps_list = n_steps
for n_steps in n_steps_list:
    print(f"Sampling with n_steps = {n_steps}...")
    output_dir = f"{output_root}/steps_{n_steps}"
    os.makedirs(output_dir, exist_ok=True)

    ###############################################################################
    # Results saver
    ###############################################################################
    results = {}

    ###############################################################################
    # Score based samplers
    ###############################################################################
    # canonical euler-maruyama
    sampler = ScoreSamplerVP(
        model=score_model,
        beta_min=ddpm_betabar_min,
        beta_max=ddpm_betabar_max,
        n_steps=n_steps,
        time_eps=0.0,
        last_noise=True,
    )
    x_T = x_init.clone()
    with torch.no_grad():
        *x_ts, x_0 = sampler.euler_maruyama(x_T)
    results["score, canonical Euler Maruyama"] = {
        "trajectory": torch.stack([x_T] + x_ts + [x_0], dim=0)
    }

    # denoise
    for sampling_eps in [1e-3, 1e-5]:
        sampler = ScoreSamplerVP(
            model=score_model,
            beta_min=ddpm_betabar_min,
            beta_max=ddpm_betabar_max,
            n_steps=n_steps,
            time_eps=sampling_eps,
            last_noise=False,
        )
        x_T = x_init.clone()
        with torch.no_grad():
            *x_ts, x_0 = sampler.euler_maruyama_eps(x_T)
        results[f"score, denoise, eps={sampling_eps}"] = {
            "trajectory": torch.stack([x_T] + x_ts + [x_0], dim=0)
        }

    ###############################################################################
    # Prox based samplers
    ###############################################################################
    # Hybrid
    sampling_eps = 0.0
    sampler = ProxSamplerVP(
        model=prox_model,
        beta_min=ddpm_betabar_min,
        beta_max=ddpm_betabar_max,
        n_steps=n_steps,
        time_eps=sampling_eps,
    ).to(device)
    x_T = x_init.clone()
    with torch.no_grad():
        *x_ts, x_0 = sampler.euler_maruyama_hybrid(x_T)
    results["prox, hybrid"] = {"trajectory": torch.stack([x_T] + x_ts + [x_0], dim=0)}

    # Full
    # calculate largest effective step size gamma_k
    largest_eff = (
        (
            ddpm_betabar_max
            - (ddpm_betabar_max - ddpm_betabar_min) * (1 / n_steps)
            + ddpm_betabar_max
        )
        / 2
        * (1 / n_steps)
    )
    print(f"largest effective step size: {largest_eff}")
    if largest_eff < 2:  # only apply full backward if stable (gamma_k < 2)
        sampling_eps = 0.0
        sampler = ProxSamplerVP(
            model=prox_model,
            beta_min=ddpm_betabar_min,
            beta_max=ddpm_betabar_max,
            n_steps=n_steps,
            time_eps=sampling_eps,
        ).to(device)
        x_T = x_init.clone()
        with torch.no_grad():
            *x_ts, x_0 = sampler.euler_maruyama_backward(x_T)
        results["prox, full"] = {"trajectory": torch.stack([x_T] + x_ts + [x_0], dim=0)}

    ###############################################################################
    # Compute metrics
    ###############################################################################
    methods = list(results.keys())
    for method in methods:
        trajectory = results[method]["trajectory"]
        w_dist = compute_w_dist_for_a_trajectory(
            trajectory, delta_points, delta_weights
        )
        results[method]["w_dist"] = w_dist

    ###############################################################################
    # Save results
    ###############################################################################
    torch.save(
        {
            "results": results,
            "target": {"points": delta_points, "weights": delta_weights},
            "meta": {
                "n_steps": n_steps,
                "n_samples": n_samples,
                "beta_min": ddpm_betabar_min,
                "beta_max": ddpm_betabar_max,
            },
        },
        output_dir + "/results.pt",
    )
