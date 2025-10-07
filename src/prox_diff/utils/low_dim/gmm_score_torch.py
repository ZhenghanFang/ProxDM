"""Compute the score function of an isotropic GMM in n-D using PyTorch."""

import numpy as np
import torch


def compute_score_gmm_isotropic_nd(x, pis, mus, sigmas):
    """
    Compute the score function (gradient of log-density) of an isotropic GMM in n-D.

    Args:
        x (torch.Tensor): Points at which to evaluate the score function.
            Shape (nbatch, dim).
        pis (torch.Tensor): Weights of the GMM components. Shape (ngauss,).
        mus (torch.Tensor): Means of the GMM components. Shape (ngauss, dim).
        sigmas (torch.Tensor): Standard deviations of the GMM components.
            Shape (ngauss,). The covariance matrix of each component is
            sigma^2 * I.

    Returns:
        torch.Tensor: Score function values at the input points.
            Shape (nbatch, dim).
    """
    dim = x.shape[1]

    log_pi_gauss = (
        torch.log(pis)  # ngauss
        - dim / 2 * torch.log(2 * np.pi * sigmas**2)  # ngauss
        - ((x[:, None, :] - mus[None, :, :]) ** 2).sum(dim=2) / (2 * sigmas**2)
    )  # nbatch, ngauss
    normalized_log_pi_gauss = log_pi_gauss - log_pi_gauss.max(dim=1, keepdim=True)[0]
    weights = torch.exp(normalized_log_pi_gauss)  # nbatch, ngauss
    mu_minus_x_over_sigma2 = -(x[:, None, :] - mus[None, :, :]) / (
        sigmas[None, :, None] ** 2
    )  # nbatch, ngauss, dim
    score = (mu_minus_x_over_sigma2 * weights[:, :, None]).sum(dim=1) / weights.sum(
        dim=1
    )[
        :, None
    ]  # nbatch, dim
    return score
