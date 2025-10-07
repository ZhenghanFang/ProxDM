"""Use scipy.optimize to compute prox for 2D GMM."""

import numpy as np
import scipy.optimize


def gmm_neg_log_density(u, pis, mus, sigmas):
    """
    Compute negative log density of a 2D Gaussian mixture model.

    Args:
        u (np.array): Point at which to evaluate density, shape (2,)
        pis (np.array): Mixture weights, shape (n,)
        mus (np.array): Component means, shape (n, 2)
        sigmas (np.array): Component standard deviations, shape (n,)

    Returns:
        float: Negative log density at point u
    """
    n_components = len(pis)
    densities = np.zeros(n_components)

    for i in range(n_components):
        # Compute log density for each component
        diff = u - mus[i]
        log_density = (
            -0.5 * np.sum(diff**2) / sigmas[i] ** 2
            - np.log(2 * np.pi * sigmas[i] ** 2)
            + np.log(pis[i])
        )
        densities[i] = log_density

    # Use log-sum-exp trick for numerical stability
    max_density = np.max(densities)
    log_sum = max_density + np.log(np.sum(np.exp(densities - max_density)))

    return -log_sum


def objective(u, x, lamb, pis, mus, sigmas):
    """
    Compute the full objective: 1/2||u-x||_2^2 + λf(u)

    Args:
        u (np.array): Point at which to evaluate, shape (2,)
        x (np.array): Target point, shape (2,)
        lamb (float): Regularization parameter
        pis, mus, sigmas: GMM parameters

    Returns:
        float: Objective value
    """
    squared_dist = 0.5 * np.sum((u - x) ** 2)
    neg_log_density = gmm_neg_log_density(u, pis, mus, sigmas)
    return squared_dist + lamb * neg_log_density


def compute_prox(x, lamb, pis, mus, sigmas, method="L-BFGS-B"):
    """
    Compute the proximal operator for GMM negative log density.

    Args:
        x (np.array): Point at which to compute proximal operator, shape (2,)
        lamb (float): Regularization parameter
        pis (np.array): Mixture weights, shape (n,)
        mus (np.array): Component means, shape (n, 2)
        sigmas (np.array): Component standard deviations, shape (n,)
        method (str): Optimization method to use

    Returns:
        np.array: Minimizer u*, shape (2,)
    """
    # Use x as initial guess
    result = scipy.optimize.minimize(
        objective, x, args=(x, lamb, pis, mus, sigmas), method=method
    )

    return result.x


if __name__ == "__main__":
    # Example usage:
    # Generate example GMM parameters
    n_components = 3
    pis = np.array([0.3, 0.3, 0.4])
    mus = np.array([[1.0, 1.0], [-1.0, -1.0], [2.0, -2.0]])
    sigmas = np.array([0.5, 0.7, 0.6])

    # Test point
    x = np.array([0.5, 0.5])
    lamb = 1.0

    # Compute proximal operator
    result = compute_prox(x, lamb, pis, mus, sigmas)
    print(f"Input x: {x}")
    print(f"Proximal operator result: {result}")
