import numpy as np
import ot
import scipy.spatial.distance


def wasserstein_distance(x, y, weights_x=None, weights_y=None, p=1, method="emd"):
    """
    Compute Wasserstein-1 distance using POT library with optional weights.

    Parameters:
    x: array of shape (n_samples_x, dim)
    y: array of shape (n_samples_y, dim)
    weights_x: array of shape (n_samples_x,), optional weights for x
    weights_y: array of shape (n_samples_y,), optional weights for y
    p: int, optional, the order of Wasserstein distance
    method: 'emd', 'sinkhorn', or 'sliced'
    """
    n_x = len(x)
    n_y = len(y)

    # Use provided weights or uniform weights
    a = weights_x if weights_x is not None else np.ones(n_x) / n_x
    b = weights_y if weights_y is not None else np.ones(n_y) / n_y

    # Normalize weights to sum to 1
    a = a / np.sum(a)
    b = b / np.sum(b)

    M = scipy.spatial.distance.cdist(x, y, metric="euclidean")
    M = M**p

    if method == "emd":
        return ot.emd2(a, b, M) ** (1 / p)
    elif method == "sinkhorn":
        return ot.sinkhorn2(a, b, M, reg=0.01) ** (1 / p)
    elif method == "sliced":
        return ot.sliced_wasserstein_distance(x, y, n_projections=1000, a=a, b=b, p=p)
    else:
        raise ValueError("Method must be 'emd', 'sinkhorn', or 'sliced'")


if __name__ == "__main__":
    # Example usage
    # samples
    n = 1000
    samples_x = np.random.randn(n, 2)
    samples_y = np.random.randn(n, 2) + np.array([3, 2])
    # true wasserstein distance
    true_wasserstein = np.linalg.norm(np.array([3, 2]))
    print("True Wasserstein distance: ", true_wasserstein)

    d = wasserstein_distance(samples_x, samples_y, method="emd")
    print(f"EMD: {d:.4f}")
    d = wasserstein_distance(samples_x, samples_y, method="sinkhorn")
    print(f"Sinkhorn: {d:.4f}")
    d = wasserstein_distance(samples_x, samples_y, method="sliced")
    print(f"Sliced: {d:.4f}")
