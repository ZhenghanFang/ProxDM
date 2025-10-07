import numpy as np


# VP SDE
class SDE:
    """Class to compute the scalar coefficients of VP SDE with linear beta(t).

    The SDE is:
    dX_t = -1/2 * beta(t) * X_t dt + sqrt(beta(t)) dW_t.
    """

    def __init__(self, beta_min: float, beta_max: float):
        self.beta_min = beta_min
        self.beta_max = beta_max

    def beta(self, t: float) -> float:
        """
        beta(t)

        Args:
            t: float, np.ndarray, or torch.Tensor
        Returns:
            Same shape and type as input
        """
        return self.beta_min + (self.beta_max - self.beta_min) * t

    def beta_integral(self, a: float, b: float) -> float:
        """
        Integral of beta(t) from a to b.

        Args:
            a, b: float, np.ndarray, or torch.Tensor
        Returns:
            Same shape and type as input
        """
        return (self.beta(a) + self.beta(b)) * (b - a) / 2

    def alpha(self, t: float) -> float:
        """
        alpha(t) = exp(-int_0^t beta(s) ds)
        """
        return np.exp(-self.beta_integral(0, t))
