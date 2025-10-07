import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from tqdm import tqdm as tqdm_base

from .sde import SDE


def tqdm(*args, **kwargs):
    return tqdm_base(*args, dynamic_ncols=True, **kwargs)


###################
# Prox matching loss
###################
class ProximalMatchingLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input: torch.Tensor, target: torch.Tensor, gamma) -> torch.Tensor:
        """
        Args:
            input: batch, *
            target: batch, *
            gamma: float or torch.Tensor (batch,)
        Returns:
            loss: (batch,)
        """
        bsize = input.shape[0]
        mse = (input - target).pow(2).reshape(bsize, -1).mean(1)
        return 1 - torch.exp(-(mse / (gamma**2)))


##########################################
# Samplers
##########################################
class ScoreSamplerVP(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        beta_min: float,
        beta_max: float,
        n_steps: int = None,  # number of steps
        time_eps=1e-3,  # integrate the SDE to this to avoid numerical issues, see similar setting in score_sde_pytorch repo, sampling.py, get_pc_sampler, eps argument
        last_noise=False,  # whether to add noise in the last step
        sample_method=None,
        data_dim=None,
    ):
        """
        Args:
            model: torch.nn.Module, predictor of the score.
                Input: (x, t)
                    x: shape: [b, *data], torch.Tensor
                    t: shape: [b], torch.Tensor
                Output: torch.Tensor, same shape and dtype as x
        """
        super().__init__()

        self.model = model
        self.n_steps = n_steps
        self.time_eps = time_eps
        self.sde = SDE(beta_min, beta_max)
        self.last_noise = last_noise
        self.sample_method = sample_method
        self.data_dim = data_dim

    @property
    def N(self):
        return self.n_steps

    def forward(self, n: int, generator: torch.Generator = None):
        """
        n: number of samples
        Output shape: [b, *data]
        """
        x_T = torch.randn(
            n,
            *self.data_dim,
            device=next(self.model.parameters()).device,
            generator=generator,
        )
        yield x_T
        with torch.no_grad():
            if self.sample_method == "euler_maruyama":
                yield from self.euler_maruyama(x_T, generator)
            elif self.sample_method == "euler_maruyama_eps":
                yield from self.euler_maruyama_eps(x_T, generator)
            elif self.sample_method == "ode_euler_eps":
                yield from self.ode_euler_eps(x_T)
            else:
                raise ValueError(f"Unknown sample_method: {self.sample_method}")

    def euler_maruyama(self, x_T: torch.Tensor, generator: torch.Generator = None):
        """
        Canonical Euler-Maruyama discretization.
        Input and output shape: [b, *data]
        """
        x = x_T
        times = torch.linspace(self.time_eps, 1.0, self.N + 1)  # discretized time
        for i in tqdm(range(self.N - 1, -1, -1), desc="Sampling"):
            t_now, t_new = times[i + 1], times[i]
            eff = self.sde.beta(t_now) * (
                t_now - t_new
            )  # effecitve step size, beta_i's in DDPM paper

            vec_t_now = x.new_ones(x.shape[0]) * t_now  # shape: [b]
            score = self.model(x, vec_t_now)  # shape: [b, *data]

            # no noise in the last step
            if i == 0 and not self.last_noise:
                noise = 0
            else:
                # noise = torch.randn_like(x)
                noise = torch.empty_like(x).normal_(generator=generator)

            x = (1 + 0.5 * eff) * x + eff * score + np.sqrt(eff) * noise
            yield x

    def euler_maruyama_eps(self, x_T: torch.Tensor, generator: torch.Generator = None):
        """
        Euler-Maruyama discretization with a denoising step at the end (Song's implementation).
        Integrate from 1 to time_eps in N-1 steps
        Then take one more step of size time_eps
        Based on https://github.com/yang-song/score_sde_pytorch/blob/main/sampling.py#L355
        """
        assert self.time_eps > 0.0
        assert self.last_noise == False
        x = x_T

        times = torch.linspace(1.0, self.time_eps, self.N)  # discretized time
        for i in tqdm(range(self.N), desc="Sampling"):
            t_now = times[i]
            if i == self.N - 1:
                dt = self.time_eps
            else:
                dt = (1.0 - self.time_eps) / (self.N - 1)
            eff = self.sde.beta(t_now) * dt  # effecitve step size

            vec_t_now = x.new_ones(x.shape[0]) * t_now  # shape: [b]
            score = self.model(x, vec_t_now)  # shape: [b, *data]

            if i == self.N - 1 and not self.last_noise:
                # no noise in the last step
                noise = 0
            else:
                # noise = torch.randn_like(x)
                noise = torch.empty_like(x).normal_(generator=generator)

            x = (1 + 0.5 * eff) * x + eff * score + torch.sqrt(eff) * noise
            yield x

    def ode_euler_eps(self, x_T: torch.Tensor):
        """
        Euler discretization of Probability Flow ODE.
        Integrate from 1 to time_eps in N-1 steps
        Then take one more step of size time_eps
        Based on https://github.com/yang-song/score_sde_pytorch/blob/main/sampling.py#L355, with probability_flow=True
        """
        assert self.time_eps > 0.0

        x = x_T
        times = torch.linspace(1.0, self.time_eps, self.N)
        for i in tqdm(range(self.N), desc="Sampling"):
            t_now = times[i]
            if i == self.N - 1:
                # step size for the last step
                dt = self.time_eps
            else:
                # step size for the first N-1 steps
                dt = (1.0 - self.time_eps) / (self.N - 1)
            eff = self.sde.beta(t_now) * dt  # effecitve step size

            vec_t_now = x.new_ones(x.shape[0]) * t_now
            score = self.model(x, vec_t_now)
            x = x + 0.5 * eff * (x + score)
            yield x


class ProxSamplerVP(nn.Module):
    def __init__(
        self,
        model: nn.Module = None,
        beta_min: float = None,
        beta_max: float = None,
        n_steps: int = None,  # number of steps
        time_eps: float = 0.0,  # Used in score model. Not needed in prox model. Integrate the SDE to this to avoid numerical issues, see similar setting in score_sde_pytorch repo, sampling.py, get_pc_sampler, eps argument
        sample_method: str = None,
        data_dim: tuple[int] = None,
    ):
        """
        Args:
            model: torch.nn.Module, predictor of the prox.
                Input: (x, t, lamb)
                    x: shape: [b, *data], torch.Tensor
                    t: shape: [b], torch.Tensor
                    lamb: shape: [b], torch.Tensor, weight on regularizer in
                        prox
                Output: torch.Tensor, same shape and dtype as x
        """
        super().__init__()

        self.model = model
        self.n_steps = n_steps
        self.time_eps = time_eps
        self.sde = SDE(beta_min, beta_max)
        self.sample_method = sample_method
        self.data_dim = data_dim

    @property
    def N(self):
        return self.n_steps

    def forward(self, n: int, generator: torch.Generator = None):
        """
        n: number of samples
        Output shape: [b, *data]
        """
        x_T = torch.randn(
            n,
            *self.data_dim,
            device=next(self.model.parameters()).device,
            generator=generator,
        )
        yield x_T
        with torch.no_grad():
            if self.sample_method == "euler_maruyama_hybrid":
                yield from self.euler_maruyama_hybrid(x_T, generator)
            elif self.sample_method == "euler_maruyama_backward":
                yield from self.euler_maruyama_backward(x_T, generator)
            else:
                raise ValueError(f"Unknown sample_method: {self.sample_method}")

    def euler_maruyama_backward(
        self, x_T: torch.Tensor, generator: torch.Generator = None
    ):
        """
        Input and output shape: [b, *data]
        Full backward discretization.
        """
        x = x_T
        times = torch.linspace(self.time_eps, 1.0, self.N + 1)  # discretized time
        for i in tqdm(range(self.N - 1, -1, -1), desc="Sampling"):
            t_now, t_new = times[i + 1], times[i]
            eff = self.sde.beta_integral(
                t_new, t_now
            )  # effecitve step size, beta_i's in DDPM paper, gamma in prox diffusion
            lamb = eff / (1 - 0.5 * eff)  # lambda in prox diffusion

            noise = torch.empty_like(x).normal_(generator=generator)
            y = (x + np.sqrt(eff) * noise) / (1 - 0.5 * eff)

            vec_t_new = t_new * x.new_ones(x.shape[0])
            vec_lamb = lamb * x.new_ones(x.shape[0])
            x = self.model(y, vec_t_new, vec_lamb)
            yield x

    def euler_maruyama_hybrid(
        self, x_T: torch.Tensor, generator: torch.Generator = None
    ):
        """
        Input and output shape: [b, *data]
        """
        x = x_T
        times = torch.linspace(self.time_eps, 1.0, self.N + 1)  # discretized time
        for i in tqdm(range(self.N - 1, -1, -1), desc="Sampling"):
            t_now, t_new = times[i + 1], times[i]
            eff = self.sde.beta_integral(
                t_new, t_now
            )  # effecitve step size, beta_i's in DDPM paper, gamma in prox diffusion
            lamb = eff  # lambda in prox diffusion

            noise = torch.empty_like(x).normal_(generator=generator)
            y = (1 + 0.5 * eff) * x + torch.sqrt(eff) * noise

            vec_t_new = t_new * x.new_ones(x.shape[0])
            vec_lamb = lamb * x.new_ones(x.shape[0])
            x = self.model(y, vec_t_new, vec_lamb)
            yield x


##########################################
# Prox model and trainer
##########################################
class ProxModel(nn.Module):
    """Wraps an epsilon or prox model to compute the prox."""

    def __init__(self, model: nn.Module, model_type: str):
        """
        model: nn.Module, an epsilon or prox model.
        model_type: choose from ['epsilon', 'prox']
            - prox: the network predicts the prox directly
            - epsilon: the network predicts (Id - prox) / sqrt(lambda), see Sec. 3.3 in https://arxiv.org/pdf/2507.08956
        """
        super().__init__()
        self.model = model
        self.model_type = model_type
        assert model_type in ["epsilon", "prox"]

    def forward(
        self, x: torch.Tensor, t: torch.Tensor, lamb: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the prox.
        Args:
            x: shape: [b, *data]
            t: shape: [b]
            lamb: shape: [b]
        Returns:
            shape: [b, *data]
        """
        if self.model_type == "prox":
            return self.model(x, t, lamb)
        elif self.model_type == "epsilon":
            expand_shape = [x.shape[0]] + [1] * (len(x.shape) - 1)
            return x - self.model(x, t, lamb) * torch.sqrt(lamb).view(expand_shape)

    def compute_epsilon(self, x, t, lamb):
        """
        Compute epsilon.
        Data format is the same as forward.
        """
        if self.model_type == "prox":
            expand_shape = [x.shape[0]] + [1] * (len(x.shape) - 1)
            return (x - self.model(x, t, lamb)) / torch.sqrt(lamb).view(expand_shape)
        elif self.model_type == "epsilon":
            return self.model(x, t, lamb)


class ProxTrainerVP(nn.Module):
    def __init__(
        self,
        model: ProxModel,
        beta_min: float,
        beta_max: float,
        loss_on: str,
    ):
        """
        Args:
            loss_on: choose from ['epsilon', 'prox'], compute loss on which
                prediction.
                - prox: compute loss on prox directly
                - epsilon: compute loss on the noise term, see Eq. (8) in https://arxiv.org/pdf/2507.08956
        """
        super().__init__()
        self.model = model
        self.sde = SDE(beta_min, beta_max)
        self.loss_on = loss_on
        assert loss_on in ["epsilon", "prox"]

    @staticmethod
    def sample_t_lamb():
        pass

    def forward(self, x_0: torch.Tensor, loss_params: dict = None):
        """
        Args:
            x_0: shape: [b, *data]
            loss_params: loss parameters, dict, possible keys:
                type: loss type, [mse, l1, prox_match]
                sigma: sigma in prox matching loss
        """
        t, lamb, meta = self.sample_t_lamb(x_0.shape[0], self.sde, x_0.device)
        # print(t, lamb)
        alpha_t = torch.exp(-self.sde.beta_integral(0, t))
        alpha_t = alpha_t.view([x_0.shape[0]] + [1] * (len(x_0.shape) - 1))

        noise_for_x_t = torch.randn_like(x_0)
        x_t = torch.sqrt(alpha_t) * x_0 + torch.sqrt(1 - alpha_t) * noise_for_x_t

        noise_for_prox_matching = torch.randn_like(x_0)
        y = (
            x_t
            + torch.sqrt(lamb).view([x_0.shape[0]] + [1] * (len(x_0.shape) - 1))
            * noise_for_prox_matching
        )

        if self.loss_on == "prox":
            # Compute loss on prox
            pred = self.model(y, t, lamb)
            target = x_t
        elif self.loss_on == "epsilon":
            # Compute loss on epsilon
            pred = self.model.compute_epsilon(y, t, lamb)
            target = noise_for_prox_matching

        if loss_params["type"] == "mse":
            loss = F.mse_loss(pred, target, reduction="none").mean(
                dim=list(range(1, x_t.ndim))
            )
        elif loss_params["type"] == "l1":
            loss = F.l1_loss(pred, target, reduction="none").mean(
                dim=list(range(1, x_t.ndim))
            )
        elif loss_params["type"] == "prox_match":
            loss_fn = ProximalMatchingLoss()
            gamma = loss_params["gamma"]
            loss = loss_fn(pred, target, gamma)
        else:
            raise

        return loss


##########################################
# Score model and trainer
##########################################
class ScoreModel(nn.Module):
    """Wraps an epsilon model to compute the score."""

    def __init__(self, model, beta_min: float, beta_max: float):
        """
        model: nn.Module, an epsilon model.
            Input:
                x: shape: [b, *data]
                t: shape: [b]
            Output:
                Same shape and dtype as x
        """
        super().__init__()
        self.model = model
        self.sde = SDE(beta_min, beta_max)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: shape: [b, *data]
            t: shape: [b]
        Returns:
            shape: [b, *data]
        """
        alpha_t = torch.exp(-self.sde.beta_integral(0, t))
        alpha_t = alpha_t.view(
            [x.shape[0]] + [1] * (len(x.shape) - 1)
        )  # [b, 1, ..., 1]
        eps = self.model(x, t)  # [b, *data]
        return -eps / torch.sqrt(1 - alpha_t)

    def compute_epsilon(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: shape: [b, *data]
            t: shape: [b]
        Returns:
            shape: [b, *data]
        """
        return self.model(x, t)


class ScoreTrainerVP(nn.Module):
    def __init__(self, model: ScoreModel, beta_min: float, beta_max: float):
        super().__init__()
        self.model = model
        self.sde = SDE(beta_min, beta_max)

    def forward(self, x_0: torch.Tensor, loss_params=None):
        """
        Args:
            x_0: shape: [b, *data]
        """
        assert loss_params is None
        t = self.sample_t(x_0.shape[0], x_0.device)  # shape: b
        alpha_t = torch.exp(-self.sde.beta_integral(0, t))
        alpha_t = alpha_t.view([x_0.shape[0]] + [1] * (len(x_0.shape) - 1))

        noise = torch.randn_like(x_0)
        x_t = torch.sqrt(alpha_t) * x_0 + torch.sqrt(1 - alpha_t) * noise
        loss = F.mse_loss(self.model.compute_epsilon(x_t, t), noise, reduction="none")
        return loss
