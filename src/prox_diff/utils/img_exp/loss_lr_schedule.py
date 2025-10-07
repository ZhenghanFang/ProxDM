"""Schedulers for loss and learning rate."""

from typing import Union


def get_step_decay_lr(
    initial_lr: float, it: int, decay_every_iters: int, decay_factor: float
) -> float:
    """
    Returns learning rate at iteration `it` with step decay.

    Args:
        initial_lr (float): initial learning rate
        it (int): current iteration
        decay_every_iters (int): decay interval in iterations
        decay_factor (float): decay factor (e.g., 0.5)

    Returns:
        float: current learning rate
    """
    decay_steps = it // decay_every_iters
    return initial_lr * (decay_factor**decay_steps)


class PMLossSchedule:
    """Loss schedule for the Proximal Matching loss."""

    def __init__(
        self,
        total_iters: int,
        l1_iters: Union[int, float],
        pm_loss_sigma_start: float,
        pm_loss_sigma_decay: float,
        pm_loss_sigma_decay_stages: int,
        l1_lr: float,
        pm_lr: Union[float, dict],
        gamma_scaling: bool,
    ):
        self.total_iters = total_iters
        self.pm_loss_sigma_start = pm_loss_sigma_start
        self.pm_loss_sigma_decay = pm_loss_sigma_decay
        self.pm_loss_sigma_decay_stages = pm_loss_sigma_decay_stages

        if l1_iters <= 1:
            # l1_iters represents the fraction of total iterations
            l1_iters = int(total_iters * l1_iters)
        self.l1_iters = l1_iters
        self.pm_iters = total_iters - self.l1_iters
        self.decay_every = self.pm_iters // pm_loss_sigma_decay_stages

        self.l1_lr = l1_lr
        self.pm_lr = pm_lr

        self.gamma_scaling = gamma_scaling

    def get(self, it: int) -> tuple[dict, float]:
        """Return (loss_params, learning_rate) for given iteration `it`.
        `it` should be in [1, total_iters].
        """
        assert it <= self.total_iters
        if it <= self.l1_iters:
            return {"type": "l1"}, self.l1_lr

        # PM loss
        it_ = it - self.l1_iters - 1  # 0-based counter for pm iterations
        decay = min(it_ // self.decay_every, self.pm_loss_sigma_decay_stages - 1)
        sigma = self.pm_loss_sigma_start * (self.pm_loss_sigma_decay**decay)

        # Get lr for pm
        if isinstance(self.pm_lr, float):
            # Fixed learning rate
            lr = self.pm_lr
        elif isinstance(self.pm_lr, dict):
            if self.pm_lr["schedule"] == "step_decay":
                # Step decay learning rate
                lr = get_step_decay_lr(
                    self.pm_lr["initial_lr"],
                    it_,
                    self.pm_lr["decay_every"],
                    self.pm_lr["decay_factor"],
                )
            else:
                raise ValueError(
                    f"Unknown learning rate schedule: {self.pm_lr['schedule']}"
                )
        else:
            raise ValueError(f"Unknown learning rate type: {type(self.pm_lr)}")

        return {
            "type": "prox_match",
            "gamma": sigma,
            "gamma_scaling": self.gamma_scaling,
        }, lr


class ScoreLossLRSchedule:
    def __init__(self, lr: float, warmup: int):
        self.lr = lr
        self.warmup = warmup

    def warmup_lr(self, it):
        return min(it, self.warmup) / self.warmup

    def get(self, it):
        loss_params = None
        if self.warmup > 0:
            # Warmup learning rate
            lr = self.lr * self.warmup_lr(it)
        else:
            lr = self.lr
        return loss_params, lr
