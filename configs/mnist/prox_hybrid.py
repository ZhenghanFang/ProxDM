from ml_collections import ConfigDict

from configs.mnist.base import get_config as get_base_config


def get_config():
    config = get_base_config()

    config.model_class = "prox"
    config.discretization = "hybrid"  # hybrid or backward

    # t, lambda sampling
    config.train.sample_t_lamb = ConfigDict()
    config.train.sample_t_lamb.name = "step_candidates"
    config.train.sample_t_lamb.step_candidates = (5, 10, 20, 50, 100, 1000)
    config.train.sample_t_lamb.weights = ConfigDict({"type": "log"})
    config.train.sample_t_lamb.t_mode = "discrete"

    # proximal matching
    config.train.pm = ConfigDict()
    config.train.pm.l1_iters = 75_000
    config.train.pm.pm_gamma_start = 1.0  # gamma is zeta in the definition of proximal matching loss in https://arxiv.org/pdf/2507.08956 (see Eq. 7)
    config.train.pm.pm_gamma_decay = 0.5
    config.train.pm.pm_gamma_decay_stages = 2
    config.train.pm.l1_lr = 1e-4
    config.train.pm.pm_lr = 1e-5

    # network predicts proximal (vanilla) or epsilon (reparameterization trick in paper)
    config.model_type = "epsilon"

    # compute loss on proximal (vanilla) or epsilon (loss balancing trick in paper)
    config.train.loss_on = "epsilon"

    config.fid.sample_method = f"euler_maruyama_{config.discretization}"

    return config
