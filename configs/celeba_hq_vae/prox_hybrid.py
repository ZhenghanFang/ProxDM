from ml_collections import ConfigDict

from configs.celeba_hq_vae.default import get_config as get_default_config


def get_config():
    config = get_default_config()

    config.model_class = "prox"
    config.discretization = "hybrid"

    config.train.grad_clip = 1.0
    config.train.sample_every = 10_000
    config.train.sample_n_steps = (5, 10, 20, 50, 100, 1000)

    config.train.sample_t_lamb = ConfigDict()
    config.train.sample_t_lamb.name = "step_candidates"
    config.train.sample_t_lamb.step_candidates = (5, 10, 20, 50, 100, 1000)
    config.train.sample_t_lamb.weights = ConfigDict({"type": "root_p", "p": 3})
    config.train.sample_t_lamb.t_mode = "discrete"
    config.train.sample_t_lamb.discretization = config.discretization

    config.train.pm = ConfigDict()
    config.train.pm.l1_iters = 300_000
    config.train.pm.pm_gamma_start = 2.0
    config.train.pm.pm_gamma_decay = 0.5
    config.train.pm.pm_gamma_decay_stages = 2
    config.train.pm.l1_lr = 1e-4
    config.train.pm.pm_lr = 2e-5

    config.train.prox_model = ConfigDict()
    config.train.prox_model.model_type = "epsilon"

    config.train.prox_trainer = ConfigDict()
    config.train.prox_trainer.loss_on = "epsilon"

    config.sample_method = config.fid.sample_method = (
        f"euler_maruyama_{config.discretization}"
    )

    return config
