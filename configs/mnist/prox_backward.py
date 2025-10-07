from ml_collections import ConfigDict

from configs.mnist.prox_hybrid import get_config as get_base_config


def get_config():
    config = get_base_config()

    config.discretization = "backward"  # hybrid or backward

    # t, lambda sampling
    config.train.sample_t_lamb.step_candidates = (20, 50, 100, 1000)

    config.fid.sample_method = f"euler_maruyama_{config.discretization}"
    config.fid.steps = (1000, 100, 50, 20)

    return config
