"""Train prox for a few dedicated steps."""

from ml_collections import ConfigDict

from configs.cifar10.prox_hybrid import get_config as get_base_config


def get_config():
    config = get_base_config()
    config.train.sample_t_lamb.step_candidates = (5, 10, 20)
    config.fid.steps = (5, 10, 20)
    return config
