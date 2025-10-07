"""Train score for a few dedicated steps."""

from ml_collections import ConfigDict

from configs.cifar10.score import get_config as get_base_config


def get_config():
    config = get_base_config()
    config.train.sample_t.name = "step_candidates"
    config.train.sample_t.step_candidates = (5, 10, 20)
    config.fid.steps = (5, 10, 20)
    return config
