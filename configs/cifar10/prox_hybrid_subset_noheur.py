"""Train prox for a few dedicated steps."""

from ml_collections import ConfigDict

from configs.cifar10.prox_hybrid import get_config as get_base_config


def get_config():
    config = get_base_config()
    config.train.sample_t_lamb.step_candidates = (5, 10, 20)
    config.train.loss_on = "prox"  # compute loss on proximal, not epsilon
    config.model_type = "prox"  # network predicts proximal, not epsilon
    config.train.pm.pm_gamma_decay_stages = 1  # no decay
    config.fid.steps = (5, 10, 20)
    return config
