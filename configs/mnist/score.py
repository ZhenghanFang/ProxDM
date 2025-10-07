from ml_collections import ConfigDict

from configs.mnist.base import get_config as get_base_config


def get_config():
    config = get_base_config()

    config.model_class = "score"

    config.train.lr = ConfigDict()
    config.train.lr.warmup = 0
    config.train.lr.base = 2e-4

    config.fid.sample_method = "euler_maruyama_eps"

    return config
