from ml_collections import ConfigDict

from configs.celeba_hq_vae.default import get_config as get_default_config


def get_config():
    config = get_default_config()

    config.model_class = "score"

    config.train.grad_clip = 1.0
    config.train.lr = ConfigDict()
    config.train.lr.warmup = 5000
    config.train.lr.base = 2e-5
    config.train.sample_every = 10_000
    config.train.sample_n_steps = (5, 10, 20, 50, 100, 1000)

    config.sample_method = config.fid.sample_method = "euler_maruyama_eps"

    return config
