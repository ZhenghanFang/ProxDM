"""Base config for MNIST"""

from ml_collections import ConfigDict


def get_config():
    config = ConfigDict()

    config.data = ConfigDict()
    config.data.dataset = "mnist"
    config.data.image_size = 32
    config.data.channels = 1
    config.data.num_workers = 8

    config.net = ConfigDict()
    config.net.ch = 64
    config.net.ch_mult = [1, 2, 2, 2]
    config.net.attn = [1]
    config.net.num_res_blocks = 2
    config.net.dropout = 0.1
    config.net.in_ch = 1

    config.diffusion = ConfigDict()
    config.diffusion.beta_min = 0.1
    config.diffusion.beta_max = 20.0

    config.fid = ConfigDict()
    config.fid.n_samples = 1000
    config.fid.batch_size = 1000
    config.fid.save_samples = 100
    config.fid.stat_npz = "assets/fid_stats/mnist.npz"
    config.fid.model_labels = (
        "ema0.9999",
    )  # use ("ema0.9999", "ema0.999", "base") for multiple models
    config.fid.steps = tuple(sorted([5, 10, 20, 50, 100, 1000], reverse=True))
    config.fid.seed = 0

    config.train = ConfigDict()
    config.train.resume = ""
    config.train.total_iters = 225_000
    config.train.save_every = 25_000
    config.train.ema_decay = (0.9999,)  # use (0.999, 0.9999) for multiple ema decays
    config.train.batch_size = 512  # total batch size across all GPUs

    return config
