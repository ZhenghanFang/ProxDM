"""Default config for CelebA-HQ"""

from ml_collections import ConfigDict


def get_config():
    config = ConfigDict()

    config.data = ConfigDict()
    config.data.dataset = "celeba_hq"
    config.data.image_size = 256
    config.data.channels = 3
    config.data.num_workers = 8

    config.net = ConfigDict()
    config.net.ch = 128
    config.net.ch_mult = [1, 2, 4, 4]
    config.net.attn = [1, 2, 3]
    config.net.num_res_blocks = 2
    config.net.dropout = 0.1
    config.net.in_ch = 4  # 4 for VAE
    config.net.use_checkpoint = False

    config.diffusion = ConfigDict()
    config.diffusion.beta_min = 0.1
    config.diffusion.beta_max = 20.0

    config.eval = ConfigDict()
    config.eval.fid_during_training = False
    config.eval.fid_after_training = ConfigDict()
    config.eval.fid_after_training.enabled = False
    config.eval.fid_after_training.all_checkpoints = True

    config.train = ConfigDict()
    config.train.resume = ""
    config.train.batch_size = 512  # total batch size
    config.train.total_iters = 1_500_000
    config.train.save_every = 50_000
    config.train.ema_decay = (0.9999,)

    config.model = ConfigDict()
    config.model.use_vae = True
    config.model.vae_latent_shape = (4, 32, 32)

    config.fid = ConfigDict()
    config.fid.n_samples = 1000
    config.fid.batch_size = 1000
    config.fid.save_samples = 100
    config.fid.stat_npz = "assets/fid_stats/celeba_hq_256.npz"
    config.fid.model_labels = ("ema0.9999",)
    config.fid.steps = tuple(sorted([5, 10, 20, 50, 100, 1000], reverse=True))
    config.fid.seed = 0

    return config
